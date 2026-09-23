"""The encoder seam, and the trap it exists to close.

`PROJECT_PLAN.txt` names one specific failure for this stage: nomic's task
prefixes degrade retrieval *silently* when omitted. No error, just quietly worse
results. So the prefix assertions here are the point of the file, and they run
offline against a pure function -- a test that needed a server would be skipped
in CI, which is exactly when the regression would land.

The network-marked tests at the bottom pin the facts this project measured on its
own Ollama build rather than taking the model card's word for them. They are
deselected by default, like every other live test in this suite.
"""

from __future__ import annotations

import httpx
import pytest

from filing_copilot.embed.encoder import (
    SEARCH_DOCUMENT,
    SEARCH_QUERY,
    EmbeddingModel,
    EncoderError,
    l2_normalize,
    prepare,
    task_prefix,
    truncate_dimensions,
)
from filing_copilot.embed.ollama_encoder import NOMIC_EMBED_TEXT, OllamaEncoder

HOST = "http://localhost:11434"
CHUNK = "We face substantial credit risk in our credit card portfolio."

# A model that handles prefixes itself. Double-prefixing one of these is the
# mirror image of the trap, and just as silent.
PREFIX_FREE = EmbeddingModel(
    name="already-prefixed", dimensions=4, needs_task_prefix=False, max_input_chars=1000
)


def encoder(
    handler: object, *, model: EmbeddingModel = NOMIC_EMBED_TEXT
) -> tuple[OllamaEncoder, list[httpx.Request]]:
    """An encoder backed by a scripted transport. Returns the captured requests."""
    seen: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert callable(handler)
        return handler(request)

    return (
        OllamaEncoder(
            host=HOST, model=model, client=httpx.Client(transport=httpx.MockTransport(capture))
        ),
        seen,
    )


def vector(*values: float) -> object:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": list(values)})

    return handler


def sent_prompt(request: httpx.Request) -> str:
    import json

    return str(json.loads(request.content)["prompt"])


# --- the trap -----------------------------------------------------------------


def test_document_text_carries_the_search_document_prefix() -> None:
    """Omitting this degrades retrieval with no error. That is why it is asserted."""
    assert prepare(NOMIC_EMBED_TEXT, SEARCH_DOCUMENT, CHUNK) == f"search_document: {CHUNK}"


def test_query_text_carries_the_search_query_prefix() -> None:
    assert prepare(NOMIC_EMBED_TEXT, SEARCH_QUERY, CHUNK) == f"search_query: {CHUNK}"


def test_documents_and_queries_are_prefixed_differently() -> None:
    """The model is trained asymmetrically; using one prefix for both loses that."""
    doc = prepare(NOMIC_EMBED_TEXT, SEARCH_DOCUMENT, CHUNK)
    query = prepare(NOMIC_EMBED_TEXT, SEARCH_QUERY, CHUNK)
    assert doc != query


def test_the_prefix_reaches_the_server() -> None:
    """The pure function is right; assert the encoder actually uses it."""
    enc, seen = encoder(vector(1.0, 0.0, 0.0))
    enc.encode([CHUNK], task=SEARCH_DOCUMENT)
    assert sent_prompt(seen[0]) == f"search_document: {CHUNK}"


def test_a_query_reaches_the_server_with_the_query_prefix() -> None:
    enc, seen = encoder(vector(1.0, 0.0, 0.0))
    enc.encode([CHUNK], task=SEARCH_QUERY)
    assert sent_prompt(seen[0]) == f"search_query: {CHUNK}"


def test_a_model_that_needs_no_prefix_is_not_double_prefixed() -> None:
    """The mirror of the trap: prefixing a model that prefixes itself."""
    assert prepare(PREFIX_FREE, SEARCH_DOCUMENT, CHUNK) == CHUNK
    assert task_prefix(PREFIX_FREE, SEARCH_DOCUMENT) == ""


# --- normalization ------------------------------------------------------------


def test_vectors_leave_the_encoder_normalized() -> None:
    """Ollama returns L2 ~20; OpenSearch cosine assumes unit length."""
    enc, _ = encoder(vector(3.0, 4.0, 0.0, 0.0), model=_dims(4))
    [got] = enc.encode([CHUNK], task=SEARCH_DOCUMENT)
    assert pytest.approx(sum(v * v for v in got), abs=1e-9) == 1.0
    assert pytest.approx(got[0], abs=1e-9) == 0.6


def test_l2_normalize_leaves_a_zero_vector_alone() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_truncation_keeps_the_leading_dimensions_and_renormalizes() -> None:
    """Matryoshka: a truncated vector is no longer unit length until renormalized."""
    got = truncate_dimensions([3.0, 4.0, 99.0, 99.0], 2)
    assert got == pytest.approx([0.6, 0.8])


def test_truncation_beyond_the_vector_is_a_no_op_but_still_normalizes() -> None:
    got = truncate_dimensions([3.0, 4.0], 768)
    assert got == pytest.approx([0.6, 0.8])


def test_the_configured_dimensionality_is_what_comes_back() -> None:
    enc, _ = encoder(vector(*([1.0] * 768)), model=_dims(256))
    [got] = enc.encode([CHUNK], task=SEARCH_DOCUMENT)
    assert len(got) == 256


# --- refusing rather than truncating ------------------------------------------


def test_oversized_input_is_refused_before_it_is_sent() -> None:
    """A silently truncated chunk embeds its opening and cites its whole span."""
    enc, seen = encoder(vector(1.0))
    with pytest.raises(EncoderError, match="over the"):
        enc.encode(["x" * (NOMIC_EMBED_TEXT.max_input_chars + 1)], task=SEARCH_DOCUMENT)
    assert seen == []


def test_the_prefix_counts_toward_the_limit() -> None:
    enc, _ = encoder(vector(1.0))
    just_over = "x" * (NOMIC_EMBED_TEXT.max_input_chars - len("search_document: ") + 1)
    with pytest.raises(EncoderError):
        enc.encode([just_over], task=SEARCH_DOCUMENT)


# --- failure surfaces ---------------------------------------------------------


def test_an_unreachable_server_names_the_fix() -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    enc, _ = encoder(refuse)
    with pytest.raises(EncoderError, match="ollama serve"):
        enc.encode([CHUNK], task=SEARCH_DOCUMENT)


def test_a_server_error_is_reported_with_its_status() -> None:
    enc, _ = encoder(lambda _: httpx.Response(500, text="out of memory"))
    with pytest.raises(EncoderError, match="500"):
        enc.encode([CHUNK], task=SEARCH_DOCUMENT)


def test_an_empty_embedding_is_an_error_not_an_empty_vector() -> None:
    enc, _ = encoder(lambda _: httpx.Response(200, json={"embedding": []}))
    with pytest.raises(EncoderError, match="empty embedding"):
        enc.encode([CHUNK], task=SEARCH_DOCUMENT)


def test_encoding_preserves_input_order() -> None:
    order = iter([[1.0, 0.0], [0.0, 1.0]])
    enc, _ = encoder(lambda _: httpx.Response(200, json={"embedding": next(order)}), model=_dims(2))
    assert enc.encode(["a", "b"], task=SEARCH_DOCUMENT) == [[1.0, 0.0], [0.0, 1.0]]


# --- model identity -----------------------------------------------------------


def test_the_cache_key_separates_models() -> None:
    assert _dims(768).cache_key != _dims(256).cache_key


def test_the_cache_key_separates_dimensionalities() -> None:
    """Switching 768 -> 256 must not serve stale full-width vectors."""
    assert _dims(256).cache_key == "nomic-embed-text@256"


def _dims(n: int) -> EmbeddingModel:
    return EmbeddingModel(
        name="nomic-embed-text",
        dimensions=n,
        needs_task_prefix=True,
        max_input_chars=NOMIC_EMBED_TEXT.max_input_chars,
    )


# --- live, deselected by default ----------------------------------------------


@pytest.mark.network
def test_ollama_does_not_apply_the_task_prefix_for_us() -> None:
    """The measurement D9's trap warning depends on.

    If this ever fails, Ollama has started applying prefixes itself and
    ``needs_task_prefix`` must flip -- otherwise every chunk is double-prefixed.
    """
    import math

    with OllamaEncoder(host=HOST) as enc:
        bare = enc.encode([CHUNK], task=SEARCH_DOCUMENT)[0]
        # Pass the prefix in as content; if Ollama stripped or applied prefixes
        # itself, this would be indistinguishable from the bare text.
        doubled = enc.encode([f"search_document: {CHUNK}"], task=SEARCH_DOCUMENT)[0]

    cosine = sum(a * b for a, b in zip(bare, doubled, strict=True))
    assert not math.isclose(
        cosine, 1.0, abs_tol=1e-6
    ), "Ollama appears to normalize away the task prefix; re-verify needs_task_prefix."


@pytest.mark.network
def test_live_vectors_are_768_dimensional_and_unit_length() -> None:
    with OllamaEncoder(host=HOST) as enc:
        [got] = enc.encode([CHUNK], task=SEARCH_DOCUMENT)
    assert len(got) == 768
    assert pytest.approx(sum(v * v for v in got), abs=1e-6) == 1.0
