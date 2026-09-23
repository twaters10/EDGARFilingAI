"""Ollama-backed encoder -- the local implementation of :class:`Encoder`.

Ollama embeds one prompt per request; there is no batch endpoint on
``/api/embeddings``. At 14,043 chunks that is 14,043 requests to a local server,
which takes minutes, not hours. Batching is therefore a Stage 8 concern (a
SageMaker Processing job over the same pure functions), not a reason to complicate
this.

What this module does *not* do is decide policy. Prefixing, normalization and
truncation all live in :mod:`.encoder` as pure functions, so the SageMaker wrapper
in Stage 8 inherits identical behaviour rather than a second implementation of it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import TracebackType

import httpx

from .encoder import (
    EmbeddingModel,
    EncoderError,
    Task,
    l2_normalize,
    prepare,
    truncate_dimensions,
)

# Measured on this project's Ollama build (see tests/test_encoder.py and the
# module docstring in .encoder). Input is honoured to ~12,000 characters and the
# server returns HTTP 500 above ~16,000, so the ceiling sits below the observed
# failure point. An 800-token chunk plus its contextual prefix is ~3,300
# characters, so this is headroom rather than a constraint.
NOMIC_MAX_INPUT_CHARS = 12_000

NOMIC_EMBED_TEXT = EmbeddingModel(
    name="nomic-embed-text",
    dimensions=768,
    # Verified, not assumed: cos(bare, "search_document: " + text) = 0.862 on
    # this build, so the prefix reaches the model as content and is ours to add.
    needs_task_prefix=True,
    max_input_chars=NOMIC_MAX_INPUT_CHARS,
)


class OllamaEncoder:
    """Embeds through a local Ollama server.

    Satisfies :class:`~.encoder.Encoder`. Nothing above this class knows the
    transport, which is what makes the Stage 8 swap a constructor change.
    """

    def __init__(
        self,
        *,
        host: str,
        model: EmbeddingModel = NOMIC_EMBED_TEXT,
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout_seconds)

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    def encode(self, texts: Sequence[str], *, task: Task) -> list[list[float]]:
        """Embed ``texts`` for ``task``. See :meth:`Encoder.encode`."""
        return [self._encode_one(text, task) for text in texts]

    def _encode_one(self, text: str, task: Task) -> list[float]:
        prompt = prepare(self._model, task, text)

        if len(prompt) > self._model.max_input_chars:
            # Refuse rather than let the server truncate or fail obscurely. A
            # silently truncated chunk embeds its opening and cites its whole
            # span, which is a citation that validates and does not mean what it
            # says.
            raise EncoderError(
                f"Input is {len(prompt):,} characters, over the "
                f"{self._model.max_input_chars:,} limit for {self._model.name}. "
                f"Shorten the chunk (see chunkers.TARGET_TOKENS) rather than "
                f"raising this limit -- the server refuses above ~16,000."
            )

        try:
            response = self._client.post(
                f"{self._host}/api/embeddings",
                json={"model": self._model.name, "prompt": prompt},
            )
        except httpx.HTTPError as exc:
            raise EncoderError(
                f"Could not reach Ollama at {self._host}: {exc}\n"
                f"Is it running? `ollama serve`, then `ollama pull {self._model.name}`."
            ) from exc

        if response.status_code >= 400:
            raise EncoderError(
                f"Ollama returned {response.status_code} embedding "
                f"{len(prompt):,} characters with {self._model.name}: "
                f"{response.text[:200]}"
            )

        try:
            raw = response.json()["embedding"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EncoderError(f"Ollama response had no 'embedding' field: {exc}") from exc

        if not raw:
            raise EncoderError(f"Ollama returned an empty embedding for {self._model.name}.")

        # Ollama returns unnormalized vectors (L2 ~20 on this build). Truncation
        # re-normalizes; when no truncation is wanted it still normalizes, so
        # every vector leaving an encoder is unit length.
        return truncate_dimensions(raw, self._model.dimensions)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaEncoder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["NOMIC_EMBED_TEXT", "NOMIC_MAX_INPUT_CHARS", "OllamaEncoder", "l2_normalize"]
