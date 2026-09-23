"""The embedding surface: what a model is, and what an encoder must do.

This protocol is to Stage 3 what :mod:`filing_copilot.structured.backend` is to
Stage 1 -- the seam that contains a later substitution. Ollama runs locally now,
SageMaker runs the identical encoder in Stage 8, and a different model entirely
may win the Stage 4 A/B. The only thing that should have to change is which
class is constructed.

Three properties are enforced here rather than left to each implementation, because
every one of them fails *silently* when a caller forgets it:

1. **Task prefixes.** ``nomic-embed-text`` is trained asymmetrically: corpus text
   is embedded as ``search_document: ...`` and questions as ``search_query: ...``.
   Omitting them produces no error and quietly worse retrieval.

   Measured against this project's Ollama build, the prefixes are **not** applied
   for us -- ``cos(bare, "search_document: " + text)`` is 0.862, so the prefix
   reaches the model as ordinary content. That is verified in
   ``tests/test_encoder.py`` rather than assumed, and it is why
   :class:`EmbeddingModel` carries ``needs_task_prefix``: a model that does apply
   them must not be double-prefixed.

2. **Normalization.** Ollama returns vectors with an L2 norm around 20, not 1.
   OpenSearch's cosine similarity and any dot-product shortcut both assume unit
   vectors, so normalization happens here, once, for every implementation.

3. **Model identity travels with the vectors.** :attr:`EmbeddingModel.cache_key`
   is what stops a cached vector from one model being served for another, or a
   768-dimension vector being reused after a switch to 256.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

#: Embedding a corpus chunk, versus embedding a question. Not interchangeable:
#: the model is trained to place them in the same space from different sides.
Task = Literal["search_document", "search_query"]

SEARCH_DOCUMENT: Task = "search_document"
SEARCH_QUERY: Task = "search_query"


class EncoderError(RuntimeError):
    """Embedding failed. Carries a message meant for a human."""


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Everything about a model that changes its output.

    Every field here is part of :attr:`cache_key`, because every field here can
    change a vector. A cache keyed on text alone would serve a 256-dimension
    vector to a 768-dimension index without complaining.
    """

    name: str
    dimensions: int
    needs_task_prefix: bool
    """Whether the caller must prepend ``search_document:`` / ``search_query:``.

    True for the nomic family. Verified per deployment -- a model that applies
    prefixes internally would be double-prefixed if this were set wrongly.
    """
    max_input_chars: int
    """Refuse above this rather than let the server truncate or 500.

    Measured on this Ollama build: input is honoured to ~12,000 characters and
    returns HTTP 500 above ~16,000. The limit is deliberately below the observed
    failure point, and refusing loudly is the point -- a silently truncated chunk
    embeds its first half and cites its whole span.
    """

    @property
    def cache_key(self) -> str:
        """Identity of this model for an embedding cache. See the class docstring."""
        return f"{self.name}@{self.dimensions}"


def task_prefix(model: EmbeddingModel, task: Task) -> str:
    """The prefix to prepend for ``task``, or ``""`` when the model needs none."""
    return f"{task}: " if model.needs_task_prefix else ""


def prepare(model: EmbeddingModel, task: Task, text: str) -> str:
    """The exact string sent to the model.

    Kept pure and separate from any transport so a test can assert the prefix is
    applied without a running server -- which is the whole defence against the
    Stage 3 trap.
    """
    return f"{task_prefix(model, task)}{text}"


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Scale to unit length. A zero vector is returned unchanged, not divided by zero."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


def truncate_dimensions(vector: Sequence[float], dimensions: int) -> list[float]:
    """Matryoshka truncation: keep the leading ``dimensions`` and re-normalize.

    nomic-embed-text-v1.5 is trained so leading dimensions carry most of the
    signal, which is what makes 768 -> 256 a size/quality trade rather than
    corruption. Re-normalizing is not optional: a truncated vector is no longer
    unit length, and cosine similarity against full-length vectors would be
    scaled by an arbitrary constant.

    Whether *this* Ollama build is v1.5 is unverified -- its manifest does not
    say -- so treat truncation as unproven until the Stage 4 A/B measures it.
    """
    if dimensions >= len(vector):
        return l2_normalize(vector)
    return l2_normalize(vector[:dimensions])


class Encoder(Protocol):
    """Turns text into vectors. Implemented by Ollama now, SageMaker later.

    Implementations are responsible for applying :func:`prepare` and returning
    normalized vectors of exactly ``model.dimensions`` length. They are *not*
    responsible for caching -- that sits above, keyed on
    :attr:`EmbeddingModel.cache_key`, so swapping an encoder cannot invalidate
    the wrong entries.
    """

    @property
    def model(self) -> EmbeddingModel:
        """Which model this encoder speaks to, and at what dimensionality."""
        ...

    def encode(self, texts: Sequence[str], *, task: Task) -> list[list[float]]:
        """Embed ``texts`` for ``task``, in order.

        Returns one unit vector per input. ``task`` is required rather than
        defaulted: a caller that has to name it cannot forget that documents and
        queries are embedded differently.

        Raises:
            EncoderError: on transport failure, or when an input exceeds
                ``model.max_input_chars``.
        """
        ...

    def close(self) -> None:
        """Release any underlying resources."""
        ...
