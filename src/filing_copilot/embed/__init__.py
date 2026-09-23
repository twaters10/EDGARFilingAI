"""Embeddings: the encoder seam, and the models behind it."""

from .cache import (
    PREFIX_SEPARATOR,
    CacheStats,
    EmbeddingCache,
    digest,
    embedding_input,
)
from .encoder import (
    SEARCH_DOCUMENT,
    SEARCH_QUERY,
    EmbeddingModel,
    Encoder,
    EncoderError,
    Task,
    l2_normalize,
    prepare,
    task_prefix,
    truncate_dimensions,
)
from .ollama_encoder import NOMIC_EMBED_TEXT, OllamaEncoder
from .pipeline import CorpusPlan, EmbedResult, document_input, embed_missing, plan_corpus

__all__ = [
    "NOMIC_EMBED_TEXT",
    "PREFIX_SEPARATOR",
    "SEARCH_DOCUMENT",
    "SEARCH_QUERY",
    "CacheStats",
    "CorpusPlan",
    "EmbedResult",
    "EmbeddingCache",
    "EmbeddingModel",
    "Encoder",
    "EncoderError",
    "OllamaEncoder",
    "Task",
    "digest",
    "document_input",
    "embed_missing",
    "embedding_input",
    "l2_normalize",
    "plan_corpus",
    "prepare",
    "task_prefix",
    "truncate_dimensions",
]
