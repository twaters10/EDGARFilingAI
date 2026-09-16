# ADR-0009: Embedding model: nomic-embed-text-v1.5

**Status:** Accepted  
**Stage:** 3

## Context

The corpus is long-form filing prose. Open weights are required so embedding stays free and reproducible.

## Decision

Use `nomic-embed-text-v1.5`.

## Alternatives considered

`mxbai-embed-large-v1` is comparable but has a shorter context and no Matryoshka property.

## Consequences

8192-token context suits filing prose. Matryoshka truncation allows trading dimensions for index size later **without re-embedding**. Cost: the model requires task prefixes (`search_document:` / `search_query:`) and omitting them degrades retrieval *silently* — a test asserts they are applied.
