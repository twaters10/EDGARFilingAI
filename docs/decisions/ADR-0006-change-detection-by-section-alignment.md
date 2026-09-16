# ADR-0006: Change detection by section alignment

**Status:** Accepted  
**Stage:** 6

## Context

'Which risk factors were added or removed versus the prior 10-K?' is the highest-value query pattern and the one plain top-k RAG handles worst. A retrieval system *structurally cannot* find a removal, because there is nothing to retrieve.

## Decision

Change detection **does not touch the search index**. Resolve two filings, extract Item 1A from each, split into risk-factor units by subheading, embed each unit, bipartite match across years on cosine similarity with a threshold. Unmatched-in-new = added; unmatched-in-old = removed; matched-below-high-threshold = materially reworded.

## Alternatives considered

Top-k RAG cannot find removals. Raw text diff drowns in boilerplate rewording.

## Consequences

Embeddings are used as a similarity function, not a search index. Removals surface symmetrically for free. This is a distinct code path, not a retrieval mode — treating it as one leads to building it wrong.
