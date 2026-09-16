# ADR-0010: Reciprocal Rank Fusion for hybrid search

**Status:** Accepted  
**Stage:** 4

## Context

Hybrid retrieval must combine BM25 and vector results. BM25 scores and cosine similarities live on incomparable scales.

## Decision

Fuse with **Reciprocal Rank Fusion**, k=60: score each document as the sum of 1/(k + rank) across result lists.

## Alternatives considered

Weighted score blending requires normalizing incomparable scales and invites premature tuning by eyeballing results.

## Consequences

Rank-based, so no score normalization is needed — which is the entire point. Fewer knobs to overtune before an eval set exists. Weight tuning is forbidden until Tier A metrics exist.
