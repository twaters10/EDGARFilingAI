# ADR-0002: Item-aware chunking

**Status:** Accepted  
**Stage:** 2, settled by measurement in 4

## Context

10-Ks have a standardized item structure. Chunking on item boundaries should make both diffing and targeted queries better than fixed token windows. But Item 1A can run 30k tokens, so 'one chunk per item' is not viable.

## Decision

Hybrid: **item boundaries determine sections; sections carry metadata; token-window chunking with overlap happens within section.** Settle against pure fixed-window by A/B on the eval set rather than by assertion.

## Alternatives considered

Pure fixed-window loses section metadata. Pure item chunks are too large to retrieve.

## Consequences

Chunks carry section provenance, which is what makes change detection and section-targeted retrieval possible at all. The recommendation is measured, not asserted.
