# Architecture Decision Records

One file per decision. Each records the context, the decision, the alternatives that
lost, and the consequences — including the bad ones.

The point is that a reader (including a future me, or an interviewer) can see that a
choice was *made* rather than defaulted into.

| ADR | Decision | Stage |
|---|---|---|
| [0001](ADR-0001-filing-type-scope.md) | Filing type scope: 10-K/10-Q now, 8-K later, DEF 14A out | 1–2, 9 |
| [0002](ADR-0002-item-aware-chunking.md) | Item-aware chunking (sections, then windows within) | 2, 4 |
| [0003](ADR-0003-restatement-convention.md) | Restatement convention: as-restated, with a divergence flag | 1 |
| [0004](ADR-0004-fiscal-period-normalization.md) | Fiscal normalization: normalize *and* warn | 1 |
| [0005](ADR-0005-router-mechanism.md) | Router: LLM tool choice + deterministic guardrails | 5 |
| [0006](ADR-0006-change-detection-by-section-alignment.md) | Change detection by section alignment, not retrieval | 6 |
| [0007](ADR-0007-citation-enforcement-in-code.md) | Citation enforcement: four assertions, verbatim span check | 5 |
| [0008](ADR-0008-three-mcp-tools-not-two.md) | Three MCP tools, not two | 5 |
| [0009](ADR-0009-embedding-model-nomic-embed-text-v15.md) | Embedding model: nomic-embed-text-v1.5 | 3 |
| [0010](ADR-0010-reciprocal-rank-fusion-for-hybrid-search.md) | RRF for hybrid fusion | 4 |
| [0011](ADR-0011-local-first-development-aws-at-stage-8.md) | Local-first development, AWS at Stage 8 | 1–8 |
| [0012](ADR-0012-opensearch-serverless-nextgen-never-classic.md) | OpenSearch Serverless NextGen, never Classic | 8 |
| [0013](ADR-0013-cache-layout-mirrors-the-url-path.md) | Cache layout mirrors the URL path | 0 |
| [0014](ADR-0014-canonical-cik-form.md) | Canonical CIK form | 0 |
| [0015](ADR-0015-rate-limit-of-5-requests-per-second.md) | Rate limit of 5 req/s | 0 |
| [0016](ADR-0016-bulk-endpoints-over-per-company-api-calls.md) | Bulk endpoints; never extract companyfacts.zip | 0–1 |
| [0017](ADR-0017-cache-entries-never-expire.md) | Cache entries never expire | 0 |
