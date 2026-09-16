# filing-copilot-mcp

A research assistant over SEC EDGAR filings that answers questions requiring **both**
exact figures from structured financial data and narrative explanation from filing
text — routed to the right tool, and grounded in citations.

The flagship query shape:

> *"Synchrony's allowance for credit losses increased in 2024 — by how much, and what
> did management say drove it?"*

The number comes from structured XBRL via SQL. The explanation comes from hybrid
retrieval over filing text. **The model never does arithmetic on retrieved text.**

## Status

**Stage 0 of 10 — foundations and EDGAR citizenship.** The full staged plan is in
[`PROJECT_PLAN.txt`](PROJECT_PLAN.txt).

What works today: ticker→CIK resolution and a throttled, cached, retrying EDGAR client.

## Quick start

```bash
brew install uv
uv sync

cp .env.example .env
# Edit .env: EDGAR_USER_AGENT must contain a real contact address.
# SEC requires it, and the app refuses to start with the placeholder.

uv run fc resolve --ticker SYF
uv run fc fetch-submissions --ticker SYF   # MISS — fetches
uv run fc fetch-submissions --ticker SYF   # HIT  — zero network requests
uv run fc cache-info
```

## Architecture

| Layer | Choice |
|---|---|
| Structured | Amazon Athena over S3-stored XBRL (DuckDB locally) |
| Unstructured | OpenSearch Serverless **NextGen**, hybrid BM25 + vector |
| Embeddings | `nomic-embed-text-v1.5` via SageMaker Processing |
| Interface | MCP server: `resolve_entity`, `query_financials`, `search_filings` |
| Grounding | Citations enforced in code, not requested in a prompt |

## Development

```bash
uv run pytest                  # offline; network tests deselected by default
uv run pytest -m network       # opt in to live SEC calls
uv run mypy src                # strict
uv run ruff check . && uv run black --check .
```

## SEC EDGAR etiquette

This client declares a `User-Agent` with a real contact address and paces itself at
**5 requests/second** — SEC permits 10 and monitors the limit, so we leave headroom.
Every response is cached on disk; re-running ingestion makes zero network requests.

See [`docs/decisions/`](docs/decisions/) for the reasoning behind these and other
choices.

## License

Personal portfolio project. Filing data is public domain, courtesy of SEC EDGAR.
