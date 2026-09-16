# filing-copilot-mcp

## What this is

Research assistant over SEC EDGAR. Exact figures from XBRL via SQL; narrative from
hybrid retrieval. Portfolio project for senior DS/MLE interviews in financial services
(credit risk, card intelligence, model risk, AML). **Clarity counts as much as
correctness** — interviewers will read this code.

It is also a learning project. Prefer the path that makes mechanics visible over the
one that hides them, and say when you are making that tradeoff.

Full staged plan: `PROJECT_PLAN.txt`. Decisions: `docs/decisions/`.

## Working agreements

- **Never run `git commit`, `git push`, or any write to the remote.** Commits are the
  operator's alone. Say when work is ready; leave it for them.
- Git commit happens at the end of each stage — by the operator.
- Explicit and readable over clever.

## Inviolable constraints

- The LLM never does arithmetic on retrieved text. Figures come from
  `query_financials` or they do not appear.
- Every answer cites filing + section, enforced by `grounding/validator.py`, not
  requested in a prompt.
- Hybrid retrieval (BM25 + vector) is not optional.
- No hardcoded credentials, bucket names, or regions. `config.py` only.
- EDGAR: declared User-Agent with a real contact, ≤5 req/s, cache everything.

## This repository is PUBLIC

Anything committed is world-readable immediately. `.env` is git-ignored and must stay
that way; `tests/test_no_secrets.py` enforces it. Never put a real address or key in
`.env.example`.

## Architecture (decided — do not substitute)

Athena over S3 XBRL (DuckDB locally) · OpenSearch Serverless **NextGen** — never
Classic, which has a $175–350/mo idle floor · `nomic-embed-text-v1.5` via SageMaker
Processing · three MCP tools: `resolve_entity`, `query_financials`, `search_filings`.

## Scope

- **IN:** 10-K, 10-Q. Stage 9 adds 8-K items 4.01 / 4.02 / 5.02.
- **OUT:** DEF 14A — deliberate, see `docs/decisions/`. Do not add without asking.
- Corpus: ~20 companies × 3 years, designed to scale by config.

## Local-first

Stages 1–7 run on DuckDB + Docker OpenSearch. AWS is Stage 8, behind unchanged
interfaces. **CI never hits live AWS and makes zero API calls.**

## Conventions

Python 3.12, type hints throughout, `mypy --strict`. `black` + `ruff`, line length 100.
Tests use `httpx.MockTransport` — no network in CI. Live-network tests are marked
`@pytest.mark.network` and deselected by default.

## Environment

16GB M1 Pro. **Disk is the binding constraint, not cost.** Docker is not installed yet
(needed at Stage 3, ~4GB). Ollama holds `qwen2.5:14b` (Tier C judge), `llama3.1:8b`,
and `nomic-embed-text`.

## Known traps

- **`companyfacts.zip` extracts to ~15GB and will not fit.** Read the 20 CIKs you need
  directly out of the archive with `zipfile`. Never extract it.
- **`cik_str` in `company_tickers.json` is an int despite the name.** There are three
  CIK renderings — canonical internal form is 10-digit zero-padded, no prefix. See
  `edgar/identifiers.py`. Losing padding shows up as a 404 far from its cause.
- **`nomic-embed-text-v1.5` requires task prefixes** (`search_document:` /
  `search_query:`). Omitting them degrades retrieval *silently*. Verify whether Ollama
  applies them for you before embedding 90k chunks.
- **"Total debt" is not an XBRL tag.** See `structured/concepts.yaml` (Stage 1).
- **Global top-k gives concentration, not coverage.** Peer questions use
  `retrieval/strategies.py:fan_out_peers`.
- **Change detection does not use the search index.** It is set alignment.
- **Retries must consume rate budget.** Never wrap the throttle in a retry loop.
- `www.sec.gov` and `data.sec.gov` are different hosts with different CIK renderings.
  Every URL lives in `edgar/endpoints.py`.
