# ADR-0013: Cache layout mirrors the URL path

**Status:** Accepted  
**Stage:** 0

## Context

The original plan specified `data/raw/{cik}/{accession}/`, which does not fit the data: `submissions.json` and `companyfacts.json` are company-level and have no accession, and `company_tickers.json` is global.

## Decision

Derive the cache path deterministically from the URL path:

```
data/raw/
├── reference/company_tickers.json
├── submissions/CIK0001601712.json
├── companyfacts/CIK0001601712.json
└── filings/1601712/000160171225000012/
```

Each entry gets a `.meta.json` sidecar recording source URL, fetch time, status, content type, ETag and a SHA-256 of the body.

## Alternatives considered

A hashed-filename cache is undebuggable at exactly the moment you need to debug it.

## Consequences

Satisfies 'keyed by URL' while staying readable — and these files get inspected constantly in Stages 1-2. The sidecar means a cached file can always be traced to its origin, which is what makes citations verifiable later.
