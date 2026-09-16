# ADR-0014: Canonical CIK form

**Status:** Accepted  
**Stage:** 0

## Context

SEC uses three renderings of the same Central Index Key. `company_tickers.json` gives `cik_str` as an **integer** despite the name (`1601712`); `data.sec.gov` wants `CIK0001601712`; `sec.gov/Archives/edgar/data/` wants unpadded `1601712`.

## Decision

Canonical internal form is a **10-digit zero-padded string with no prefix** (`"0001601712"`). `to_data_api_cik` and `to_archives_cik` convert outward at the boundary. Never pass a bare int across a module boundary.

## Alternatives considered

Passing CIKs around in whatever form SEC happened to return guarantees eventual mismatch.

## Consequences

Losing zero-padding is the single most likely bug in this area, and it surfaces as a 404 in Stage 1 — far from its cause. One normalization function with exhaustive tests makes that class of bug impossible.
