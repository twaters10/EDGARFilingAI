# ADR-0016: Bulk endpoints over per-company API calls

**Status:** Accepted  
**Stage:** 0-1

## Context

XBRL facts can be fetched per-company from `data.sec.gov`, or in bulk as `companyfacts.zip` (~1.2GB).

## Decision

Use the bulk archive for corpus-wide ingestion. **Read members in place with `zipfile`; never extract.**

## Alternatives considered

Per-company calls scale linearly with corpus size and burn rate budget for no benefit.

## Consequences

One request instead of twenty, and far kinder to SEC. Extraction is not merely wasteful — the archive expands to roughly 15GB, which does not fit on the development machine. This constraint is recorded in CLAUDE.md so it is not rediscovered the hard way.
