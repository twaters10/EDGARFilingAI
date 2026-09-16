# ADR-0017: Cache entries never expire

**Status:** Accepted  
**Stage:** 0

## Context

A cache needs a freshness policy. The corpus for this project is deliberately frozen: reproducibility of the demo and the eval set matters more than currency.

## Decision

**No TTL.** Entries are served indefinitely; `--refresh` forces a re-fetch. ETag is recorded in the sidecar but conditional requests are not implemented.

## Alternatives considered

A TTL would re-fetch a deliberately frozen corpus for no benefit.

## Consequences

Re-running ingestion makes zero network requests, which is what keeps CI offline and free. Deferring conditional requests is a recorded decision, not an oversight — if the corpus ever needs to track live filings, the ETag is already stored.
