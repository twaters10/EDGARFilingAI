# ADR-0015: Rate limit of 5 requests per second

**Status:** Accepted  
**Stage:** 0

## Context

SEC permits up to 10 requests/second and states the limit is 'carefully monitored to preserve equitable access.' Being blocked mid-ingestion costs hours.

## Decision

Default to **5 req/s**, configurable via `EDGAR_RATE_LIMIT` but capped at SEC's 10 by validation. Every network request passes through the limiter, **retries included**.

## Alternatives considered

Running at the 10 req/s ceiling saves minutes and risks hours.

## Consequences

Headroom is free at this corpus size. Pacing retries matters: if retries bypassed the limiter, backoff would silently violate the ceiling under exactly the conditions that caused the failure. A test asserts retries consume rate budget.
