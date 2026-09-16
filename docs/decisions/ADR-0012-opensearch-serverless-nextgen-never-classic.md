# ADR-0012: OpenSearch Serverless NextGen, never Classic

**Status:** Accepted  
**Stage:** 8

## Context

A Classic Serverless collection bills a minimum OCU floor continuously: roughly $175/month with redundancy disabled, $350/month standard — **idle**. NextGen collections went GA 2026-05-28 with no minimum OCU and scale-to-zero after 10 minutes.

## Decision

Use **NextGen** collections only. The deploy script asserts collection type as a precondition, backed by an AWS Budget with an action at $3, both in place before the first deploy.

## Alternatives considered

Classic collections are 17-35x the entire project budget per month.

## Consequences

Idle cost falls from $175-350/month to about $0.03/month, which makes the whole project affordable within a $10 budget. Trade-off: a ~10-second cold start on the first query after idle — warm the collection before a live demo.

The trap: every tutorial written before mid-2026 creates a Classic collection. This assertion is the highest-leverage line of code in the repository.
