# ADR-0001: Filing type scope

**Status:** Accepted  
**Stage:** 1-2, revisited at 9

## Context

Periodic reports (10-K/10-Q) carry five of six non-adversarial query archetypes. But executive compensation, sudden executive departures, restatements (8-K Item 4.02) and auditor changes (Item 4.01) are unanswerable from them, which limits the red-flag screening archetype to Item 9A material weakness and going-concern language.

## Decision

Ingest 10-K and 10-Q now. Add 8-K items 4.01, 4.02 and 5.02 as a bounded Stage 9. **Explicitly de-scope DEF 14A** and document why.

## Alternatives considered

Full coverage including DEF 14A was rejected: proxy parsing is table-heavy and substantially harder, and serves a persona that is not primary. Realistically a week for little interview value.

## Consequences

Red-flag screening is partial until Stage 9. DEF 14A absence is a stated decision rather than an oversight, which is the point.
