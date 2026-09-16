# ADR-0005: Router mechanism

**Status:** Accepted  
**Stage:** 5

## Context

Something must decide whether a question goes to SQL, to retrieval, or both.

## Decision

**LLM tool choice plus deterministic guardrails.** The tool description *is* the routing logic and is written as a contract. Rigor lives in the guardrails: resolver-gated refusal, and a validator that rejects any numeric claim not traceable to `query_financials`.

## Alternatives considered

A trained classifier needs labeled data that does not exist and fails on novel phrasings. Rules cannot handle the hybrid sequential archetype.

## Consequences

No training data needed, novel phrasings handled, and the safety property is enforced in code rather than hoped for in a prompt.
