# ADR-0003: Restatement convention

**Status:** Accepted  
**Stage:** 1

## Context

The same fiscal period has different values depending on which filing reports it. 'What was 2023 revenue?' has more than one correct answer: as originally reported, or as restated.

## Decision

Default to **as-restated** (latest `filed` for the period). Every response carries a `basis` field, and a `restated: true` flag when the two bases differ.

## Alternatives considered

As-reported-only answers a different question than most users ask. Returning both unconditionally makes every simple query noisy.

## Consequences

Colloquial questions get the answer people mean. Silently picking one convention was the real failure mode; the divergence flag is the actual deliverable here.
