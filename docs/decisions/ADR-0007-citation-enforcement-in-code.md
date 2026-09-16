# ADR-0007: Citation enforcement in code

**Status:** Accepted  
**Stage:** 5

## Context

'Every answer must cite a filing and section' is worthless if enforcement is a prompt instruction, and nearly worthless if enforcement only checks that a citation exists.

## Decision

Four assertions, enforced by a validator with one repair loop: (1) every cited `chunk_id` was actually in context; (2) every sentence containing a figure carries a citation; (3) Athena-sourced figures carry accession + tag + period; (4) **every quoted span appears verbatim in the cited chunk** (whitespace-normalized substring match). On repeated failure, degrade to grounded-facts-only or refuse.

## Alternatives considered

Prompt-only enforcement is unverifiable. Marker-presence checking passes trivially on fabricated citations.

## Consequences

Assertion (4) is what actually catches hallucination, and it is roughly 40 lines of code. Citation presence alone is theater.
