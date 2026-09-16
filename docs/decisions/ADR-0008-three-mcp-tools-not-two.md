# ADR-0008: Three MCP tools, not two

**Status:** Accepted  
**Stage:** 5

## Context

The original design exposed two tools: Athena and OpenSearch. Without entity resolution the model must invent CIKs and accession numbers, change detection has no way to identify which two filings to diff, and refusing a nonexistent company depends on the prompt.

## Decision

Add a third tool: `resolve_entity` / `list_filings`. Company name or ticker to CIK, and CIK to available filings with accession, form, period and available items.

## Alternatives considered

Two tools cannot support the change-detection or adversarial archetypes.

## Consequences

Refusal becomes deterministic — resolver returns empty, decline, in code. That is the difference between asking the model not to hallucinate and making it unable to.
