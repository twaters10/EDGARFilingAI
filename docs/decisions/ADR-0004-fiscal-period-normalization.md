# ADR-0004: Fiscal period normalization

**Status:** Accepted  
**Stage:** 1

## Context

Apple's FY2024 ends in September; most companies' end in December. 'Compare FY2024 revenue across five companies' is a trap that produces a confidently wrong answer.

## Decision

Do **both**. Derive `cal_year`/`cal_quarter` by the calendar year containing the majority of the period's span, and emit a `fiscal_alignment_warning` when compared entities' period ends differ by more than 45 days.

## Alternatives considered

Relabeling alone is a lie for a September filer. Warning alone makes every cross-company query a manual exercise.

## Consequences

Cross-company comparison is possible without being silently misleading. TTM recomposition from quarterly data is noted as a stretch goal, not built.
