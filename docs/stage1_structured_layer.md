# Stage 1 — Structured layer: XBRL → Parquet → SQL

> **Maintained by hand.** This document is not regenerated when code changes. It
> describes the state of the code as of **2026-09-17** and is updated only on
> request. If a number here disagrees with `fc coverage`, trust `fc coverage`.

Answer any pure-structured question correctly, with the restatement and
fiscal-period traps handled explicitly.

This is the half of the system that never uses retrieval. Figures come from SQL
over XBRL or they do not appear — the LLM is never asked to do arithmetic on
retrieved text. Everything here exists to make "what was revenue in 2024?" have
one defensible answer instead of several plausible ones.

---

## The pipeline

```text
   companyfacts.zip   ~1.2 GB, 1 member per company
        │             NEVER extracted — it expands to roughly 15 GB
        │             members are read in place with zipfile  (ADR-0016)
        ▼
╔═══ 1 · FLATTEN ══════════════════════════════ structured/flatten.py ═══╗
║                                                                        ║
║   facts → taxonomy → tag → units → unit → [observation, ...]           ║
║        four levels deep, one company per member                        ║
║                              │                                         ║
║                              ▼                                         ║
║        one row per fact-observation, wide and boring:                  ║
║        cik taxonomy tag unit period_type start end period_year         ║
║        val fy fp form filed accn frame                                 ║
║                                                                        ║
║   every hard question downstream becomes a SQL filter once it is flat  ║
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼
╔═══ 2 · PARTITION ═════════════════════════════ structured/ingest.py ═══╗
║                                                                        ║
║   data/processed/facts/cik=0001601712/period_year=2024/*.parquet       ║
║                                                                        ║
║   partition on period_year — derived from the fact's OWN period —      ║
║   never on SEC's `fy`, which describes the REPORT that carried it      ║
║                                                                        ║
║   re-running REPLACES a company's partitions rather than appending     ║
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼   763,427 rows · 3,778 tags · 74 units
╔═══ 3 · RESOLVE ═════════════════════════════ structured/resolver.py ═══╗
║                                                                        ║
║   "total_revenue"  ──►  concepts.yaml  ──►  ordered candidate tags     ║
║                                              │                         ║
║                              first_available ▼                         ║
║                         first tag this company actually reports        ║
║                                              │                         ║
║                         records WHICH tag answered ──► never hidden    ║
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼
╔═══ 4 · QUERY ═════════════════════════════ structured/financials.py ═══╗
║                                                                        ║
║   filter on unit AND period_type                                       ║
║   select on the period's own dates, never on fy                        ║
║   pick a basis: as_restated (default) | as_reported                    ║
║   compare period ends across companies ──► alignment warning           ║
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼
                    fc financials  /  Stage 5's query_financials
```

---

## 1 · Flatten — three things the real data taught us

```text
   ┌─ not every observation has a `start` ──────────────────────────────┐
   │    balance-sheet facts are INSTANTS at a single date               │
   │    income-statement facts are DURATIONS over a span                │
   │    period_type records which, because the fiscal-labelling rule    │
   │    differs between them                                            │
   └────────────────────────────────────────────────────────────────────┘

   ┌─ `fy`/`fp` describe the REPORT, not the fact ──────────────────────┐
   │    Synchrony's earliest row ends 2013-12-31 but carries            │
   │    fy=2014, fp=Q2 — a prior-period comparative inside a later 10-Q │
   │                                                                    │
   │    partitioning on fy would scatter one period across partitions   │
   │    → partition on period_year, keep fy/fp as ordinary columns      │
   └────────────────────────────────────────────────────────────────────┘

   ┌─ units are free-form ──────────────────────────────────────────────┐
   │    USD · shares · pure · business · putative_class_actio           │
   │    74 distinct units in this corpus                                │
   │                                                                    │
   │    a query that does not filter on unit can add dollars to         │
   │    lawsuit counts → unit is never optional                         │
   └────────────────────────────────────────────────────────────────────┘
```

The Parquet schema is declared explicitly, not inferred. An inferred schema can
change type between runs when a column happens to be all-null for one company.

---

## 2 · The concept-mapping layer

**"Total debt" is not an XBRL tag.** One company reports `LongTermDebt`, another
`LongTermDebtNoncurrent`, a third `LongTermDebtAndCapitalLeaseObligations`. A
query written against any single tag silently returns nothing for most of a peer
set.

```text
   concepts.yaml  (declarative — adding a concept is a data change)
        │
        ├─ label          "Total revenue"
        ├─ rule           first_available
        ├─ applies_to     all | lenders     ← scopes the coverage threshold
        ├─ unit           USD               ← part of the concept, not a filter
        ├─ period_type    duration          ← ditto
        └─ candidates     ordered, EXPLICIT allow-list
                            - RevenuesNetOfInterestExpense
                            - Revenues
                            - ...
                               │
                               ▼
              ┌────────────────┬───────────────┐
              │ does THIS company report it?   │──NO──► next candidate
              └────────────────┬───────────────┘
                              YES
                               ▼
                 resolved — the rank and tag are recorded
                 rank 0 = preferred · rank > 0 = a fallback, reported as such
```

Two rules that look like detail and are not:

- **Allow-lists, never patterns.** A substring match on `Debt` pulls in
  `AvailableForSaleDebtSecurities*` — investment *assets*, not money owed.
  Pattern matching here would report a bank's bond portfolio as its debt.
- **`applies_to` scopes the threshold.** A card network reports no allowance for
  credit losses. Counting that as a miss measures nothing and pressures the
  corpus toward 20 identical lenders (ADR-0018).

---

## 3 · The two traps

### Restatement — one period has more than one reported value

```text
   a figure for FY2023 appears in the FY2023 10-K,
   then again as a comparative in FY2024's, sometimes restated

   observations for one (cik, concept, period)
        │
        ├── earliest `filed`  ──► as_reported    what they said at the time
        └── latest `filed`    ──► as_restated    what they say now  ← DEFAULT
                                                                    (ADR-0003)

   the response ALWAYS carries `basis`,
   and `restated: true` when the two differ

   picking one silently is the failure mode
```

### Fiscal alignment — the same "FY2024" is not the same period

```text
   Visa        FY2024 ended 2024-09-30
   Mastercard  FY2024 ended 2024-12-31
                        │
                        ▼
          period ends differ by > 45 days?
                        │
                       YES ──► fiscal_alignment_warning, surfaced to the user
                        │
                        NO  ──► comparable  (45 days tolerates 52/53-week drift)
```

Comparing them without saying so is the trap ADR-0004 exists to prevent.

---

## 4 · The backend protocol is deliberately narrow

```text
   ┌───────────────────────────────────────────────────┐
   │  SqlBackend  (Protocol)     structured/backend.py │
   │                                                   │
   │    tags_present(...)                              │
   │    observations(...)                              │
   │    fiscal_year_end(cik, period_year)              │
   │    close()                                        │
   └───────────────┬──────────────────┬────────────────┘
                   │                  │
                   ▼                  ▼
          DuckDBBackend          AthenaBackend
          Stages 1–7             Stage 8
          local Parquet          Parquet on S3

   Four methods. If the caller builds its own SQL, the Stage 8 port
   stops being one new class and becomes a rewrite.
```

---

## Commands

```bash
fc fetch-companyfacts                          # the bulk archive (Stage 0)
fc ingest-facts                                # flatten → Parquet
fc coverage --from 2022 --to 2025              # the concept coverage matrix
fc financials --concept total_revenue \
              --ticker SYF --ticker COF --fy 2024
```

`fc coverage` exits 1 when a concept does not resolve for every expected company.
That is a finding, not necessarily a bug — companyfacts excludes company custom
extension tags.

---

## Where it stands

```
fact table   763,427 rows · 3,778 distinct tags · 74 distinct units
             398 Parquet files · 9.9 MiB · cik=/period_year= partitions

coverage matrix, period years 2022–2025
  total_revenue                  20/20 all       5 tags used
  total_debt                     19/20 all       3 tags used   MISSING: JPM
  allowance_for_credit_losses    16/17 lenders   3 tags used   MISSING: AXP
  total_assets                   20/20 all       1 tag
  net_income                     20/20 all       2 tags
```

Against ADR-0018's thresholds — `total_revenue` and `total_debt` ≥18/20,
`allowance_for_credit_losses` ≥14/17 lenders — **all three pass.**

The two gaps are real findings, not parser failures. JPMorgan and American
Express report those concepts under custom extension tags, which SEC's
companyfacts API excludes by design. No candidate list can recover them; the
honest output is a named absence.

The `tags used` counts are the argument for the whole mapping layer: five
different tags answer "total revenue" across twenty companies. A query written
against `Revenues` alone would return nothing for Synchrony, Truist, Regions,
Fifth Third, Visa or Apple.

```
tests   52 across corpus, flatten, ingest, resolver, coverage,
        duckdb_backend and financials — all offline
```

---

## Related decisions

- [ADR-0003](decisions/ADR-0003-restatement-convention.md) — restatement convention
- [ADR-0004](decisions/ADR-0004-fiscal-period-normalization.md) — fiscal period normalization
- [ADR-0016](decisions/ADR-0016-bulk-endpoints-over-per-company-api-calls.md) — bulk endpoints over per-company calls
- [ADR-0018](decisions/ADR-0018-concept-mapping-layer.md) — concept mapping layer
- [ADR-0019](decisions/ADR-0019-annual-period-selection.md) — annual period selection

Previous: [Stage 0 — foundations](stage0_foundations.md) ·
Next: [Stage 2 — filing text](stage2_filing_text.md)
