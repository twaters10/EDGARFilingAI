# ADR-0019: Annual period selection

**Status:** Accepted  
**Stage:** 1

## Context

"FY2024 revenue" has to be resolved to specific rows in the fact table, and three properties of the data make the obvious approaches wrong.

`fy` cannot be used for selection: it is the fiscal year of the *report that contained* the fact, not of the fact itself. Synchrony's earliest row ends 2013-12-31 but carries `fy=2014, fp=Q2` — a prior-period comparative inside a later 10-Q.

Filtering only by period year is not enough either. A duration concept returns the annual figure alongside three quarterly ones; an instant concept returns the fiscal year-end balance alongside three quarter-end balances.

And a fiscal year end cannot be assumed. Measured across the corpus for FY2024: Capital One and Mastercard end 2024-12-31, Visa 2024-09-30, Jefferies 2024-11-30, and Apple **2024-09-28** — a 52/53-week calendar that lands on neither a month end nor a fixed date.

## Decision

**Duration concepts**: the annual observation is the one whose span falls in a **330–400 day window**. Measured against the corpus, spans cluster at 90/91 (quarterly), 181 (half), 273 (nine months) and 365, so the window isolates the annual figure with no overlap.

**Instant concepts**: select the observation whose `end` equals that company's **derived** fiscal year end — taken from the most frequently reported annual duration end date for that company and year. Falls back to the latest `end` in the year when a company reports no annual duration fact.

Deriving rather than assuming is what handles Apple's 09-28 without a special case, and what stops Visa's 31 December quarter-end balance being mistaken for its fiscal year end.

## Alternatives considered

Selecting on `fy`/`fp` looks natural and is wrong for the reason above. A hardcoded month/day per company pushes a maintenance burden into config and is wrong for any 52/53-week filer. Taking the maximum period end in the calendar year returns Visa's Q1-FY2025 balance as its FY2024 year end.

## Consequences

One rule covers December, September and November filers and 52/53-week calendars alike, with no per-company configuration.

Quarterly selection is deliberately not built. The window generalizes to it (80–100 days), but Q4 is not reported as a 10-Q — three 10-Qs per year, with Q4 folded into the 10-K — so quarterly support needs Q4 derived as full-year minus the three reported quarters. That is a stretch goal, not Stage 1.
