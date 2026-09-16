# ADR-0018: Declarative concept-mapping layer

**Status:** Accepted  
**Stage:** 1

## Context

"Total debt" is not an XBRL tag. Across the 20-company corpus, one business concept is expressed through many `us-gaap` tags: total revenue resolves through **five** different ones, and the preferred tag answers for only 6 of 20 companies. A query written against any single tag silently returns nothing for most of a peer set, which is the failure mode behind the flagship "debt increased — how much?" question.

Tag-name matching is not an option. A substring match on `Debt` pulls in `AvailableForSaleDebtSecurities*`, which are investment **assets**, not money owed — it would report a bank's bond portfolio as its borrowing.

## Decision

A declarative YAML layer (`structured/concepts.yaml`) mapping each business concept to an **ordered allow-list of candidate tags**, plus `unit`, `period_type`, and a resolution rule. `first_available` takes the first candidate a company actually reports.

Three properties are load-bearing:

* **Resolution records which tag answered, and its rank.** Two companies answered from different tags are not automatically comparable, and rank distinguishes "resolved via the preferred tag" from "resolved via a fourth-choice fallback."
* **`unit` and `period_type` are part of the concept.** Units in this data are free-form (`USD`, `shares`, `pure`, `business`, `putative_class_actio`), and an instant is not comparable to a duration.
* **`applies_to` scopes coverage thresholds** to `all` or `lenders`.

## Alternatives considered

Hardcoding tag lists in Python makes adding a concept a code change and hides the mapping from review. Pattern matching is actively wrong, as above. Accepting a single tag per concept was the original implicit design and fails for 14 of 20 companies on revenue alone.

## Consequences

Adding a concept is a data change. The YAML file is itself a portfolio artifact: it is the unglamorous layer that separates a system from a demo.

The coverage matrix reports honest gaps rather than fabricating figures. Two are known and structural: **SEC's companyfacts excludes company custom extension tags**, so AmEx's card-member allowance and JPMorgan's long-term debt are absent from this data source entirely. Closing those requires per-filing XBRL, which is out of scope for Stage 1.

This supersedes the stage's original DONE WHEN threshold, which required `allowance_for_credit_losses` to resolve for ≥18 of 20 companies. Card networks and non-financial filers hold no credit-loss reserve, so that threshold was unreachable for any corpus containing a contrast filer, and the only way to pass it was to pack the corpus with 20 lenders. Thresholds are now per-concept and scoped by `applies_to`.
