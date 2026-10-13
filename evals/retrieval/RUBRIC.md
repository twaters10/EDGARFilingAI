# Retrieval relevance rubric

Every retrieval metric in this project -- recall@k, MRR, nDCG@10 -- measures agreement
with judgments made under this rubric. The numbers mean exactly what this page says
"relevant" means, and nothing more.

**Status:** Frozen -- changes go through the change log below  
**Frozen on:** 2026-09-24

## The unit being judged

A **passage**: a contiguous span of one filing document, identified by
`accession + document + char_start/char_end`. Not a chunk -- chunk boundaries differ
between chunkers, and the chunker A/B needs labels that do not depend on either one. A
retrieved chunk earns the highest grade of any gold span it overlaps.

## Grades

| Grade | Meaning | Test |
|---|---|---|
| **2** | Answers the question | A reader could answer from this passage alone, citing it. |
| **1** | Partial or context | On-topic and useful beside a grade-2 passage, but does not answer by itself. |
| **0** | Not relevant | Anything else, including text that shares the question's words but not its subject. |

Judge the passage against the **question as written**, not against what the retriever
seems to be attempting.

## Edge cases

Each carries a decision and a one-line reason. The reason is what makes a later judgment
consistent with an earlier one -- apply it to borderline passages the decision does not
literally cover.

1. **Boilerplate risk language.** Generic text most filers carry ("we are subject to
   extensive regulation...") when the question asks about one company's specific risk.
   - Decision: **0** if the passage could appear word for word in another filer's 10-K. **1** if it names something specific to this company (a partner, a program, a product).
   - Reason: Boilerplate outranking specific text is a failure mode to tag in error analysis (`boilerplate_outranked`); grading it relevant would hide that failure.
2. **Wrong period.** A passage from the FY2023 filing when the question names FY2024 --
   including when the text is identical year to year.
   - Decision: **0** when the question names a fiscal year and the passage is from a different one, even if the text is identical. When the question names no year, any year counts.
   - Reason: Retrieval runs unfiltered through Phase 5, so the wrong year *is* the failure; it should show in the metrics, which gives Phase 6's year filters something to fix.
3. **Tables and captions.** A table caption or row labels without the figures, or figures
   whose meaning depends on a header outside the passage.
   - Decision: **0**, unless the surrounding prose in the same passage answers the question.
   - Reason: Figures come from `query_financials`, never from text, and Stage 2 replaces data tables with placeholders -- a caption without its figures cannot answer anything.
4. **Auditor's report.** For control or "material weakness" questions, the independent
   auditor's opinion versus management's own Item 9A assessment.
   - Decision: **2**, the same as management's own Item 9A assessment.
   - Reason: The auditor's opinion on internal control is direct evidence, often the most explicit statement; it frequently sits in Item 8, not 9A, which `scope.items` should not hide.
5. **Cross-references.** A passage that says only "see Item 7, Credit Risk Management."
   - Decision: **0**. The passage it points to is what gets labeled.
   - Reason: A pointer answers nothing on its own; crediting it would reward retrieving the table of contents.
6. **Negative evidence.** A passage stating a thing is absent ("management concluded that
   internal control ... was effective") for a question asking whether it is present.
   - Decision: **2**.
   - Reason: For a presence question, a filer's explicit statement of absence *is* its answer -- the credible-negatives point of demo query 7. Without it, questions like r002 have almost no gold and score zero however good retrieval is.
7. **Generic regulatory framework** _(added 2026-09-24)_. A "supervision and regulation"
   passage describing a regime every bank is under (Bank Secrecy Act, USA PATRIOT Act,
   OFAC sanctions) when the question asks what *this* company says about its obligations.
   - Decision: **1** if the passage names and describes the regime the question asks about,
     even in language any bank could use. **2** only if it is specific to the company: its
     own program, exposures, findings, or enforcement history. **0** if the regime is only
     one name in a list of regulations.
   - Reason: The question asks about *obligations*, and a description of the regime states
     them, so it is context (unlike case 1, where generic risk text is not an answer). But
     only company-specific text answers what *this* company says.
8. _(add cases here as labeling surfaces them -- each addition is dated, and the questions
   it affects are re-judged)_

## Change log

| Date | Change | Questions re-judged |
|---|---|---|
| 2026-09-24 | Rubric frozen with edge cases 1-6 | none -- no labels yet |
| 2026-09-24 | Added case 7 (generic regulatory framework); affects r006, r009 | none -- no labels yet |
