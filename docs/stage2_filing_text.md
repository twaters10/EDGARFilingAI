# Stage 2 — Filing text: acquisition, sectioning, chunking

> **Maintained by hand.** This document is not regenerated when code changes. It
> describes the state of the pipeline as of **2026-09-23** and is updated only on
> request. If a number here disagrees with `fc sections`, trust `fc sections`.

Turns 60 messy HTML filings into clean, section-attributed, citable chunks.

Everything downstream rests on one property: a chunk's `char_start`/`char_end`
must select exactly that chunk's text out of the normalized filing. That is what
makes a Stage 5 citation checkable rather than decorative, and it is asserted at
every step rather than assumed.

---

## The pipeline

```text
   config/corpus.yaml  ·  20 companies x 3 years
                  │
╔═════════════════▼══════════ 1 · ACQUIRE ═══════ filings/download.py ═══╗
║                                                                        ║
║   GET submissions.json          (data.sec.gov)                         ║
║        │                                                               ║
║   parse_recent  →  select_annual_filings                               ║
║        │            └─ selects on report_date, never filing_date       ║
║        │               dedupes a 10-K/A against its original           ║
║        ▼                                                               ║
║   got 3 years? ──NO──► walk filings.files newest-first  (LAZY)         ║
║        │               parse_page → re-select → stop when enough       ║
║        │◄──────────────┘   JPM needs 16 of ~50 pages                   ║
║        ▼                                                               ║
║   assert_complete                                                      ║
║        ├── short + unread pages ──────► ✗ FAIL, naming the page        ║
║        └── short, pages exhausted ────► OK, the company is just young  ║
║        │                                                               ║
║        ▼                                                               ║
║   GET primary document          (www.sec.gov/Archives)  → cached       ║
║        │                                                               ║
║   GET filing index (-index.htm) → any EX-13?  ──YES──► GET it too      ║
║        └─ WFC, USB: the 10-K is a wrapper; the Annual Report holds     ║
║           the items. SEC's primaryDocument names only the wrapper.     ║
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼   60 filings + 6 Annual Reports · cached
              ┌──────────────────────────────┐
              │ 2 · NORMALIZE   normalize.py │   HTML → text, 10.6x smaller
              └──────────────┬───────────────┘   10-K + EX-13 joined into ONE string
                             ▼   offsets index THIS string
              ┌──────────────────────────────┐
              │ 3 · SECTION     sections.py  │   headings → referrals → page table
              └──────────────┬───────────────┘   declared beats inferred
                             ▼
              ┌──────────────────────────────┐
              │ 4 · CHUNK       chunkers.py  │   item_aware | fixed_window
              └──────────────┬───────────────┘   labelled segments, multi-item
                             ▼   16,370 chunks · median 694 tokens
              ┌──────────────────────────────┐
              │ 5 · REPORT      coverage.py  │   located | partial | failed
              └──────────────┬───────────────┘
                             ▼
                  Stage 3 — embeddings + index
```

`assert_complete` runs **before** any document is fetched. Discovering truncation
after 40 downloads wastes rate budget and buries the message.

**Why pagination exists.** `filings.recent` caps at ~1000 rows. JPMorgan files so
heavily that its window spans a single year, so two of the three 10-Ks the corpus
wanted were simply not in it. Pages are read newest-first and the walk stops as
soon as enough annual filings are in hand — JPM needs 16 of its ~50 pages, and
reading them all would be 100,000 rows to find two filings.

---

## 2 · Normalize — HTML to text

A 10-K is not a document with markup around it. Capital One's is 9.2MB of HTML
carrying 79,167 `style=` attributes, 19,502 `<span>` elements and 5,251
inline-XBRL tags, flattening to about 920,000 characters of prose — a **10.6×**
reduction, corpus-wide.

```text
   raw HTML  (COF: 9.2 MB, 79,167 style= attrs, 19,502 spans, 501 tables)
       │
       ├─ drop <script> <style> <noscript>
       │
       ├─ for each <table>, innermost first ──┐
       │                                      │
       │        ┌─────────────────────────────┴──────────────────────────┐
       │        │  DATA?  >=3 rows AND >=2 cols                          │
       │        │         AND >=6 filled cells   ← keeps footers out     │
       │        │         AND >=30% of them numeric                      │
       │        └───────────┬───────────────────────────┬────────────────┘
       │                   YES                          NO
       │                    ▼                            ▼
       │        "[TABLE: 12 rows x 5 cols]"      drop_tag() — unwrap,
       │         figures come from XBRL           keep the text; a heading
       │         via SQL, never from text         in a cell stays anchored
       │
       ├─ block tags (p div tr td li h1-h6 table br) → newline
       │     without this, a heading runs into the paragraph below it
       │
       └─ collapse runs of spaces and blank lines
              │
              ▼
   normalized text   (COF: 920,339 chars)  ← every char_start/char_end
                                              in the system indexes this
```

Three rules, each learned from a filing that broke without it:

| Rule | Why |
|---|---|
| Data tables become placeholders | Figures come from XBRL via SQL, never from retrieved text |
| Layout tables keep their text | Synchrony's item headings live in one; stripping them made the filing unsectionable |
| Block elements become newlines | `text_content()` alone runs a heading into the paragraph below it |

**Size is part of the data-table test.** A page footer is one page number against
one company name — 50% numeric, sailing past a 30% threshold. Capital One wraps
every footer in a 3×3 table, so 261 of its 400 tables were footers. Treating them
as data destroyed the filing's entire page index, which §3's fallback depends on.
Hence `MIN_DATA_TABLE_CELLS = 6`: footers hold 2 cells, the smallest genuine data
table in the corpus holds 9.

---

## 3 · Section — locating the items

Naive matching does not work. One Capital One 10-K has **121 candidate item
headings across 12 items**; Item 8 alone matches 51 times. The *first* match for
every single item is the table of contents. First-match-wins is wrong 12 times
out of 12.

```text
   normalized text
        │
   ┌────▼──────────────── STRATEGY A · item headings ────────────────────┐
   │                                                                     │
   │   find_candidates      ^Item N at the start of a line               │
   │        │               COF: 121 raw matches → ~21 survive           │
   │        ▼                                                            │
   │   drop_toc_runs        5+ candidates within 300 chars = a contents  │
   │        │               listing; drop the whole run                  │
   │        ▼                                                            │
   │   longest_monotonic    keep the longest subsequence whose item rank │
   │        │               AND position both increase                   │
   │        ▼                                                            │
   │   found anything? ──YES──► sections [item_heading]        54 / 60   │
   └────────┬────────────────────────────────────────────────────────────┘
            NO
   ┌────────▼──────────── STRATEGY B · the filer's page table ───────────┐
   │                                     crossref.py                     │
   │   page_index()             longest run of consecutive standalone    │
   │        +                   number lines = the real page footers     │
   │   parse_page_references()  "Item 1A. / Risk Factors / 58 - 81"      │
   │        │                                                            │
   │        ▼                                                            │
   │   page table parsed AND both endpoints resolve?                     │
   │        ├── NO ──► ✗ REFUSE — never invent a span                    │
   │        └── YES ─► sections [crossref_pages]                3 / 60   │
   └────────┬────────────────────────────────────────────────────────────┘
            ▼
     neither worked ──► ✗ FAILED, reported by name              3 / 60

   Strategies are never blended. One filing, one strategy,
   recorded on Section.strategy.
```

The monotonic constraint is what makes this an algorithm rather than a pile of
special cases: it never needs to know *why* a candidate is wrong, only that a
document presents its items in order.

Its assumption is also the one filer style it cannot serve. Synchrony's 10-K
contains the string `Item 1A` **exactly once** in 560,000 characters — inside a
page-reference table — and its body is not in item order (7A precedes 1A). Those
filings fall through to `crossref.py`, which reads the mapping the filer
published instead of guessing at headings.

**Strategies are never blended.** A filing is sectioned one way or the other, and
`Section.strategy` records which. The two rest on different evidence, and an
aggregate that blurs them hides the thing worth watching.

### Referrals — following an item that says "see elsewhere" (`referrals.py`)

Four filers write some or all items as one-sentence pointers. The pointer is the
filer's *declaration* of where the item lives, so it is followed rather than
counted as a stub:

```text
   heading section, short, whose FIRST sentence is a referral
        │
        ├─ page references  "on pages 22 to 59"          USB (into its EX-13),
        │     → the target document's page index           JPM (its bundled report)
        │     → trimmed to a named heading if one sits
        │       inside the first page ("under 'Risk Factors'")
        │
        ├─ heading references  "under 'Financial Review – Risk Factors'"   WFC
        │     → walk the quoted path heading by heading
        │     → end at the next title in the exhibit's own table of contents
        │       (read from raw HTML — normalize turns it into a placeholder)
        │
        └─ into a note  "in Note 22 under 'Litigation and Regulatory Matters'"
              → start inside that note, end at the next note         USB, RF, TFC
```

Four refusals, each learned from a real filing that broke without it:

| Rule | Failure it prevents |
|---|---|
| Only the item's **first sentence** is read | JPM Item 2 is real content that ends "Refer to … pages 51–54"; Fifth Third's second sentence cites page 19 for forward-looking statements |
| "beginning on page 81" is **not** a range | a start with no end resolved to one page of market-risk text labelled Item 1C (BAC) |
| A heading with no known end **does not resolve** | running to end-of-document turned Mastercard's Item 10 into a 380,000-char span that erased every other section |
| Items 10–14 are **never followed** | they point at the proxy statement, which is not in the filing |

### Declared beats inferred

A heading span is *inferred* — it runs to wherever the next heading happens to
be. A referral or page-table span is *declared* by the filer. Where they overlap,
the declared span is carved out of the inferred one. Without this, JPMorgan's
Item 15 heading runs to the end of the file and claims its entire annual report.

Declared spans may overlap **each other**, and are kept as declared: Citigroup
lists pages 64–120 under both Item 7 and Item 7A. See §4 for how that is chunked.

---

## 4 · Chunk — two chunkers, one engine

```text
   a span of text  (one located section, or the whole document)
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  place a window of 3,200 chars  (800 tokens x 4 chars/token) │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  retreat to a clean edge, searching the window's 2nd half:   │
   │     last ¶ break  →  last line break  →  last sentence end   │
   │     →  hard cut                                              │
   │  (typical landing: ~2,768 chars, not 3,200)                  │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
                         emit the chunk
                               │
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  next window starts 480 chars back from where this one       │
   │  ACTUALLY ended — not from a nominal stride                  │
   │                                                              │
   │      ├────────── chunk N ──────────┤                         │
   │                          ├───────── chunk N+1 ─────────┤     │
   │                          ├─ 480 ─┤                           │
   └───────────────────────────┬──────────────────────────────────┘
                               │
                               └──► loop until the span is consumed
```

| | scope | item labels |
|---|---|---|
| `item_aware` | inside each labelled segment, never across a boundary | every item that claims the text |
| `fixed_window` | each source document, structure-blind | none — it has none to give |

**Labelled segments.** `item_aware` does not chunk section by section. It cuts the
filing at every span boundary of every section, and labels each piece with the set
of items whose spans cover it. Each character is chunked **once**; text two items
both declare carries both labels, and a search filtered to either finds it.

```text
   Item 7  declares  ├──────────────── pages 8–36, 64–120 ──────────────┤
   Item 7A declares                       ├──── 64–120 ────┤
   segments          ├── (7) ───────────┤├── (7, 7A) ─────┤
```

(Before this, SYF's Item 3 — nested inside its Item 8 — was chunked, embedded and
indexed twice under two labels.)

`fixed_window` exists as the baseline Stage 4 measures `item_aware` against
(ADR-0002). "Item-aware chunking is better" is a claim; two retrieval scores from
the same harness are evidence.

**Step E is load-bearing.** Anchoring the next window to a nominal stride instead
lets the boundary retreat eat the overlap — measured on a real filing that left a
median overlap of 84 characters against a configured 480, and eliminated it
entirely at 43% of seams. Overlap is now exactly 480 at every seam within a
continuous span, and zero across a boundary the filing itself defines (item to
item, or between a cross-reference item's disjoint blocks), which is deliberate.

Each chunk carries:

```
Chunk(chunk_id, cik, accession, form, period_end,
      items, section_path, document, char_start, char_end, text)
```

`chunk_id` is `accession:char_start` — deterministic, and independent of the
labels: each character is chunked once per chunker, so the start offset alone
identifies a chunk. Relabelling a span after a better parse leaves ids unchanged.
`document` names the source file (the 10-K, or its Annual Report). The **contextual prefix is never
stored in `.text`**; it is built at embed time by `contextual_prefix()`. Folding
it in would break the offset round trip and put words in a filing's mouth.

> Stage 3 must send this to the encoder as
> `"search_document: " + contextual_prefix(...) + "\n\n" + chunk.text`,
> with nomic's task prefix **outermost**. Wrong order degrades retrieval silently.

---

## The three refusal gates

The load-bearing design choice across the whole stage is where it declines to
guess. Each gate turns a silent corruption into a visible failure.

| Gate | Refuses when | Would otherwise cause |
|---|---|---|
| `assert_complete` | short, with unread history pages | a silently one-third-sized corpus |
| `crossref._span_offsets` | a declared page will not resolve | "sections" spanning the whole filing |
| `coverage` plausibility | a section is under 1,000 chars | a referral stub counted as content |

---

## Where things live

```
data/raw/filings/<cik>/<accession>/<document>.htm   401 MB   original SEC HTML
                                   + .meta.json              url, sha256, fetched_at
data/raw/submissions/                                11 MB   submissions.json + history pages
```

The cache path mirrors SEC's own URL path (ADR-0013), so provenance is readable
off the location. Entries never expire (ADR-0017) — a filed 10-K is immutable.

**Normalized text is re-derived, and its hash is kept.** `normalize()` is a pure
function; `fc sections` re-derives everything from cached HTML each run.
`char_start` is meaningful only against one specific normalized string — the
footer fix above shifted Capital One's text by 5,406 characters — so Stage 3's
manifest stores each chunk's text **and** `text_sha256`, the hash of the full
normalized filing its offsets index. A normalization change is then a detectable
invalidation rather than a silent corruption.

---

## Commands

```bash
fc fetch-filings --years 3              # acquire; 0 requests when warm
fc sections --years 3                   # the coverage report — the deliverable
fc sections --chunker fixed_window      # same sectioning, baseline chunker
fc show --ticker COF --item 1A          # print a located section + its offsets
fc show --ticker SYF --item 1 --offsets # offsets only; shows disjoint blocks
```

`fc sections` exits 1 below `--threshold` (default 0.90). That is the gate
working, not a crash.

---

## Where it stands

```
60 filings + 6 Annual Report exhibits (WFC, USB)
core items 1A, 7, 9A located in 59/60 (98%)      — was 46/60 (77%) on 2026-09-17
  item_heading 41 · referral 13 · crossref_pages 6
offsets round-trip: OK across 60 filings
item_aware: 16,370 chunks · median 694 tokens · p95 794 · 1,205 carry two items
```

Above the 90% bar. Every shortfall from the 2026-09-17 table is fixed:

| | n | was | fix |
|---|---|---|---|
| WFC, USB | 6 | 10-K wrapper only | fetch the EX-13; follow page and heading referrals into it |
| Citigroup | 3 | page table unreadable | bare `1A.` item column; ranges rejoined across line wraps |
| JPMorgan | 3 | page index covered pages 204–328 only | footer chain skips stray numbers; stubs followed by page |
| SYF FY2023 | 1 | partial | page index now spans the whole filing |

**The one remaining:** RF FY2023's Item 9A is real content that is only 980
characters long — under the 1,000-character plausibility floor. Not a parser
failure; the floor is doing its job conservatively, and lowering it to pass one
filing would let genuine stubs through. Its Item 7A ("set forth in the Risk
Management section of Item 7") names a section without quotes or pages, so it
stays an unresolved referral, reported by name.

**Not implemented:** page tables that list a single *start* page per item. No
filer in the corpus needs it — Capital One sections by heading — so building it
would be untestable against real data.

---

## Related decisions

- [ADR-0001](decisions/ADR-0001-filing-type-scope.md) — filing type scope
- [ADR-0002](decisions/ADR-0002-item-aware-chunking.md) — item-aware chunking
- [ADR-0013](decisions/ADR-0013-cache-layout-mirrors-the-url-path.md) — cache layout
- [ADR-0017](decisions/ADR-0017-cache-entries-never-expire.md) — cache expiry
- [ADR-0019](decisions/ADR-0019-annual-period-selection.md) — annual period selection
