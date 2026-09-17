# Stage 2 — Filing text: acquisition, sectioning, chunking

> **Maintained by hand.** This document is not regenerated when code changes. It
> describes the state of the pipeline as of **2026-09-17** and is updated only on
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
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼   60 filings · 400.6 MiB · 62 requests
              ┌──────────────────────────────┐
              │ 2 · NORMALIZE   normalize.py │   HTML → text, 10.6x smaller
              └──────────────┬───────────────┘
                             ▼   offsets index THIS string
              ┌──────────────────────────────┐
              │ 3 · SECTION     sections.py  │   locate Items 1A / 7 / 9A …
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ 4 · CHUNK       chunkers.py  │   item_aware | fixed_window
              └──────────────┬───────────────┘
                             ▼   14,043 chunks · median 696 tokens
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

| | scope | item label |
|---|---|---|
| `item_aware` | inside each located section, never across an item boundary | yes |
| `fixed_window` | the whole document, structure-blind | no — it has none to give |

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
      item, section_path, char_start, char_end, text)
```

`chunk_id` is `accession:item:char_start` — deterministic, so re-chunking the
same filing the same way yields identical ids. The **contextual prefix is never
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

**Normalized text and chunks are not persisted.** `normalize()` is a pure function
held in memory; `fc sections` re-derives everything from cached HTML each run and
throws it away. Persisting it is Stage 3 work, and it matters: `char_start` is
meaningful only against one specific normalized string. The footer fix above
shifted Capital One's text by 5,406 characters — had chunks been stored with
offsets, every citation into that filing would have silently pointed at the wrong
text. Stage 3 should store the text **and its hash**, so a normalization change
is a detectable invalidation rather than a silent corruption.

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
60 filings · 400.6 MiB · 62 requests
core items 1A, 7, 9A located in 46/60 (77%)
  item_heading 54 · crossref_pages 3 · none 3
offsets round-trip: OK across 60 filings
item_aware: 14,043 chunks · median 696 tokens · p95 795
```

Below the 90% bar. The 14 shortfalls, all named by `fc sections`:

| | n | cause |
|---|---|---|
| WFC, USB | 6 | **acquisition bug** — SEC's `primaryDocument` points at a wrapper; WFC's real 10-K is an 11.6MB sibling in the same accession |
| Citigroup | 3 | has a page table, but its left column is bare (`1A.` under an "Item Number" header) and ranges wrap across lines |
| JPMorgan | 3 | bundled annual report with its own pagination; item headings are stubs citing pages 46–160 |
| SYF FY2023, RF FY2023 | 2 | one-off partials, undiagnosed |

Fixing the first two would reach ~55/60 (92%). `PROJECT_PLAN.txt:314` flags this
as the stage most likely to overrun and says to report rather than grind — hence
the map instead of more patches.

**Also known:** Capital One and Citigroup list a single *start page* per item
where Synchrony lists *ranges*. `crossref.py` assumes ranges, so enabling it for
those filers without handling the distinction would collapse every span to about
one page.

---

## Related decisions

- [ADR-0001](decisions/ADR-0001-filing-type-scope.md) — filing type scope
- [ADR-0002](decisions/ADR-0002-item-aware-chunking.md) — item-aware chunking
- [ADR-0013](decisions/ADR-0013-cache-layout-mirrors-the-url-path.md) — cache layout
- [ADR-0017](decisions/ADR-0017-cache-entries-never-expire.md) — cache expiry
- [ADR-0019](decisions/ADR-0019-annual-period-selection.md) — annual period selection
