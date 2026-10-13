# Stage 4 — Retrieval, built by deliberate failure

> **Maintained by hand.** This document is not regenerated when code changes. It
> describes retrieval as of **2026-09-25** and is updated only on request. If a
> number here disagrees with `fc search` or `fc eval-retrieval`, trust the command.

Turns Stage 3's two indexes into retrieval you can defend **with numbers**: a vector
retriever and a BM25 retriever that were each watched failing on their own, fused by
Reciprocal Rank Fusion, scoped by metadata filters and strategies, and measured by an
eval harness whose labels come from reading the filings.

The property everything here rests on: **relevance is judged on passages of filing
text, never on chunks and never on what a retriever returned.** Labels are spans
(`accession + char_start/char_end`) found by searching the filing itself, so the same
labels score both chunkers, and no retriever can shape the labels it is scored against.

---

## What Stage 4 outputs

| Artifact | Where | What it is |
|---|---|---|
| **Retrieval package** | `src/filing_copilot/retrieval/` | Retrievers, fusion, filters, strategies, and the eval: metrics, eval set, runner, labeling, adjudication. |
| **Search CLI** | `fc search` | Vector, BM25 or hybrid; filters; fan-out; `--compare` shows all three with fused provenance. |
| **Eval CLI** | `fc label-span`, `fc eval-retrieval`, `fc adjudicate` | Label gold from filings; score the grid; judge pooled hits blind; recheck label agreement. |
| **Eval set** | `evals/retrieval/` | `RUBRIC.md` (frozen), `questions.yaml` (30 questions, 6 blind), `judgments.yaml`, `failures.yaml`. Committed. |
| **Eval runs** | `data/processed/eval/retrieval-<split>-<strategy>-<time>.csv` | One row per question × configuration, with the top-10 chunk_ids. Regenerable; not committed. |
| **Walkthrough** | `notebooks/stage4_walkthrough.ipynb` | One section per checkpoint (0, A–E), each rerunnable after Setup. |

---

## The pipeline

```text
                         question
                            │
      ┌─────────────────────┼──────────────────────────────┐   strategies.py
      │ global              │ scoped: filters from scope   │
      │                     │ peer (≥2 tickers): fan_out_peers → one search per company
      │                     │ screening: section_targeted → items filter
      └─────────────────────┼──────────────────────────────┘
                            ▼
   ┌──────────── search(query, mode, k, filters) ─────────────┐   search.py
   │                                                          │
   │  vector:  encode_query ("search_query: " + q)            │
   │           → knn { vector, k, filter }   ← filter INSIDE  │
   │  bm25:    bool { must: match(text), filter }             │
   │  hybrid:  both at depth CANDIDATES=100                   │
   │           → reciprocal_rank_fusion(k=60) → top k         │   fusion.py
   └──────────────────────────┬───────────────────────────────┘
                              ▼
                 list[Hit]  (rank, score, citation fields, sources)
                              │
   ┌──────────────────────────┼─────────── fc eval-retrieval ─┐
   │ questions.yaml gold  +  judgments.yaml                   │   evalset.py
   │   → graded spans (0/1/2) per question                    │
   │ covers(hit, span): ≥ 200 chars shared (or all of it)     │   metrics.py
   │ recall@5/@10 · MRR · nDCG@10 (graded) · unjudged@10      │
   │ peer coverage@10 · win/loss per question                 │   evaluate.py
   └──────────────────────────────────────────────────────────┘
```

---

## 1 · Two retrievers, each watched on its own

`search.py` keeps vector and BM25 as separate functions, and builds every request with a
pure function (`vector_query`, `bm25_query`), so the exact request is readable and
testable without a server.

**What a kNN score is.** The index scores by inner product. The vectors are unit length,
so inner product is cosine similarity. OpenSearch (faiss) does not return it raw: it
reports `1 + ip` when `ip ≥ 0` and `1/(1 − ip)` otherwise. `cosine_from_knn_score`
inverts this. It was verified on the live index: a chunk's own vector scores exactly 2.0, and
querying with that vector *negated*, the top score 0.636 inverts to −0.572, which is exactly
the dot product.

**Cosines are compressed.** Across 3,000 random chunk pairs, cosine has median **0.72**,
5th percentile 0.64, minimum 0.55. Every 10-K chunk shares register and boilerplate, so the
corpus sits in a narrow cone. A vector retriever therefore always returns something that
looks confident, and a fixed threshold ("relevant if > 0.7") means nothing here. Scores
are comparable only within one result list.

**The two retrievers see different text.** A vector was embedded from
`contextual_prefix + chunk`, and the prefix names company, **ticker**, form, period and
item. BM25 indexes only the chunk text. For Synchrony, all **703** chunks carry "SYF" in
the embedded input, but only **3** contain it in the BM25-indexed text. On the query
"SYF", vector returns 20 of 20 Synchrony chunks and BM25 can return only 3. That is the
*reverse* of the textbook failure ("vector misses exact tickers"), and it is a direct
consequence of the Stage 2 prefix design. A ticker belongs in a filter, not the query.

**The task prefix, measured.** Embedding the query without `search_query:` gives a
different query vector (cosine to the prefixed one as low as 0.927, for "customer
concentration") and changes up to half of the top 10 (5/10 overlap). No error is raised. `encode_query` always names the task.

---

## 2 · Fusion: Reciprocal Rank Fusion

`reciprocal_rank_fusion` is a pure function: each chunk scores `Σ 1/(k + rank)` over the
lists that returned it, with **k = 60** fixed by ADR-0010. Ties break by `chunk_id`, and
every fused hit records `sources` (for example `v#6 b#1`), so a result explains itself.

**Why ranks.** BM25 is unbounded and query-dependent. Cosine sits near 0.72 for unrelated
chunks. Any score blend needs a normalization, which is another knob. RRF needs none, and
rescaling one list's scores changes nothing (tested).

**What k = 60 does.** Rank 1 in one list scores 1/61 = 0.0164. Rank *r* in **both** lists
beats that whenever 2/(60 + r) > 1/61, which holds for every *r* up to 61. Agreement is the
signal.

**Candidate depth is a real parameter.** `hybrid_search` asks each retriever for its top
**100** before fusing. Fusion can only promote what a list returned, and on the probes the
fused top 5 reached down to source rank 95. Fusing at depth 10 instead changed 4 of 5
results for three of eight probes. The depth is fixed until the eval can measure it.

**What RRF cannot do.** It does not know which retriever is right for a query. On the
probes, **6 of 8** fused top-5s came entirely from chunks both lists returned (the exceptions: "SYF", where BM25 found only 3 chunks, and "Item 9A"). On
"material weakness", BM25's genuine restatement disclosures at #2 and #3 were displaced by
chunks both lists ranked only in the 20s, so only 1 of BM25's top 5 survived. Whether that
costs or gains on balance is exactly what the labelled eval measures.

### Probe findings (Checkpoints A–B)

Eight probes, kept clear of the blind split. They show *how* each retriever fails, not
*how often*.

| Probe | Vector alone | BM25 alone | Fused top 5 |
|---|---|---|---|
| material weakness | "weakness" as economic weakness (RF, BFH, WFC) | SoFi's restatement; ICFR risk factors | 5/5 from both lists; SoFi #1 (v#6 b#1) |
| SYF | Synchrony Item 1 (via the prefix) | 3 hits: Item 5 headers | mixed |
| Item 9A | unrelated BAC/AAPL text | the "Item 9A … can be found in" referral stubs | 4/5 from both |
| Durbin Amendment | SoFi, MA, V, on topic | the same filers, plus litigation text | 4 of vector's top 5 kept |
| retail-partner paraphrase | Synchrony partner-concentration risk factors | Synchrony Item 1 partner lists | on topic |
| depositor paraphrase | USB deposit-outflow risk factors | regulatory boilerplate ("cease and desist") | 3 of vector's top 5 kept, none of BM25's |

---

## 3 · Filters and strategies

**Filter placement is a silent failure.** `SearchFilters` becomes `terms` clauses: OR
within a field, AND across fields. For kNN the filter goes **inside** the `knn` clause, so
OpenSearch searches only matching vectors and still returns `k`. As a `post_filter` it
trims an unfiltered top-k instead. Measured: "customer concentration risk" for AAPL Item
1A returns **10** hits with the filter inside, and **0** as a `post_filter`. Neither raises
an error. There is a network test for it.

**Only `item_aware` can be targeted by section.** `fixed_window` chunks carry no items
(19,763 of 19,763), so an item filter matches nothing: `section_targeted("material
weakness", ["9A"])` returns 10 hits on `item_aware` and 0 on `fixed_window`. Section
targeting, and Stage 6's section-aligned change detection, exist only with the item-aware
chunker. This is a structural argument for ADR-0002 that holds whatever the A/B metrics
say.

**`fan_out_peers`** runs one filtered search per company and interleaves the results by
rank (every company's #1, then every #2). A company with no hits is reported in `empty`,
which is a finding rather than a dropped row. **`section_targeted`** names the common
screening pattern: the search scoped to the items where the answer lives.

**Concentration vs coverage, measured.** "How do card issuers describe their CECL
allowance methodology?" with a global hybrid top 10 returns **only Citigroup and Bank of
America**, so 0% of the four card issuers asked about. Fan-out covers all four. Across the
five dev peer questions (the eval's `peer coverage@10`, which needs no relevance labels):

| Configuration | global | scoped (fan-out) |
|---|---|---|
| vector / item_aware | 0.67 | **1.00** |
| bm25 / item_aware | 0.58 | **1.00** |
| hybrid / item_aware | 0.78 | **1.00** |
| any / fixed_window | 0.52–0.78 | 0.60 *(item filters match nothing)* |

---

## 4 · The eval

**Relevance is span overlap.** A gold span is `accession + char_start/char_end` in the
filing's normalized text, the same coordinates every chunk carries. A chunk earns a
span's grade if it shares at least **200** characters of it (`MIN_OVERLAP_CHARS`), or all
of it for a shorter span. Clipping the last line of a labelled paragraph is not
retrieving it. A chunk that covers no judged span is **unjudged**: relevance unknown, not
zero.

**The metrics** (`metrics.py`, each with a hand-worked test):

| Metric | Definition | Rewards |
|---|---|---|
| recall@k | share of relevant spans (grade ≥ 1) covered by some top-k chunk | finding everything |
| MRR | mean of 1 / rank of the first relevant chunk | one right answer, early |
| nDCG@10 | graded gain / log₂(rank + 1), normalized by the ideal ordering of the labelled spans | ranking all of it well |
| unjudged@10 | mean top-10 chunks covering no judged span | nothing: it measures how much to trust the other three |
| peer coverage@10 | share of the question's named companies present | coverage over concentration |

nDCG credits **each span once**, to the highest-ranked chunk covering it, so overlapping
windows or duplicated text cannot push it above 1. Recall is span-level for the same
reason: a chunker cannot score twice for one passage.

**Configurations.** 3 retrievers × 2 chunkers × 2 strategies. `global` is the question
text alone, which is what the raw retriever does and what the Phase 4–5 comparisons use.
`scoped` applies the question's recorded scope as filters, with fan-out for peer
questions. Means are over labelled questions only. `win_loss` reports, question by
question, which configuration did better, because with ~30 questions one question can
move a mean.

**Labels cannot drift silently.** Every gold span records the `filing_sha256` of the text
its offsets index. `check_gold` compares it with the manifest's `text_sha256` and refuses
to score if Stage 2's normalization has changed underneath the labels.

---

## 5 · The human in the loop

The metrics measure agreement with *your* judgments, so the judgments are where the rigor
lives. Every human step has a tool, and none of the tools make the judgment.

| Step | Who | How | Guards against |
|---|---|---|---|
| Rubric, frozen before labelling | operator | `evals/retrieval/RUBRIC.md`: 7 edge cases, each with a decision and a reason | "relevant" drifting between questions |
| Questions, before any search code | operator (r001–r030 drafted by Claude on request) | `questions.yaml`, `written_on` recorded | an eval shaped by the retriever's failures |
| Blind split, drawn by seed | machine | 20% per archetype, seed 20260924; r006 r009 r014 r015 r017 r025 | tuning on the test set |
| Gold spans, from the filing | operator | `fc label-span`: substring search, paragraph offsets, paste-ready YAML | labels biased towards what was retrieved |
| Adjudication, blind and pooled | operator | `fc adjudicate`: all six configurations' top 10, deduplicated, seeded shuffle, no system shown | false negatives in the gold; favoring the expected winner |
| Error analysis | operator | `failures.yaml`, fixed taxonomy; the eval lists untagged misses | changing the system because of an average |
| Label recheck | operator, ≥ 3 days later | `fc adjudicate --recheck`: 15% re-graded blind; percent agreement and Cohen's κ | reading noise as a difference |

**Blindness is structural.** `PoolItem`, the unit a judge sees, has no field for the
configuration, rank or score that produced it. A test asserts that no retriever or chunker
name appears in what is rendered.

**Adjudication only runs on questions that already have gold.** Adjudicating an
unlabelled question would build its labels entirely from retrieval output.

**Why κ and not only agreement.** A labeller who grades almost everything 2 reaches 90%
self-agreement by chance and κ = 0.00. One who uses all three grades reaches 90% with
κ = 0.85.

---

## Commands

`fc` is a zsh builtin. Every `fc` below means `.venv/bin/fc`.

### Search

```bash
fc search "material weakness"                                  # vector (default), item_aware
fc search "material weakness" --mode hybrid                    # fused; each hit shows v#/b# ranks
fc search "material weakness" --compare                        # vector, bm25, hybrid together
fc search "material weakness" --mode hybrid -i 9A              # section-targeted
fc search "CECL allowance methodology" --mode hybrid \
    -t COF -t SYF -t AXP -t BFH --fan-out -n 3                 # one search per company
fc search "allowance" --chunker fixed_window --fy 2024         # the other index, one year
```

### The eval loop

```bash
fc label-span -t SYF --fy 2024 -i 1A -f "largest partners"     # 1 · find passages; paste YAML under gold:
fc eval-retrieval                                              # 2 · dev split, global, both chunkers
fc eval-retrieval --strategy both                              #     + scoped (filters, fan-out)
fc adjudicate                                                  # 3 · judge pooled unjudged hits, blind
fc adjudicate -q r004                                          #     one question
#                                                              # 4 · tag misses in failures.yaml
fc adjudicate --recheck                                        # 5 · days later: label agreement
fc eval-retrieval --split blind                                # 6 · once, at the end (logged)
```

| Command | Options | What it does | Needs |
|---|---|---|---|
| `fc search` | `--mode vector\|bm25\|hybrid` · `--compare` · `--chunker` · `-k` · `-t/--fy/-i` (repeatable) · `--fan-out -n` | One index, one or three retrievers, optional filters or fan-out | OpenSearch; Ollama unless `bm25` |
| `fc label-span` | `-t` · `-f` · `--fy` · `-i` · `--grade` · `--max` | Paragraphs containing a phrase, with offsets and a gold YAML entry | filings cached |
| `fc eval-retrieval` | `--split dev\|blind\|all` · `--strategy global\|scoped\|both` · `--chunker` | Validates the eval set and checks gold for drift; scores the grid; prints means, win/loss, failure table, agreement; writes CSV | OpenSearch; Ollama |
| `fc adjudicate` | `-q` · `--split` · `--depth` · `--recheck --fraction` | Interactive blind judging; appends to `judgments.yaml` / `rechecks.yaml` as you go | OpenSearch; Ollama (not for `--recheck`) |

### Tests

```bash
pytest                    # 469 offline -- 85 of them Stage 4 (fake OpenSearch, fake encoder, no services)
pytest -m network         # 5 live, 2 of them Stage 4: all three retrievers, and the post_filter failure
```

**Step through it:** `notebooks/stage4_walkthrough.ipynb`, with one section per
checkpoint, calls these same functions cell by cell.

---

## Where it stands

**Done, with numbers, and not dependent on labels:**
- both retrievers characterized on the probes, and fusion's provenance visible per hit;
- the score transform, cosine compression, prefix asymmetry and task-prefix effect, all
  measured on the live index;
- peer coverage: global top-10 52–78% against fan-out 100% (item_aware);
- the filter-placement failure (10 vs 0 hits), and the structural chunker result
  (only item_aware can be section-targeted).

**Built and tested, waiting on labels.** The eval set has 30 questions and **0 gold
spans**. Until labels exist:
- recall@k, MRR and nDCG@10 per retriever × chunker are blank, and so are the win/loss
  counts;
- there is no recheck κ, so no noise floor;
- so the stage's "done when", *what hybrid buys over vector-only, and which chunker won*,
  is not yet answered.

The risk register's instruction applies: **time the first 10 questions' labels**
(`PROJECT_PLAN.txt:878`), and multiply by seven before committing to Stage 7's ~70.

---

## Known gaps

- **No gold labels yet.** This is the operator's step, and deliberately not automated:
  labels written by a model would measure agreement with that model.
- **ADR-0002 and ADR-0010 are not updated.** Their verdicts need the labelled grid.
  ADR-0002 can already record the structural result above.
- **The scoped `fixed_window` numbers are not a fair chunker comparison.** Item filters
  match none of its chunks, so any item-scoped question returns nothing. Compare chunkers
  on the `global` strategy.
- **Adjudicated positives on whole chunks are coarse.** A positive recorded without a
  narrowing phrase spans a ~2,800-character chunk, and the 200-character overlap rule
  then credits any chunk sharing a small slice of it. Narrow positives when judging.
- **No reranker, no k or depth tuning, no HNSW tuning.** All deliberately out of scope
  until the eval has numbers to move (ADR-0010; `PROJECT_PLAN.txt`, fusion-tuning risk).
- **`mypy --strict` over `tests/` reports 24 errors**, all in test files that predate
  Stage 4 (`test_financials`, `test_resolver`, and five others). `src/` and every Stage 4
  test file are clean.

---

## Related decisions

- [ADR-0002](decisions/ADR-0002-item-aware-chunking.md) — item-aware chunking (the A/B this stage settles)
- [ADR-0006](decisions/ADR-0006-change-detection-by-section-alignment.md) — change detection by section alignment (needs item filters)
- [ADR-0009](decisions/ADR-0009-embedding-model-nomic-embed-text-v15.md) — embedding model and task prefixes
- [ADR-0010](decisions/ADR-0010-reciprocal-rank-fusion-for-hybrid-search.md) — reciprocal rank fusion, k = 60
