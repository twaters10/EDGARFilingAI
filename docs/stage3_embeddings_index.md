# Stage 3 — Embeddings and the local index

> **Maintained by hand.** This document is not regenerated when code changes. It
> describes the state of the pipeline as of **2026-09-23** and is updated only on
> request. If a number here disagrees with `fc embed` or `fc build-index`, trust
> the command.

Turns Stage 2's 14,043 chunks into a searchable index — lexically (BM25) and
semantically (k-NN) — that can be rebuilt from stored artifacts in seconds.

The property everything here rests on: **the string that gets embedded, the key
it is cached under, and the vector that comes back are bound together by
construction.** Change any part of the input — the chunk text, the contextual
prefix, the task prefix, the model, the dimensionality — and the key changes with
it, so a stale vector can never be served for new text.

---

## What Stage 3 actually outputs

Three artifacts. Two are the source of truth; the third is derived from them.

| Artifact | Where | Size | What it is |
|---|---|---|---|
| **Embedding cache** | `data/processed/embeddings/nomic-embed-text@768/part-*.parquet` | 65 MB, 55 parts | `(digest, vector[768])` rows. Durable. Append-only. |
| **Manifest** | `data/processed/manifests/item_aware__nomic-embed-text@768.parquet` | 15 MB | One row per chunk: metadata, text, and the `digest` of its vector. Durable. Replaced each run. |
| **Index** | OpenSearch `filings-item_aware` (Docker) | 290 MB | The join of the two above. **Disposable** — rebuilt in ~14s. |

The index is what Stage 4 queries. The cache and manifest are what you keep:
lose the index and `fc build-index` recreates it without touching Ollama or SEC.

---

## The pipeline

```text
   Stage 2:  cached HTML → normalize → find_sections → item_aware
                                                          │
                                        14,043 Chunk(text, offsets, item …)
                                                          │
╔═════════════════════════════════ fc embed ═════════════▼═══════════════════════╗
║                                                                                ║
║   plan_corpus                                          embed/pipeline.py       ║
║     for each chunk:                                                            ║
║       document_input  =  contextual_prefix + "\n\n" + chunk.text               ║
║       digest          =  sha256( "search_document: " + document_input )        ║
║       ManifestRow(chunk metadata, text, digest)                                ║
║                                   │                                            ║
║   cache.cached_digests()  ────────┤  one Parquet column, no vectors loaded     ║
║                                   ▼                                            ║
║   missing = digests not in cache                                               ║
║        │                                                                       ║
║        ├─ 0 missing ──────────────────────────► skip straight to manifest      ║
║        ▼                                                                       ║
║   embed_missing, 256 at a time                                                 ║
║     OllamaEncoder.encode  →  prepare (adds task prefix)  →  POST /api/embeddings ║
║                           →  L2-normalize (Ollama returns norm ≈ 20)           ║
║     cache.extend  → new part-NNNNN.parquet     ← after EVERY batch             ║
║                                   │                                            ║
║   write_manifest ◄────────────────┘                                            ║
╚═══════════════════════════════════╤════════════════════════════════════════════╝
                                    ▼   14,043 vectors · 917s cold · 0s warm
╔══════════════════════════════ fc build-index ══════════════════════════════════╗
║                                                                                ║
║   read_manifest  +  cache.load(digests)                  index/build.py        ║
║        │                                                                       ║
║   CHECK  every row's model == this build's model   ──► ✗ refuse                ║
║   CHECK  every digest has a cached vector          ──► ✗ refuse: run fc embed  ║
║   CHECK  every vector has norm 1 ± 1e-3            ──► ✗ refuse                ║
║        │              (all three before the old index is touched)              ║
║        ▼                                                                       ║
║   DELETE index → PUT index_body(model) → _bulk ×29 → _refresh → _count         ║
║                                                          │                     ║
║                                   _count ≠ manifest rows ──► ✗ refuse          ║
╚═══════════════════════════════════╤════════════════════════════════════════════╝
                                    ▼   14,043 documents · 12–14s · 0 embeddings
                          Stage 4 — hybrid retrieval (RRF)
```

---

## 1 · The embedding input — assembled in exactly one place

```text
  "search_document: Synchrony Financial (SYF) · 10-K · period ending 2025-12-31 · Item 1A. Risk Factors\n\n<chunk text>"
   └─── task prefix ─┘└──────────────────── contextual prefix ──────────────────────┘└sep┘└─ chunk.text ─┘
     added by the         chunkers.contextual_prefix()                              PREFIX_SEPARATOR
     encoder (prepare)    — built at embed time, never stored in .text
```

**The task prefix is not optional, and Ollama does not add it.** nomic is trained
asymmetrically — documents as `search_document:`, questions as `search_query:`.
Measured on this Ollama build, `cos(bare, "search_document: " + text) = 0.862`:
the prefix reaches the model as content, so it is ours to add. That is pinned by a
live test (`test_ollama_does_not_apply_the_task_prefix_for_us`), not assumed.

**The contextual prefix** is Anthropic's contextual-retrieval idea at zero cost:
the chunk carries its own company, form, period and item into the vector, and the
facts already exist in filing metadata, so no LLM call generates them. It is
**not** in the manifest's `text` — BM25 indexes the filing's own words, and the
`ticker` filter is what scopes to a company.

`document_input()` builds the body; `prepare()` adds the task prefix. The digest
is computed through the same `prepare()`, so the cache key and the model's input
cannot drift apart — a test hashes what the fake encoder received and compares.

---

## 2 · The encoder — a seam, not a service

```text
   Encoder (Protocol)                    embed/encoder.py
     .model   → EmbeddingModel(name, dimensions, needs_task_prefix, max_input_chars)
     .encode(texts, task=…)   task is REQUIRED — no default to forget
        │
        ├── OllamaEncoder          now      embed/ollama_encoder.py
        └── SageMaker encoder      Stage 8  — same pure functions, different transport
```

Policy lives in pure functions (`prepare`, `l2_normalize`, `truncate_dimensions`),
so Stage 8 inherits identical behaviour rather than a second implementation.

| Guard | Why |
|---|---|
| Refuse inputs over 12,000 chars | Measured: input is honoured to ~12k and the server 500s above ~16k. The limit sits below the failure point; a truncated chunk would embed its first half and cite its whole span. Longest real input: 3,354 chars. |
| Always L2-normalize | Ollama returns norm ≈ 20. The index scores by inner product, which ranks correctly only for unit vectors. |
| Only `nomic-embed-text` accepted from config | Prefix behaviour and the input limit were *measured* for nomic. Any other name would reuse measurements taken on a different model. |

---

## 3 · The cache — keyed on content, namespaced by model

```text
   key  = sha256( exact string sent to the model )
   path = embeddings/<model>@<dimensions>/part-NNNNN.parquet
```

Every narrower key fails the same way — silently:

| Key on… | Survives… | And then serves |
|---|---|---|
| `chunk.text` alone | a change to the contextual prefix | vectors built from the old prefix |
| `chunk_id` | a normalization change that keeps offsets | vectors for text that no longer exists |
| no model namespace | a switch to 256 dimensions | a 768-d vector into a 256-d index |

The cost of the wide key is that editing the prefix format re-embeds everything —
~15 minutes here, hours at S&P 500 scale. So `tests/test_chunkers.py` **pins the
prefix's exact output**: a casual edit fails a test instead of quietly starting a
long rebuild.

Writes **append a part per batch** rather than rewriting. An interrupted run keeps
every finished batch (tested by failing a fake encoder mid-run and resuming), and
re-embedding one filing never rewrites 65 MB of unrelated vectors.

---

## 4 · The manifest — what the index is rebuilt from

```
ManifestRow(chunk_id, cik, ticker, accession, form, period_end, fiscal_year,
            item, section_path, char_start, char_end, text, digest, model)
```

**The text is stored, not re-derived.** Re-deriving at build time would re-run
`normalize` and the chunkers, and any change to either would put text in the index
beside a vector computed from different text. Storing it freezes the pair.

It is **replaced, not appended**: a chunk that no longer exists must vanish from
the next rebuild. Duplicate `chunk_id`s are refused on write — `chunk_id` becomes
the OpenSearch `_id`, so a duplicate would silently overwrite another chunk.

The schema is explicit: `period_end` is `date32`, not whatever pyarrow infers, so
it maps cleanly to an OpenSearch `date` and range filters work.

---

## 5 · The index — searchable two ways

```text
   one document per chunk                                   index/mapping.py
   ├── text           text, english analyzer  ──► BM25       lexical half
   ├── embedding      knn_vector[768]         ──► HNSW       semantic half
   │                   faiss · innerproduct · m=16 · ef_construction=128
   ├── filters        keyword: cik ticker form item accession section_path chunk_id
   │                  date: period_end        integer: fiscal_year
   └── stored only    char_start char_end digest model   (index: false)

   dynamic: strict  — an unmapped field fails the bulk request instead of
                      quietly creating an unfiltered one
```

**Inner product, because every vector is unit length.** For unit vectors inner
product *is* cosine, without the division. The dependency runs one way —
`build_index` checks every norm before writing, so an unnormalized vector cannot
reach this space.

**Reading scores.** faiss reports inner-product similarity as `1 + dot` for
non-negative dots, so a k-NN `_score` of **1.748 means cosine 0.748**. BM25 scores
are on an unrelated scale — which is exactly why Stage 4 fuses by *rank* (RRF,
ADR-0010), not by score.

**One index per chunker** (`filings-item_aware`, `filings-fixed_window`), so the
ADR-0002 A/B compares two indexes over one cache rather than two pipelines.

### The client is ~150 lines of `httpx`, on purpose

`opensearch-py` would do all of it. It is not used because Stage 3 is where the
mechanics should be visible: an index build *is* a `PUT` with a mapping, a `POST`
of NDJSON, a refresh and a count. It also keeps tests on `httpx.MockTransport`
like every other HTTP test here.

The cost lands in **Stage 8**: Serverless needs SigV4-signed requests, which this
client will get as an `httpx.Auth` built on botocore's `SigV4Auth`. A constructor
argument, not a rewrite.

One API trap worth knowing: **`_bulk` returns HTTP 200 when documents fail.**
Failures are inside the body under `errors: true`. The client checks the body;
checking only the status would report a partial load as success.

---

## The refusal gates

As in Stage 2, the load-bearing choices are where it declines to proceed.

| Gate | Refuses when | Would otherwise cause |
|---|---|---|
| `OllamaEncoder` input limit | input > 12,000 chars | a silently truncated vector citing a full span |
| `_embedding_model` | `EMBEDDING_MODEL` isn't nomic | unmeasured prefix behaviour, silent regression |
| `write_manifest` | duplicate `chunk_id` | one chunk silently overwriting another in the index |
| `build_index` · model | manifest was embedded with another model | 768-d vectors in a 256-d mapping, or mixed models |
| `build_index` · missing | a digest has no cached vector | a build that quietly embeds — or quietly skips |
| `build_index` · norm | any vector's norm ≠ 1 ± 1e-3 | inner product ranking by length, not similarity |
| `build_index` · count | `_count` ≠ manifest rows | a partial index reported as complete |

The three `build_index` checks on model, cache and norm all run **before the
existing index is deleted**, so a build that is going to fail leaves the old one intact.

---

## Local infrastructure — what Docker is doing here

### In plain terms

Docker runs the search engine, **OpenSearch**, on this laptop.

After `fc embed` there are two files: the chunks (the manifest) and their vectors
(the cache). You *can* search those by hand — the walkthrough notebook does, by
comparing a question against every vector — but that stops scaling at millions of
chunks and can't do keyword search at all. OpenSearch is built for exactly this:

- **Keyword search (BM25)** — chunks containing *"allowance for credit losses"*.
- **Meaning search (k-NN)** — chunks *about* credit card losses, whatever words they use.
- **Filters** — only Synchrony, only FY2024, only Item 1A.

**Why Docker rather than installing OpenSearch directly:** OpenSearch is a large
Java server with its own dependencies. Docker runs it inside a **container** — a
sealed box holding the program and everything it needs, isolated from the rest of
the machine, and removable without a trace. Containers need Linux, so on a Mac
**Colima** runs a small Linux virtual machine for Docker to live in:

```text
   your Mac
    └── Colima VM         2 CPU · 4 GiB RAM · 20 GiB sparse disk (grows as used)
         └── Docker        runs containers
              └── opensearchproject/opensearch:2.19.1
                    single node · security off · 1 GiB heap
                    listening on 127.0.0.1:9200 only
                    image 2.0 GB · index 290 MB
```

`docker-compose.yml` is the recipe for that box: which OpenSearch version, how much
memory, which port, and a named volume where the index data persists between restarts.

### OpenSearch holds a copy, not the original

```text
   fc embed        →  cache + manifest        files on disk — the real output
   fc build-index  →  copied into OpenSearch  a fast lookup desk, not the filing cabinet
   Stage 4         →  queries OpenSearch
```

Delete the container, the volume, even Colima itself, and nothing is lost:
`fc build-index` refills the index from the two files in ~14 seconds.

### Why it's temporary

Stages 3–7 run locally and free. Stage 8 moves the same engine to AWS as
OpenSearch Serverless **NextGen** (ADR-0012 — never Classic). The code does not
change; only `OPENSEARCH_URL` points somewhere else. Docker is the local stand-in.

Security is disabled locally because the port binds to localhost only and the data
is public SEC filings. Do not copy this configuration anywhere reachable.

Configuration (`config.py`, overridable in `.env`): `OLLAMA_HOST`,
`EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `OPENSEARCH_URL`,
`OPENSEARCH_INDEX_PREFIX`.

---

## Where things live

```
data/processed/embeddings/nomic-embed-text@768/part-NNNNN.parquet    65 MB   (digest, vector)
data/processed/manifests/<chunker>__<model>@<dims>.parquet            15 MB   one row per chunk
OpenSearch: <prefix>-<chunker>                                       290 MB   derived; disposable
```

---

## Commands

**zsh users first:** `fc` is a zsh builtin (the history editor) and shadows the CLI.
Every `fc` below means `.venv/bin/fc` — or activate the venv, or
`alias fcp=.venv/bin/fc`.

### One-time setup

```bash
brew install colima docker docker-compose     # the VM, the Docker CLI, compose
ollama pull nomic-embed-text                  # the embedding model
uv sync --group notebook                      # optional: the walkthrough notebooks
```

### A full Stage 3 run, in order

```bash
# 1 · embed — needs Ollama running; no Docker needed
fc embed --dry-run                  # how many chunks are cached vs still to embed
fc embed                            # embed only what's missing (~15 min cold, 0s warm),
                                    # then write the manifest

# 2 · start the search engine
colima start --cpu 2 --memory 4 --disk 20     # boot the Linux VM (flags only matter the first time)
docker-compose up -d                          # start OpenSearch in the background
curl localhost:9200                           # alive? prints a version block

# 3 · build the index — needs OpenSearch; never touches Ollama or SEC
fc build-index                      # drop + recreate filings-item_aware from the files
```

### Command reference

| Command | Options | What it does | Needs |
|---|---|---|---|
| `fc embed` | `--chunker item_aware\|fixed_window` (default `item_aware`) · `--years N` (default 3) · `--dry-run` | Chunks the corpus, embeds only digests the cache lacks, saves every 256-chunk batch as it goes, writes the manifest. Safe to interrupt and re-run. | Ollama; filings cached |
| `fc build-index` | `--chunker item_aware\|fixed_window` | Checks model, vectors and norms, then drops and recreates `<prefix>-<chunker>`, bulk-loads, and verifies the count. Stops before touching the old index if anything is missing. | OpenSearch; a manifest |
| `fc sections` | `--chunker`, `--years`, `--threshold` | Stage 2's coverage report — the same text `fc embed` chunks. | filings cached |

For the ADR-0002 A/B, repeat both with `--chunker fixed_window`. It uses the same
cache directory but shares no vectors with `item_aware` — its prefix says "full
document" rather than an item — so expect another ~15 minutes of embedding.

### Looking inside the index

```bash
curl 'localhost:9200/_cat/indices/filings-*?v'                 # indexes, doc counts, sizes
curl 'localhost:9200/filings-item_aware/_count'                # should equal the manifest rows
curl 'localhost:9200/filings-item_aware/_mapping?pretty'       # the fields and their types

curl -s localhost:9200/filings-item_aware/_search -H 'Content-Type: application/json' -d '{
  "size": 3, "_source": ["ticker", "fiscal_year", "item"],
  "query": { "match": { "text": "allowance for credit losses" } } }'
```

A k-NN query needs a query vector, so it is easier from Python — cell 12 of
`notebooks/stage2_3_walkthrough.ipynb` runs BM25, k-NN and filtered k-NN side by side.

### Day to day

```bash
colima start && docker-compose up -d     # resume: the index is still there
docker-compose down                      # stop OpenSearch; index data is kept
colima stop                              # shut the VM, free its 4 GiB of RAM
docker-compose down -v                   # also delete the index volume — safe,
                                         # fc build-index recreates it in ~14s
```

OpenSearch only needs to run while building the index or searching. `fc embed`
does not need it.

### Tests

```bash
pytest                              # offline: fake encoder, fake OpenSearch — no services
pytest -m network                   # live: Ollama (2 tests) + OpenSearch (1 test)
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `fc embed` runs zsh's history editor | `fc` is a zsh builtin | `.venv/bin/fc embed` |
| `Could not reach Ollama` | Ollama not running | `ollama serve` |
| `Could not reach OpenSearch … docker compose up` | VM or container stopped | `colima start && docker-compose up -d` |
| `docker-credential-desktop: executable file not found` | `~/.docker/config.json` still has `"credsStore": "desktop"` from Docker Desktop | delete that line |
| `docker compose` (with a space) not found | compose plugin not registered | use `docker-compose`, or add `cliPluginsExtraDirs` per `brew info docker-compose` |
| `build-index`: *N chunks have no cached vector* | manifest newer than the cache, or embedding interrupted | `fc embed`, then retry |
| `No manifest at …` | `fc embed` never completed for that chunker | `fc embed --chunker <name>` |

**Step through it:** `notebooks/stage2_3_walkthrough.ipynb` follows one filing
from raw HTML to search hits, calling these same functions cell by cell.

---

## Where it stands

```
fc embed         cold: 14,043 embedded in 917s     warm: 0 embedded in 0s
cache            14,043 vectors · 55 parts · 65.2 MiB
manifest         14,043 rows · 15 MB
fc build-index   14,043 documents verified by _count in 12–14s
                 run with OLLAMA_HOST pointed at a dead port — 0 embeddings
tests            346 offline + 3 live (Ollama ×2, OpenSearch ×1)
```

**Done when** (`PROJECT_PLAN.txt:340`): *a full rebuild from manifest runs in
minutes with zero re-embedding* — met, in seconds.

Spot checks against the live index:

| Query | Mode | Top hit |
|---|---|---|
| "allowance for credit losses" | BM25 | JPM FY2025 Item 15 — Note 13, *Allowance for credit losses* |
| "credit card net charge-off risk" | k-NN | CFG FY2025 Item 7 (cos 0.748); COF Items 7 and 1A follow |
| same, `ticker = SYF` | k-NN + filter | SYF FY2023 Item 1A |

These are sanity checks, not an evaluation. Whether retrieval is *good* is a
Stage 4 question with gold labels.

---

## Known gaps

- **The normalized full text is not persisted, nor its hash.** Stage 2 asked for
  both, so a normalization change would be *detectable*. The manifest stores each
  chunk's own text, which covers citation quoting, and a normalization change does
  change digests (so vectors never go stale). But nothing yet records *which*
  normalized string a chunk's offsets index into. Adding a `text_sha256` per filing
  to the manifest is the small fix.
- **Stage 2's shortfalls are in the index.** WFC and USB's six filings were
  chunked from SEC's wrapper document, not the real 10-K; JPM and Citigroup are
  partially sectioned. Their chunks are embedded and searchable as they are.
- **Overlapping sections are indexed twice.** In SYF FY2025, Item 3
  (chars 528,513–531,695) sits inside Item 8 (419,132–531,695), both from the
  filer's cross-reference table. `item_aware` chunks each section independently,
  so that text is embedded and searchable under two item labels. Different
  contextual prefixes mean different digests, so the cache does not deduplicate
  it. A Stage 2 fix (clip or reject nested cross-reference spans), found via the
  walkthrough notebook.
- **The index stores each vector twice** — in the HNSW graph and in `_source`.
  Excluding `embedding` from `_source` would roughly halve the 290 MB. Not done:
  it also hides the vector from inspection, and disk is not yet tight here.
- **768 dimensions only.** Matryoshka truncation to 256 is implemented but
  unmeasured; whether this Ollama build is v1.5 is unverified. Stage 4's A/B
  decides, and the model namespace in every key makes switching safe.
- **`fixed_window` is not embedded yet.** Needed for the ADR-0002 A/B in Stage 4;
  another ~15 minutes. Its prefix says "full document" rather than an item, so no
  digests overlap with `item_aware` and nothing is reused.

---

## Related decisions

- [ADR-0002](decisions/ADR-0002-item-aware-chunking.md) — item-aware chunking (the A/B this stage sets up)
- [ADR-0009](decisions/ADR-0009-embedding-model-nomic-embed-text-v15.md) — embedding model
- [ADR-0010](decisions/ADR-0010-reciprocal-rank-fusion-for-hybrid-search.md) — reciprocal rank fusion
- [ADR-0011](decisions/ADR-0011-local-first-development-aws-at-stage-8.md) — local first, AWS at Stage 8
- [ADR-0012](decisions/ADR-0012-opensearch-serverless-nextgen-never-classic.md) — Serverless NextGen, never Classic
