# Stage 0 — Foundations and EDGAR citizenship

> **Maintained by hand.** This document is not regenerated when code changes. It
> describes the state of the code as of **2026-09-17** and is updated only on
> request. If a number here disagrees with the code, trust the code.

A repo that can politely fetch from SEC EDGAR and never leak a credential.

Everything in Stages 1–2 is built on one HTTP client. Three properties are
enforced *in that client* rather than left to the caller to remember: identity,
pacing, and caching. A convention that every future caller must honour is a
convention that will eventually be broken.

---

## The request path

```text
   caller asks for a URL
        │
        ▼
╔═══════════════════ EdgarClient ═══════════════ edgar/client.py ════════╗
║                                                                        ║
║   cache.get(url)  ──HIT──► return bytes            0 requests          ║
║        │                                                               ║
║       MISS                                                             ║
║        ▼                                                               ║
║   ┌─ retry loop, max 5 ──────────────────────────────────────────┐     ║
║   │                                                              │     ║
║   │   limiter.acquire()   ◄── INSIDE the loop, deliberately      │     ║
║   │        │                  a retry is a request and must      │     ║
║   │        │                  consume rate budget                │     ║
║   │        ▼                                                     │     ║
║   │   GET with User-Agent header                                 │     ║
║   │        │                                                     │     ║
║   │        ├── 2xx ────────────────────► break                   │     ║
║   │        ├── 429/500/502/503/504 ────► backoff, retry          │     ║
║   │        │      honours Retry-After, else 2^n with jitter      │     ║
║   │        └── any other 4xx ──────────► ✗ raise immediately     │     ║
║   │                a 404 will still be a 404 next time           │     ║
║   └──────────────────────────────────────────────────────────────┘     ║
║        │                                                               ║
║        ▼                                                               ║
║   cache.put(url, body, status, content_type, etag)                     ║
║        └─ writes <name> and <name>.meta.json with a sha256             ║
╚════════════════════════════╤═══════════════════════════════════════════╝
                             ▼
                        bytes, cached
```

Two entry points share that path. `get()` returns the body in memory;
`download()` streams to disk and returns a `Path`, because `companyfacts.zip` is
~1.2GB and must never have to fit in memory.

---

## Rate limiting

```text
   RateLimiter(5.0)                      edgar/throttle.py
        │
        ▼
   acquire()  ──► now < next_allowed?  ──YES──► sleep the difference
                        │                            │
                        NO                           ▼
                        └──────────────────────► next_allowed = now + 1/rate
                                                      │
                                                      ▼
                                                    return

   · time.monotonic, never time.time
        a system clock adjustment mid-ingestion cannot collapse the spacing
   · guarded by a threading.Lock
        single-threaded today; Stage 2 parallelises downloads, and an
        unsynchronised limiter fails silently exactly then
```

SEC permits 10 req/s and says the limit is "carefully monitored." The default is
**5** (ADR-0015). Getting blocked mid-ingestion costs hours and teaches nothing;
headroom is free at this corpus size.

---

## The cache layout is the contract

Every later stage depends on it, so the path is **derived from the URL**, not
hashed (ADR-0013):

```text
   https://www.sec.gov/Archives/edgar/data/927628/000092762826000024/cof-20251231.htm
                                           └────────────────────┬───────────────────┘
                                                                │ mirrored
   data/raw/                                                    ▼
   ├── reference/company_tickers.json          ← /files/company_tickers.json
   ├── submissions/CIK0001601712.json          ← /submissions/…
   ├── companyfacts/CIK0001601712.json         ← /api/xbrl/companyfacts/…
   ├── bulk/xbrl/companyfacts.zip              ← /Archives/edgar/daily-index/…
   ├── filings/927628/000092762826000024/
   │      ├── cof-20251231.htm
   │      └── cof-20251231.htm.meta.json
   └── misc/<host>/<path>                      ← anything unmatched
```

A hashed-filename cache is undebuggable at exactly the moment you need to debug
it — and you will be reading these files constantly in Stages 1–2.

Each sidecar records where a byte came from:

```json
{ "url": "...", "fetched_at": "...", "status": 200,
  "content_type": "text/html", "etag": null,
  "bytes": 8252507, "sha256": "b2f611cf62a6..." }
```

That `sha256` is what lets a Stage 5 citation be traced back to bytes SEC served.
Entries **never expire** (ADR-0017) — a filed 10-K is immutable; `refresh=True`
is the explicit override.

**Streamed downloads are atomic.** A 1.2GB fetch writes to `<name>.partial` and
is renamed into place only on success, so an interrupted download can never be
mistaken for a cached one.

---

## The three CIK renderings

The single most expensive trap in Stage 0, because it fails far from its cause.

```text
   SEC uses three forms of the same number:

     company_tickers.json  "cik_str"   1601712          ← an int, despite the name
     data.sec.gov                      CIK0001601712    ← 10-digit, padded, prefixed
     sec.gov/Archives/edgar/data/      1601712          ← unpadded

                               │
                    normalize_cik()   at every boundary
                               ▼
     canonical internal form   "0001601712"   ← 10-digit padded, no prefix

                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     to_data_api_cik()                 to_archives_cik()
     "CIK0001601712"                   "1601712"
```

Losing the padding produces a 404 several stages downstream. Never pass a bare
int across a module boundary, and never build a URL by hand —
`edgar/endpoints.py` spells every SEC URL exactly once, because `www.sec.gov` and
`data.sec.gov` are different hosts with different conventions.

---

## Configuration and secrets

`config.py` is the single settings source. Nothing else reads the environment.

```text
   environment  ──►  .env  ──►  Settings (pydantic-settings)
                                   │
                                   ├─ edgar_user_agent   ← validated, see below
                                   ├─ edgar_rate_limit   = 5.0   (0 < r <= 10)
                                   ├─ edgar_max_retries  = 5
                                   ├─ data_dir, corpus_path
                                   └─ aws_region, s3_bucket  ← declared, unused
                                                                until Stage 8, so
                                                                nothing is ever
                                                                hardcoded later
```

The User-Agent is validated at startup, not at request time:

```text
   empty?                              ──► refuse to start
   still the .env.example placeholder? ──► refuse to start
   contains no contact email?          ──► refuse to start
```

SEC requires a real contact address. Introducing ourselves under a fake identity
is the failure this prevents, and startup is the only honest place to catch it.

**This repository is public.** `.env` is git-ignored;
`tests/test_no_secrets.py` (7 tests) enforces that no secret, real address, or
bucket name appears in a tracked file.

---

## Commands

```bash
fc resolve --ticker SYF                 # ticker → canonical CIK
fc fetch-submissions --ticker SYF       # filing history into the cache
fc fetch-companyfacts                   # the ~1.2GB bulk archive, streamed
fc cache-info                           # entries, size, oldest/newest
```

**Done-when, verified:** `fc fetch-submissions --ticker SYF` run twice reports
`HIT` and `requests: 0` on the second run.

---

## Where it stands

```
data/raw   176 entries · 1,785.5 MiB      oldest 2026-09-15 · newest 2026-09-17
tests      56 across config, throttle, cache, client, download, identifiers,
           plus 7 secret-leak tests — all offline via httpx.MockTransport
```

The whole test suite makes **zero network calls**. Live-network tests are marked
`@pytest.mark.network` and deselected by default, which is what keeps CI free and
reproducible.

---

## Related decisions

- [ADR-0013](decisions/ADR-0013-cache-layout-mirrors-the-url-path.md) — cache layout mirrors the URL path
- [ADR-0014](decisions/ADR-0014-canonical-cik-form.md) — canonical CIK form
- [ADR-0015](decisions/ADR-0015-rate-limit-of-5-requests-per-second.md) — rate limit of 5 req/s
- [ADR-0016](decisions/ADR-0016-bulk-endpoints-over-per-company-api-calls.md) — bulk endpoints over per-company calls
- [ADR-0017](decisions/ADR-0017-cache-entries-never-expire.md) — cache entries never expire

Next: [Stage 1 — the structured layer](stage1_structured_layer.md)
