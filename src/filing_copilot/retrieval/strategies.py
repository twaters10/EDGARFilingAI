"""Retrieval strategies: how a question is turned into searches, above the raw search.

A strategy decides *which* searches to run; :func:`~.search.search` runs one. The
layer exists because some questions cannot be answered by any single global
top-k, however good the ranking (``PROJECT_PLAN.txt`` §1.5):

* **Peer questions** -- "how do card issuers describe CECL?" A global top-10 gives
  *concentration*: the filer with the most, or wordiest, matching text takes most
  of the slots. :func:`fan_out_peers` gives *coverage*: one filtered search per
  company, so every peer is represented or explicitly reported as empty.
* **Section-targeted questions** -- "any material weakness?" belongs to Item 9A
  (and the auditor's report). :func:`section_targeted` scopes the search to the
  items where the answer lives, so boilerplate elsewhere cannot crowd it out.

Change detection (Stage 6) will also live here, as a strategy that does not search
at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..embed.encoder import Encoder
from ..index.opensearch import OpenSearchClient
from .filters import NO_FILTERS, SearchFilters
from .hits import Hit
from .search import Mode, search


@dataclass(frozen=True, slots=True)
class FanOutResult:
    """Hits interleaved across peers, plus the peers that returned nothing.

    ``empty`` is part of the answer, not an error: "Visa discloses nothing on this
    in its 10-K" is a finding, and a caller must be able to say it.
    """

    hits: list[Hit]
    per_company: dict[str, list[Hit]]
    empty: tuple[str, ...]


def fan_out_peers(
    client: OpenSearchClient,
    encoder: Encoder | None,
    index: str,
    query: str,
    peers: Sequence[str],
    *,
    n_per_company: int = 3,
    mode: Mode = "hybrid",
    filters: SearchFilters = NO_FILTERS,
) -> FanOutResult:
    """One filtered search per peer; results interleaved by rank.

    The assembled list is round-robin -- every peer's #1, then every peer's #2 --
    so the first ``len(peers)`` results already cover every company that has any.
    Ordering *within* a round follows ``peers`` as given; scores from different
    filtered searches are not compared, for the same reason RRF ignores scores.

    ``filters`` (years, items) apply to every peer; its ``tickers`` are replaced
    by each peer in turn.
    """
    tickers = [t.strip().upper() for t in peers]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"Duplicate peers in {list(peers)}.")

    per_company = {
        ticker: search(
            client,
            encoder,
            index,
            query,
            mode=mode,
            k=n_per_company,
            filters=filters.merge(SearchFilters(tickers=(ticker,))),
        )
        for ticker in tickers
    }

    interleaved: list[Hit] = []
    for round_ in range(n_per_company):
        for ticker in tickers:
            hits = per_company[ticker]
            if round_ < len(hits):
                interleaved.append(hits[round_])

    return FanOutResult(
        hits=[replace(hit, rank=rank) for rank, hit in enumerate(interleaved, start=1)],
        per_company=per_company,
        empty=tuple(t for t in tickers if not per_company[t]),
    )


def section_targeted(
    client: OpenSearchClient,
    encoder: Encoder | None,
    index: str,
    query: str,
    items: Sequence[str],
    *,
    k: int,
    mode: Mode = "hybrid",
    filters: SearchFilters = NO_FILTERS,
) -> list[Hit]:
    """A search restricted to chunks claimed by any of ``items``.

    A thin wrapper on purpose: the value is in naming the strategy -- so a caller
    says "screen Item 9A" rather than hand-assembling a filter -- not in logic.
    """
    if not items:
        raise ValueError("section_targeted needs at least one item, e.g. ['9A'].")
    scoped = filters.merge(SearchFilters.of(items=items))
    return search(client, encoder, index, query, mode=mode, k=k, filters=scoped)


def distinct_issuers(hits: Sequence[Hit]) -> int:
    """How many different companies a result list covers."""
    return len({hit.ticker for hit in hits})


def issuer_coverage(hits: Sequence[Hit], expected: Sequence[str]) -> float:
    """Fraction of ``expected`` companies with at least one hit. 1.0 if none expected."""
    wanted = {t.upper() for t in expected}
    if not wanted:
        return 1.0
    return len(wanted & {hit.ticker for hit in hits}) / len(wanted)
