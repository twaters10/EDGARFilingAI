"""One search result, and how to read it back out of an OpenSearch response.

A :class:`Hit` is the unit every later layer consumes: fusion (Stage 4) re-ranks
lists of them, the eval harness scores them against gold spans, and the Stage 5
``search_filings`` tool returns them with citations. So it carries everything a
citation needs -- ``accession``, ``document``, ``char_start``/``char_end`` -- and
nothing that only matters to OpenSearch.

**The vector is never read back.** Every search excludes ``embedding`` from
``_source``: 768 floats is ~15 KB of JSON per hit, the largest part of the response,
and nothing downstream uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Hit:
    """A chunk as one retriever ranked it.

    ``rank`` is 1-based and is what fusion uses. ``score`` is whatever the
    retriever reported -- a BM25 score, or OpenSearch's transform of an inner
    product -- and is only comparable *within* one result list, never across two.
    That is the property RRF is built around (:mod:`.fusion`).
    """

    rank: int
    score: float
    chunk_id: str
    cik: str
    ticker: str
    accession: str
    form: str
    fiscal_year: int
    items: tuple[str, ...]
    section_path: str
    document: str
    char_start: int
    char_end: int
    text: str
    sources: tuple[tuple[str, int], ...] = ()
    """Where each input list ranked this chunk, e.g. ``(("vector", 31), ("bm25", 2))``.

    Set by fusion only, so a fused result can explain itself. Empty for a hit
    straight from one retriever, whose own ``rank`` already says it all.
    """

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> list[Hit]:
        """Every hit in a ``_search`` response, ranked in the order returned."""
        return [
            cls.from_source(rank, float(raw["_score"]), raw["_source"])
            for rank, raw in enumerate(response["hits"]["hits"], start=1)
        ]

    @classmethod
    def from_source(cls, rank: int, score: float, source: dict[str, Any]) -> Hit:
        return cls(
            rank=rank,
            score=score,
            chunk_id=source["chunk_id"],
            cik=source["cik"],
            ticker=source["ticker"],
            accession=source["accession"],
            form=source["form"],
            fiscal_year=int(source["fiscal_year"]),
            items=tuple(source["items"]),
            section_path=source["section_path"],
            document=source["document"],
            char_start=int(source["char_start"]),
            char_end=int(source["char_end"]),
            text=source["text"],
        )
