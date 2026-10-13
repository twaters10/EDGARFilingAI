"""Metadata filters: "Capital One, FY2024, Item 1A" as exact matches, not query words.

Checkpoint A showed why a ticker belongs here rather than in the query text: BM25
cannot see a ticker that only appears in the embedding prefix, and the vector side
treats "SYF" as a vague semantic hint. A filter is neither -- it is a hard
constraint on ``keyword`` / ``integer`` fields (``index/mapping.py``).

**Where the filter goes matters, and getting it wrong is silent.** For kNN the
filter sits *inside* the ``knn`` clause, so OpenSearch (faiss, 2.19) searches only
matching vectors and still returns ``k`` of them. A ``post_filter`` instead
trims an unfiltered top-k afterwards. Measured on this index: "customer
concentration risk" for AAPL Item 1A returns 10 hits with the filter inside, and
**0** with ``post_filter`` -- the global top 10 held no Apple chunks, and nothing
was fetched to replace them. No error either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Exact-match constraints. Within a field, values are OR-ed; fields are AND-ed.

    ``SearchFilters(tickers=("COF", "SYF"), items=("1A",))`` means "Item 1A of
    Capital One *or* Synchrony". An empty field constrains nothing.
    """

    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    forms: tuple[str, ...] = ()
    items: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        tickers: Iterable[str] = (),
        fiscal_years: Iterable[int] = (),
        forms: Iterable[str] = (),
        items: Iterable[str] = (),
    ) -> SearchFilters:
        """Build from any iterables, normalizing case the way the index stores it."""
        return cls(
            tickers=tuple(t.strip().upper() for t in tickers),
            fiscal_years=tuple(int(y) for y in fiscal_years),
            forms=tuple(f.strip().upper() for f in forms),
            items=tuple(i.strip().upper() for i in items),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.tickers or self.fiscal_years or self.forms or self.items)

    def clauses(self) -> list[dict[str, Any]]:
        """One ``terms`` clause per constrained field, for a ``bool.filter``.

        ``items`` is an array field, so ``terms`` matches a chunk claimed by any of
        the requested items -- a Citigroup chunk declared for both 7 and 7A is found
        by a filter on either.
        """
        fields: tuple[tuple[str, tuple[Any, ...]], ...] = (
            ("ticker", self.tickers),
            ("fiscal_year", self.fiscal_years),
            ("form", self.forms),
            ("items", self.items),
        )
        return [{"terms": {field: list(values)}} for field, values in fields if values]

    def to_query(self) -> dict[str, Any]:
        """The filter as a query clause: ``{"bool": {"filter": [...]}}``."""
        return {"bool": {"filter": self.clauses()}}

    def merge(self, other: SearchFilters) -> SearchFilters:
        """``other``'s constraints replace this one's, field by field, where set."""
        return SearchFilters(
            tickers=other.tickers or self.tickers,
            fiscal_years=other.fiscal_years or self.fiscal_years,
            forms=other.forms or self.forms,
            items=other.items or self.items,
        )


NO_FILTERS = SearchFilters()
