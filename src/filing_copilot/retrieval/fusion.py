"""Reciprocal Rank Fusion: combine ranked lists using ranks alone (ADR-0010).

Each chunk scores ``sum over lists of 1 / (k + rank)``, where ``rank`` is its
1-based position in a list, and lists that did not return it contribute nothing.

**Why ranks, not scores.** A BM25 score and a cosine similarity are on unrelated
scales -- BM25 is unbounded and depends on the query's terms; cosine here sits in
a narrow band around 0.72 for *unrelated* chunks. Blending them needs a
normalization, and every normalization is a tuning knob. Ranks need none, which
is why RRF is the default until the eval harness can justify anything else.

**What k does.** It flattens the curve. With ``k = 60``:

* rank 1 in one list scores ``1/61 = 0.0164``;
* rank 30 in *both* lists scores ``2/90 = 0.0222``.

So a chunk both retrievers find, even modestly, beats a chunk only one of them
loves. Agreement is the signal. A small ``k`` would let one list's top hit
dominate; a large one would make every rank nearly equal. 60 is the value from
Cormack, Clarke & Buettcher (2009), and it stays fixed until the eval exists.

**What RRF cannot do.** It does not know which retriever is right for a given
query. When one list is confidently wrong, its top ranks still earn the same
credit as the other list's -- fusion averages, it does not adjudicate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .hits import Hit

#: Cormack, Clarke & Buettcher (2009). Fixed by ADR-0010 -- no tuning before the eval.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Hit]], *, k: int = RRF_K, limit: int | None = None
) -> list[Hit]:
    """Fuse named result lists into one ranking.

    Args:
        ranked_lists: retriever name -> its hits, best first. Each hit's own
            ``rank`` is used, so a list does not need to start at 1 or be dense.
        k: the RRF constant. See the module docstring.
        limit: keep only the top ``limit`` fused hits.

    Returns:
        Hits re-ranked from 1, with ``score`` set to the RRF score and
        ``sources`` recording where each input list ranked them. Ties are
        broken by ``chunk_id``, so the output does not depend on dict order.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}.")

    scores: dict[str, float] = {}
    sources: dict[str, list[tuple[str, int]]] = {}
    representative: dict[str, Hit] = {}

    for retriever, hits in ranked_lists.items():
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
            sources.setdefault(hit.chunk_id, []).append((retriever, hit.rank))
            # Every list stores the same chunk fields; which copy is kept is immaterial.
            representative.setdefault(hit.chunk_id, hit)

    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    if limit is not None:
        order = order[:limit]

    return [
        replace(
            representative[chunk_id],
            rank=rank,
            score=scores[chunk_id],
            sources=tuple(sources[chunk_id]),
        )
        for rank, chunk_id in enumerate(order, start=1)
    ]
