"""The OpenSearch index definition: one document per chunk, searchable two ways.

Written out as a plain dict rather than generated, so every field's type is a
line a reviewer can read and disagree with. Three kinds of field:

* **``text``** -- analysed with the ``english`` analyzer (stemming, stopwords)
  and scored by BM25, OpenSearch's default similarity. This is the lexical half
  of hybrid retrieval: it finds "CECL" and "Item 9A" when the vector half
  paraphrases them away.
* **``embedding``** -- a ``knn_vector`` searched by HNSW. The semantic half.
* **Filters** -- ``keyword`` / ``date`` / ``integer`` fields, exact-match only,
  for "Capital One, FY2024, Item 1A". Peer questions (Stage 4) depend on these:
  one filtered query per company, rather than trusting a global top-k to cover
  all of them.

**The vector space is inner product, and that is only correct because every
vector is unit length.** For unit vectors, inner product *is* cosine similarity,
without the per-comparison division. The guarantee comes from
:func:`~filing_copilot.embed.encoder.truncate_dimensions`, which every encoder
applies. Feed this index an unnormalized vector and ranking silently favours
long vectors over similar ones -- which is why ``build_index`` checks norms
before writing.
"""

from __future__ import annotations

from typing import Any

from ..embed.encoder import EmbeddingModel

# HNSW graph parameters. m=16 / ef_construction=128 are the common defaults for
# corpora far larger than 14k chunks; at this size recall is effectively exact
# and the build takes seconds, so there is nothing to tune yet. Stage 4 measures.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128


def index_body(model: EmbeddingModel) -> dict[str, Any]:
    """Settings and mappings for an index holding ``model``'s vectors."""
    return {
        "settings": {
            "index": {
                "knn": True,
                # One node locally. Serverless (Stage 8) manages its own shards.
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            # Refuse fields the mapping does not name. A typo in build_index
            # should fail the bulk request, not create a new, unfiltered field.
            "dynamic": "strict",
            "properties": {
                # --- lexical ------------------------------------------------
                "text": {"type": "text", "analyzer": "english"},
                # --- semantic -----------------------------------------------
                "embedding": {
                    "type": "knn_vector",
                    "dimension": model.dimensions,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "innerproduct",
                        "parameters": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
                    },
                },
                # --- filters ------------------------------------------------
                "chunk_id": {"type": "keyword"},
                "cik": {"type": "keyword"},
                "ticker": {"type": "keyword"},
                "accession": {"type": "keyword"},
                "form": {"type": "keyword"},
                # An array: a chunk Citigroup declares for Items 7 and 7A is
                # stored once with both, and a term filter matches either.
                "items": {"type": "keyword"},
                "section_path": {"type": "keyword"},
                "document": {"type": "keyword"},
                "period_end": {"type": "date", "format": "strict_date"},
                "fiscal_year": {"type": "integer"},
                # --- citation only: stored, never searched ------------------
                "char_start": {"type": "integer", "index": False},
                "char_end": {"type": "integer", "index": False},
                "text_sha256": {"type": "keyword", "index": False},
                "digest": {"type": "keyword", "index": False},
                "model": {"type": "keyword", "index": False},
            },
        },
    }
