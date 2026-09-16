"""Structured financial layer: XBRL facts -> Parquet -> SQL.

Exact figures come from here. The LLM never does arithmetic on retrieved text;
it calls into this layer or the figure does not appear in an answer.
"""

from .backend import Observation, SqlBackend
from .corpus import Corpus, CorpusCompany, CorpusError
from .coverage import CoverageMatrix, build_matrix
from .duckdb_backend import DuckDBBackend
from .financials import (
    AS_REPORTED,
    AS_RESTATED,
    Basis,
    ConceptValue,
    FinancialsResult,
    query_financials,
)
from .resolver import Concept, ConceptError, ConceptRegistry, Resolution

__all__ = [
    "AS_REPORTED",
    "AS_RESTATED",
    "Basis",
    "Concept",
    "ConceptError",
    "ConceptRegistry",
    "ConceptValue",
    "Corpus",
    "CorpusCompany",
    "CorpusError",
    "CoverageMatrix",
    "DuckDBBackend",
    "FinancialsResult",
    "Observation",
    "Resolution",
    "SqlBackend",
    "build_matrix",
    "query_financials",
]
