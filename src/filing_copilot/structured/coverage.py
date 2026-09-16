"""The coverage matrix: which concepts resolve, for which companies, via which tag.

This is the cheapest early probe of the highest-likelihood data risk in the
project. It either validates the candidate lists in ``concepts.yaml`` or tells you
they are wrong while fixing them is still cheap -- as opposed to discovering it in
Stage 6 when a peer query quietly returns two companies.

A cell is not a boolean. It records the tag that answered and its rank in the
candidate list, because "resolved via the fourth-choice fallback" is a materially
different answer from "resolved via the preferred tag".
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import SqlBackend
from .corpus import Corpus
from .resolver import Concept, ConceptRegistry, Resolution, resolve_concept


@dataclass(frozen=True, slots=True)
class ConceptCoverage:
    """One concept's row in the matrix."""

    concept: Concept
    resolutions: dict[str, Resolution]
    expected_ciks: tuple[str, ...]
    """The companies this concept is measured against -- see Concept.applies_to."""

    @property
    def resolved_ciks(self) -> tuple[str, ...]:
        return tuple(c for c in self.expected_ciks if self.resolutions[c].resolved)

    @property
    def missing_ciks(self) -> tuple[str, ...]:
        return tuple(c for c in self.expected_ciks if not self.resolutions[c].resolved)

    @property
    def fallback_ciks(self) -> tuple[str, ...]:
        """Companies answered by something other than the preferred tag."""
        return tuple(
            c
            for c in self.expected_ciks
            if self.resolutions[c].resolved and (self.resolutions[c].rank or 0) > 0
        )

    @property
    def distinct_tags(self) -> tuple[str, ...]:
        """How many different tags it took to express one business concept."""
        tags = {self.resolutions[c].tag for c in self.resolved_ciks}
        return tuple(sorted(t for t in tags if t is not None))


@dataclass(frozen=True, slots=True)
class CoverageMatrix:
    """The full grid."""

    corpus: Corpus
    rows: tuple[ConceptCoverage, ...]

    def row(self, concept_name: str) -> ConceptCoverage:
        for row in self.rows:
            if row.concept.name == concept_name:
                return row
        raise KeyError(concept_name)


def build_matrix(
    registry: ConceptRegistry,
    backend: SqlBackend,
    corpus: Corpus,
    *,
    years: tuple[int, int],
) -> CoverageMatrix:
    """Resolve every concept across the corpus. One query per concept."""
    all_ciks = corpus.ciks
    lender_ciks = tuple(c.cik for c in corpus.lenders)

    rows = []
    for concept in registry:
        resolutions = resolve_concept(concept, backend, all_ciks, years=years)
        expected = lender_ciks if concept.applies_to == "lenders" else all_ciks
        rows.append(
            ConceptCoverage(concept=concept, resolutions=resolutions, expected_ciks=expected)
        )
    return CoverageMatrix(corpus=corpus, rows=tuple(rows))
