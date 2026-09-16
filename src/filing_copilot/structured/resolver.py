"""Resolve a business concept to the tag a given company actually reports.

The problem this exists for: **"total debt" is not an XBRL tag.** One company
reports ``LongTermDebt``, another ``LongTermDebtNoncurrent``, a third
``DebtLongtermAndShorttermCombinedAmount``. A query written against any single
tag silently returns nothing for most of a peer set.

The mapping is declarative (``concepts.yaml``) so that adding a concept is a data
change. Resolution records **which** tag answered, because two companies answered
from different tags are not automatically comparable, and hiding that is how a
confidently wrong peer comparison gets built.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .backend import SqlBackend

# The concept file ships with the package rather than living in config/: it is
# part of how this project reads XBRL, not a per-deployment setting.
DEFAULT_CONCEPTS_PATH = Path(__file__).parent / "concepts.yaml"

FIRST_AVAILABLE = "first_available"
_RULES = frozenset({FIRST_AVAILABLE})
_SCOPES = frozenset({"all", "lenders"})


class ConceptError(RuntimeError):
    """The concept file is missing or malformed. Message is meant for a human."""


@dataclass(frozen=True, slots=True)
class Concept:
    """One business concept and the tags that may express it."""

    name: str
    label: str
    rule: str
    applies_to: str
    """``all`` or ``lenders`` -- scopes the coverage threshold, nothing else."""
    unit: str
    period_type: str
    candidates: tuple[str, ...]
    description: str


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving one concept for one company."""

    concept: str
    cik: str
    tag: str | None
    """The tag that answered, or ``None`` when the company reports none of them."""
    rank: int | None
    """Position in the candidate list. 0 is the preferred tag; higher is a fallback."""

    @property
    def resolved(self) -> bool:
        return self.tag is not None


class ConceptRegistry:
    """The loaded concept definitions."""

    def __init__(self, concepts: tuple[Concept, ...]) -> None:
        self._by_name = {c.name: c for c in concepts}

    @classmethod
    def load(cls, path: Path) -> ConceptRegistry:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConceptError(f"Could not read the concept file at {path}.\n{exc}") from exc
        return cls.from_yaml(raw)

    @classmethod
    def from_yaml(cls, raw: str | bytes) -> ConceptRegistry:
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConceptError(f"Concept file is not valid YAML.\n{exc}") from exc

        if not isinstance(payload, dict) or "concepts" not in payload:
            raise ConceptError("Concept file must contain a top-level 'concepts:' mapping.")

        body = payload["concepts"]
        if not isinstance(body, dict) or not body:
            raise ConceptError("'concepts:' must be a non-empty mapping.")

        return cls(tuple(cls._parse(name, node) for name, node in body.items()))

    @staticmethod
    def _parse(name: str, node: Any) -> Concept:
        if not isinstance(node, dict):
            raise ConceptError(f"Concept {name!r} is not a mapping.")

        required = ("rule", "applies_to", "unit", "period_type", "candidates")
        missing = [k for k in required if k not in node]
        if missing:
            raise ConceptError(f"Concept {name!r} is missing: {', '.join(missing)}.")

        if node["rule"] not in _RULES:
            raise ConceptError(
                f"Concept {name!r} has unknown rule {node['rule']!r}. "
                f"Known rules: {', '.join(sorted(_RULES))}."
            )
        if node["applies_to"] not in _SCOPES:
            raise ConceptError(
                f"Concept {name!r} has unknown applies_to {node['applies_to']!r}. "
                f"Use one of: {', '.join(sorted(_SCOPES))}."
            )

        candidates = node["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ConceptError(f"Concept {name!r} must list at least one candidate tag.")
        if len(set(candidates)) != len(candidates):
            raise ConceptError(f"Concept {name!r} lists a candidate tag twice.")

        return Concept(
            name=name,
            label=str(node.get("label", name)),
            rule=str(node["rule"]),
            applies_to=str(node["applies_to"]),
            unit=str(node["unit"]),
            period_type=str(node["period_type"]),
            candidates=tuple(str(c) for c in candidates),
            description=str(node.get("description", "")).strip(),
        )

    def __getitem__(self, name: str) -> Concept:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ConceptError(
                f"Unknown concept {name!r}. Known: {', '.join(sorted(self._by_name))}."
            ) from exc

    def __iter__(self) -> Iterator[Concept]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)


def resolve_from_tags(concept: Concept, present: frozenset[str], cik: str) -> Resolution:
    """Pick the first candidate the company actually reports.

    Pure: takes the set of tags already known to be present, so the whole matrix
    costs one query per concept rather than one per company per candidate.
    """
    for rank, tag in enumerate(concept.candidates):
        if tag in present:
            return Resolution(concept=concept.name, cik=cik, tag=tag, rank=rank)
    return Resolution(concept=concept.name, cik=cik, tag=None, rank=None)


def resolve_concept(
    concept: Concept,
    backend: SqlBackend,
    ciks: tuple[str, ...],
    *,
    years: tuple[int, int],
) -> dict[str, Resolution]:
    """Resolve one concept across many companies in a single query."""
    present = backend.tags_present(
        ciks, unit=concept.unit, period_type=concept.period_type, years=years
    )
    return {cik: resolve_from_tags(concept, present[cik], cik) for cik in ciks}
