"""The set of companies this project ingests, loaded from ``config/corpus.yaml``.

Corpus growth is meant to be a config change rather than a code change, so the
list lives in YAML and nothing downstream hardcodes a ticker.

Two rules this module enforces on load, both of which fail far from their cause
if left to be discovered later:

* **CIKs are normalized and re-validated.** A CIK written without its leading
  zeros still looks plausible in YAML and produces a 404 several stages later.
* **Tickers and CIKs must be unique.** A duplicate silently halves a peer set,
  and the coverage matrix would report the shrunken corpus as complete.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..edgar.identifiers import InvalidCIKError, normalize_cik


class CorpusError(RuntimeError):
    """The corpus file is missing, malformed, or internally inconsistent.

    Carries a message meant for a human, not a traceback.
    """


@dataclass(frozen=True, slots=True)
class CorpusCompany:
    """One company in the ingestion set."""

    ticker: str
    cik: str
    name: str
    group: str
    lender: bool
    """Whether the company extends credit.

    Drives the coverage-matrix thresholds: a card network reports no allowance
    for credit losses, and that absence is correct rather than a gap.
    """


class Corpus:
    """The loaded company list, indexed by ticker and by CIK."""

    def __init__(self, companies: tuple[CorpusCompany, ...]) -> None:
        self._companies = companies
        self._by_ticker = {c.ticker: c for c in companies}
        self._by_cik = {c.cik: c for c in companies}

    @classmethod
    def load(cls, path: Path) -> Corpus:
        """Read and validate the corpus file at ``path``."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CorpusError(
                f"Could not read the corpus file at {path}.\n{exc}\n\n"
                "Set CORPUS_PATH in .env, or create the file."
            ) from exc
        return cls.from_yaml(raw)

    @classmethod
    def from_yaml(cls, raw: str | bytes) -> Corpus:
        """Parse corpus YAML.

        Takes text rather than a path so tests need no filesystem, mirroring
        :meth:`filing_copilot.edgar.identifiers.TickerResolver.from_json`.
        """
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise CorpusError(f"Corpus file is not valid YAML.\n{exc}") from exc

        if not isinstance(payload, dict) or "companies" not in payload:
            raise CorpusError("Corpus file must contain a top-level 'companies:' list.")

        rows = payload["companies"]
        if not isinstance(rows, list) or not rows:
            raise CorpusError("'companies:' must be a non-empty list.")

        companies = tuple(cls._parse_row(i, row) for i, row in enumerate(rows))
        cls._reject_duplicates(companies)
        return cls(companies)

    @staticmethod
    def _parse_row(index: int, row: Any) -> CorpusCompany:
        where = f"companies[{index}]"
        if not isinstance(row, dict):
            raise CorpusError(f"{where} is not a mapping.")

        missing = [k for k in ("ticker", "cik", "name", "group", "lender") if k not in row]
        if missing:
            raise CorpusError(f"{where} is missing required key(s): {', '.join(missing)}.")

        if not isinstance(row["lender"], bool):
            raise CorpusError(f"{where}.lender must be true or false, got {row['lender']!r}.")

        try:
            cik = normalize_cik(row["cik"])
        except InvalidCIKError as exc:
            raise CorpusError(f"{where}.cik is not a CIK: {exc}") from exc

        return CorpusCompany(
            ticker=str(row["ticker"]).strip().upper(),
            cik=cik,
            name=str(row["name"]).strip(),
            group=str(row["group"]).strip(),
            lender=row["lender"],
        )

    @staticmethod
    def _reject_duplicates(companies: tuple[CorpusCompany, ...]) -> None:
        for field in ("ticker", "cik"):
            seen: dict[str, int] = {}
            for c in companies:
                value = getattr(c, field)
                seen[value] = seen.get(value, 0) + 1
            dupes = sorted(v for v, n in seen.items() if n > 1)
            if dupes:
                raise CorpusError(f"Duplicate {field}(s) in corpus: {', '.join(dupes)}.")

    def by_ticker(self, ticker: str) -> CorpusCompany:
        """Look up a company by ticker, case-insensitively."""
        key = ticker.strip().upper()
        try:
            return self._by_ticker[key]
        except KeyError as exc:
            raise CorpusError(
                f"{key} is not in the corpus. Add it to the corpus file, "
                f"or pick one of: {', '.join(sorted(self._by_ticker))}."
            ) from exc

    def by_cik(self, cik: str | int) -> CorpusCompany:
        """Look up a company by CIK, in any SEC rendering."""
        try:
            key = normalize_cik(cik)
        except InvalidCIKError as exc:
            raise CorpusError(str(exc)) from exc
        try:
            return self._by_cik[key]
        except KeyError as exc:
            raise CorpusError(f"CIK {key} is not in the corpus.") from exc

    @property
    def ciks(self) -> tuple[str, ...]:
        """Every CIK, in file order."""
        return tuple(c.cik for c in self._companies)

    @property
    def lenders(self) -> tuple[CorpusCompany, ...]:
        """Companies that extend credit, for credit-specific coverage thresholds."""
        return tuple(c for c in self._companies if c.lender)

    def __iter__(self) -> Iterator[CorpusCompany]:
        return iter(self._companies)

    def __len__(self) -> int:
        return len(self._companies)
