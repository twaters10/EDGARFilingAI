"""DuckDB implementation of :class:`~filing_copilot.structured.backend.SqlBackend`.

DuckDB is in-process and reads the Parquet tree directly -- no server, no load
step. It is the local stand-in for Athena, and the SQL below is deliberately
literal rather than generated: the mechanics should be readable.

Every value is passed as a bound parameter. None of these inputs come from an end
user today, but the MCP tool layer in Stage 5 will accept arguments that do.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from .backend import ANNUAL_MAX_DAYS, ANNUAL_MIN_DAYS, Observation

# Hive partitioning is what lets a single-company filter read a single directory.
_SOURCE = "read_parquet(?, hive_partitioning = true)"

_COLUMNS = (
    'cik, tag, unit, period_type, start, "end", period_year, ' "val, fy, fp, form, filed, accn"
)


class DuckDBBackend:
    """Queries the local Parquet fact tree."""

    def __init__(self, facts_dir: Path) -> None:
        self._glob = str(facts_dir / "**" / "*.parquet")
        self._con = duckdb.connect(database=":memory:")

    def tags_present(
        self,
        ciks: tuple[str, ...],
        *,
        unit: str,
        period_type: str,
        years: tuple[int, int],
    ) -> dict[str, frozenset[str]]:
        rows = self._con.execute(
            f"""
            SELECT cik, tag
            FROM {_SOURCE}
            WHERE cik = ANY(?)
              AND unit = ?
              AND period_type = ?
              AND period_year BETWEEN ? AND ?
            GROUP BY cik, tag
            """,
            [self._glob, list(ciks), unit, period_type, years[0], years[1]],
        ).fetchall()

        found: dict[str, set[str]] = {cik: set() for cik in ciks}
        for cik, tag in rows:
            found[cik].add(tag)
        return {cik: frozenset(tags) for cik, tags in found.items()}

    def observations(
        self,
        cik: str,
        tags: tuple[str, ...],
        *,
        unit: str,
        period_type: str,
        period_year: int | None = None,
    ) -> list[Observation]:
        clause = "AND period_year = ?" if period_year is not None else ""
        params: list[Any] = [self._glob, cik, list(tags), unit, period_type]
        if period_year is not None:
            params.append(period_year)

        rows = self._con.execute(
            f"""
            SELECT {_COLUMNS}
            FROM {_SOURCE}
            WHERE cik = ?
              AND tag = ANY(?)
              AND unit = ?
              AND period_type = ?
              {clause}
            ORDER BY "end", filed
            """,
            params,
        ).fetchall()
        return [Observation(*row) for row in rows]

    def fiscal_year_end(self, cik: str, period_year: int) -> date | None:
        row = self._con.execute(
            f"""
            SELECT "end"
            FROM {_SOURCE}
            WHERE cik = ?
              AND period_type = 'duration'
              AND period_year = ?
              AND date_diff('day', start, "end") BETWEEN ? AND ?
            GROUP BY "end"
            -- Most-reported annual end date wins; ties break to the later date.
            ORDER BY count(*) DESC, "end" DESC
            LIMIT 1
            """,
            [self._glob, cik, period_year, ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS],
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> DuckDBBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
