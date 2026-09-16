"""Read corpus members out of companyfacts.zip and write the Parquet fact table.

The archive is opened once and members are read **in place** with :mod:`zipfile`.
It expands to roughly 15GB, which does not fit on the development machine, so it
is never extracted (ADR-0016).

Output is a Hive-partitioned tree::

    data/processed/facts/cik=0001601712/period_year=2024/*.parquet

Partitioning is what makes a single-company query read a single directory
instead of the whole table -- the same pruning Athena performs in Stage 8. Note
the partition key is ``period_year`` (derived from the fact's own period), not
SEC's ``fy`` (the fiscal year of the report that contained the fact). See
:mod:`filing_copilot.structured.flatten` for why that distinction matters.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .corpus import Corpus, CorpusCompany
from .flatten import FactRow, iter_fact_rows

# Explicit rather than inferred: an inferred schema can silently change type
# between runs when a column happens to be all-null for one company.
FACT_SCHEMA = pa.schema(
    [
        pa.field("cik", pa.string(), nullable=False),
        pa.field("taxonomy", pa.string(), nullable=False),
        pa.field("tag", pa.string(), nullable=False),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("period_type", pa.string(), nullable=False),
        pa.field("start", pa.date32(), nullable=True),
        pa.field("end", pa.date32(), nullable=False),
        pa.field("period_year", pa.int32(), nullable=False),
        pa.field("val", pa.float64(), nullable=False),
        pa.field("fy", pa.int32(), nullable=True),
        pa.field("fp", pa.string(), nullable=True),
        pa.field("form", pa.string(), nullable=True),
        pa.field("filed", pa.date32(), nullable=False),
        pa.field("accn", pa.string(), nullable=False),
        pa.field("frame", pa.string(), nullable=True),
    ]
)


class IngestError(RuntimeError):
    """Ingestion could not proceed. Carries a message meant for a human."""


@dataclass
class CompanyResult:
    """What one company contributed."""

    ticker: str
    cik: str
    rows: int
    tags: int
    earliest: int | None
    latest: int | None


@dataclass
class IngestReport:
    """Summary of a full ingest run."""

    companies: list[CompanyResult] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(c.rows for c in self.companies)

    @property
    def empty_companies(self) -> list[CompanyResult]:
        """Companies that produced no rows -- always a bug worth surfacing."""
        return [c for c in self.companies if c.rows == 0]


def member_name(cik: str) -> str:
    """The archive member holding one company's facts."""
    return f"CIK{cik}.json"


def rows_to_table(rows: Iterable[FactRow]) -> pa.Table:
    """Build an Arrow table with the declared schema, column by column."""
    materialized = list(rows)
    columns = {name: [getattr(row, name) for row in materialized] for name in FACT_SCHEMA.names}
    return pa.Table.from_pydict(columns, schema=FACT_SCHEMA)


def ingest_company(
    archive: zipfile.ZipFile,
    company: CorpusCompany,
    facts_dir: Path,
) -> CompanyResult:
    """Flatten one company's facts and write its partitions."""
    name = member_name(company.cik)
    try:
        with archive.open(name) as handle:
            payload = handle.read()
    except KeyError as exc:
        raise IngestError(
            f"{company.ticker} (CIK {company.cik}) is not in the archive as {name}. "
            "The archive may be stale, or the CIK in the corpus file is wrong."
        ) from exc

    rows = list(iter_fact_rows(payload, cik=company.cik))
    table = rows_to_table(rows)

    if table.num_rows:
        pq.write_to_dataset(
            table,
            root_path=str(facts_dir),
            partition_cols=["cik", "period_year"],
            # Re-running an ingest must replace a company's partitions, not
            # append to them. Without this, every run doubles the row count.
            existing_data_behavior="delete_matching",
        )

    years = [row.period_year for row in rows]
    return CompanyResult(
        ticker=company.ticker,
        cik=company.cik,
        rows=len(rows),
        tags=len({(row.taxonomy, row.tag) for row in rows}),
        earliest=min(years) if years else None,
        latest=max(years) if years else None,
    )


def ingest_corpus(
    archive_path: Path,
    corpus: Corpus,
    facts_dir: Path,
    *,
    on_company: Callable[[CompanyResult], None] | None = None,
) -> IngestReport:
    """Ingest every company in ``corpus``. Opens the archive exactly once."""
    if not archive_path.is_file():
        raise IngestError(f"No archive at {archive_path}. Run 'fc fetch-companyfacts' first.")

    facts_dir.mkdir(parents=True, exist_ok=True)
    report = IngestReport()

    with zipfile.ZipFile(archive_path) as archive:
        for company in corpus:
            result = ingest_company(archive, company, facts_dir)
            report.companies.append(result)
            if on_company is not None:
                on_company(result)

    return report
