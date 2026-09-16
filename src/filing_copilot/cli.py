"""``fc`` — command line entry point for filing-copilot-mcp."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from typing import Annotated

import typer

from .config import ConfigError, Settings, get_settings
from .edgar import EdgarClient, TickerResolver, UnknownTickerError, normalize_cik
from .edgar.endpoints import company_tickers_url, companyfacts_bulk_url, submissions_url
from .edgar.identifiers import Company, InvalidCIKError
from .structured import (
    AS_REPORTED,
    AS_RESTATED,
    ConceptError,
    ConceptRegistry,
    Corpus,
    CorpusError,
    CoverageMatrix,
    DuckDBBackend,
    FinancialsResult,
    build_matrix,
    query_financials,
)
from .structured.ingest import CompanyResult, IngestError, ingest_corpus
from .structured.resolver import DEFAULT_CONCEPTS_PATH

app = typer.Typer(
    add_completion=False,
    help="Research assistant over SEC EDGAR filings.",
    no_args_is_help=True,
)

TickerOpt = Annotated[str | None, typer.Option("--ticker", "-t", help="Ticker symbol, e.g. SYF.")]
CikOpt = Annotated[str | None, typer.Option("--cik", help="CIK, in any SEC rendering.")]
RefreshOpt = Annotated[bool, typer.Option("--refresh", help="Bypass the cache and re-fetch.")]


def _load_settings() -> Settings:
    try:
        return get_settings()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _resolve(client: EdgarClient, ticker: str | None, cik: str | None) -> Company:
    """Turn --ticker or --cik into a Company. Exactly one must be supplied."""
    if (ticker is None) == (cik is None):
        typer.secho("Supply exactly one of --ticker or --cik.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    if cik is not None:
        try:
            return Company(cik=normalize_cik(cik), ticker="", title="")
        except InvalidCIKError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    assert ticker is not None  # narrowed by the check above
    resolver = TickerResolver.from_json(client.get(company_tickers_url()))
    try:
        return resolver.by_ticker(ticker)
    except UnknownTickerError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def resolve(ticker: TickerOpt = None, cik: CikOpt = None) -> None:
    """Resolve a ticker to a CIK (or normalize a CIK you already have)."""
    settings = _load_settings()
    with EdgarClient(settings) as client:
        company = _resolve(client, ticker, cik)
        label = f" · {company.title}" if company.title else ""
        typer.echo(f"CIK {company.cik}{label}")
        typer.echo(f"requests: {client.request_count}")


@app.command("fetch-submissions")
def fetch_submissions(
    ticker: TickerOpt = None,
    cik: CikOpt = None,
    refresh: RefreshOpt = False,
) -> None:
    """Fetch a company's filing history (submissions.json) into the local cache."""
    settings = _load_settings()
    with EdgarClient(settings) as client:
        company = _resolve(client, ticker, cik)
        url = submissions_url(company.cik)

        was_cached = client.cache.get(url) is not None and not refresh
        client.get(url, refresh=refresh)
        path = client.cache.path_for(url)

        status = "HIT" if was_cached else "MISS"
        typer.echo(f"{status}  {path}")
        typer.echo(f"requests: {client.request_count}")


# The archive is ~1.2GB and grows every quarter. Refuse below this much free
# space rather than filling the disk and failing partway through.
_BULK_HEADROOM_BYTES = 4 * 1024**3


@app.command("fetch-companyfacts")
def fetch_companyfacts(refresh: RefreshOpt = False) -> None:
    """Download the bulk XBRL archive (~1.2GB).

    This is one request instead of one per company (ADR-0016). The archive is
    NEVER extracted -- it expands to roughly 15GB. Members are read in place.
    """
    settings = _load_settings()
    url = companyfacts_bulk_url()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    with EdgarClient(settings) as client:
        cached = client.cache.path_for(url).is_file()
        if cached and not refresh:
            path = client.download(url)
            size_mb = path.stat().st_size / 1_048_576
            typer.echo(f"HIT   {path}  ({size_mb:,.0f} MiB)")
            typer.echo("requests: 0")
            return

        free = shutil.disk_usage(settings.raw_dir).free
        if free < _BULK_HEADROOM_BYTES:
            typer.secho(
                f"Only {free / 1024**3:.1f} GiB free; the archive needs headroom "
                f"beyond its ~1.2GB. Free some space, or fetch per-company facts "
                f"instead.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

        path = client.download(url, refresh=refresh, on_progress=_echo_progress())
        typer.echo("")  # close the progress line
        size_mb = path.stat().st_size / 1_048_576
        typer.echo(f"MISS  {path}  ({size_mb:,.0f} MiB)")
        typer.echo(f"requests: {client.request_count}")
        typer.secho("Do not extract this archive -- read members in place.", fg=typer.colors.YELLOW)


def _echo_progress(every_bytes: int = 25 * 1024**2) -> Callable[[int], None]:
    """Return a progress callback that reports at most once per ``every_bytes``."""
    state = {"next": every_bytes}

    def report(total: int) -> None:
        if total < state["next"]:
            return
        state["next"] = total + every_bytes
        typer.echo(f"\r  downloaded {total / 1_048_576:,.0f} MiB", nl=False, err=True)

    return report


@app.command("ingest-facts")
def ingest_facts() -> None:
    """Flatten the corpus's XBRL facts out of the archive into Parquet.

    Reads members in place; the archive is never extracted. Re-running replaces
    each company's partitions rather than appending to them.
    """
    settings = _load_settings()

    try:
        corpus = Corpus.load(settings.corpus_path)
    except CorpusError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    archive = EdgarClient(settings).cache.path_for(companyfacts_bulk_url())

    typer.echo(f"{'ticker':<8} {'rows':>9} {'tags':>6}  years")
    typer.echo("-" * 46)

    def report_company(result: CompanyResult) -> None:
        span = f"{result.earliest}-{result.latest}" if result.rows else "--"
        typer.echo(f"{result.ticker:<8} {result.rows:>9,} {result.tags:>6,}  {span}")

    try:
        report = ingest_corpus(archive, corpus, settings.facts_dir, on_company=report_company)
    except IngestError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("-" * 46)
    typer.echo(f"{'total':<8} {report.total_rows:>9,}")
    typer.echo(f"\nwrote {settings.facts_dir}")

    if report.empty_companies:
        # A company present in the archive but contributing nothing is always a
        # bug -- a wrong CIK, or a member with no us-gaap facts.
        names = ", ".join(c.ticker for c in report.empty_companies)
        typer.secho(f"No facts for: {names}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("cache-info")
def cache_info() -> None:
    """Summarize the on-disk response cache."""
    settings = _load_settings()
    stats = EdgarClient(settings).cache.stats()
    typer.echo(f"root:    {settings.raw_dir}")
    typer.echo(f"entries: {stats.entries}")
    typer.echo(f"size:    {stats.total_bytes / 1_048_576:.1f} MiB")
    if stats.oldest and stats.newest:
        typer.echo(f"oldest:  {stats.oldest.isoformat()}")
        typer.echo(f"newest:  {stats.newest.isoformat()}")


def main() -> None:  # pragma: no cover - thin wrapper
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()


@app.command("coverage")
def coverage(
    start_year: Annotated[int, typer.Option("--from", help="First period year.")] = 2022,
    end_year: Annotated[int, typer.Option("--to", help="Last period year.")] = 2025,
) -> None:
    """Report which concepts resolve for which companies, and via which tag.

    The cheapest probe of the project's largest data risk. A concept that
    resolves for two companies is not a working concept; run this before
    building anything on top of the mapping layer.
    """
    settings = _load_settings()

    try:
        corpus = Corpus.load(settings.corpus_path)
        registry = ConceptRegistry.load(settings.concepts_path or DEFAULT_CONCEPTS_PATH)
    except (CorpusError, ConceptError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not settings.facts_dir.exists():
        typer.secho(
            f"No fact table at {settings.facts_dir}. Run 'fc ingest-facts' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    with DuckDBBackend(settings.facts_dir) as backend:
        matrix = build_matrix(registry, backend, corpus, years=(start_year, end_year))

    _print_matrix(matrix, start_year, end_year)


def _print_matrix(matrix: CoverageMatrix, start_year: int, end_year: int) -> None:
    concepts = [row.concept.name for row in matrix.rows]
    width = max(len(name) for name in concepts) + 2

    typer.echo(f"Coverage matrix, period years {start_year}-{end_year}")
    typer.echo("digit = rank of the tag that answered (0 = preferred); . = not reported\n")

    header = "ticker  " + "".join(f"{name:<{width}}" for name in concepts)
    typer.echo(header)
    typer.echo("-" * len(header))

    for company in matrix.corpus:
        cells = []
        for row in matrix.rows:
            resolution = row.resolutions[company.cik]
            if company.cik not in row.expected_ciks:
                cells.append("n/a")
            elif resolution.resolved:
                cells.append(str(resolution.rank))
            else:
                cells.append(".")
        typer.echo(f"{company.ticker:<8}" + "".join(f"{c:<{width}}" for c in cells))

    typer.echo("")
    by_cik = {c.cik: c.ticker for c in matrix.corpus}
    exit_code = 0

    for row in matrix.rows:
        scope = "lenders" if row.concept.applies_to == "lenders" else "all"
        n, total = len(row.resolved_ciks), len(row.expected_ciks)
        typer.echo(f"{row.concept.name}  ({scope}: {n}/{total})")
        typer.echo(f"  tags used: {len(row.distinct_tags)}  {', '.join(row.distinct_tags) or '--'}")
        if row.fallback_ciks:
            names = ", ".join(f"{by_cik[c]}={row.resolutions[c].tag}" for c in row.fallback_ciks)
            typer.secho(f"  fallback:  {names}", fg=typer.colors.YELLOW)
        if row.missing_ciks:
            typer.secho(
                f"  MISSING:   {', '.join(by_cik[c] for c in row.missing_ciks)}",
                fg=typer.colors.RED,
            )
            exit_code = 1
        typer.echo("")

    if exit_code:
        typer.secho(
            "Some concepts do not resolve. That is a finding, not necessarily a bug -- "
            "companyfacts excludes company custom extension tags.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def financials(
    concept: Annotated[str, typer.Option("--concept", "-c", help="Business concept.")],
    ticker: Annotated[list[str], typer.Option("--ticker", "-t", help="Repeatable.")],
    fy: Annotated[int, typer.Option("--fy", help="Period year (the fact's own period).")],
    basis: Annotated[
        str, typer.Option("--basis", help="as_restated (default) or as_reported.")
    ] = AS_RESTATED,
) -> None:
    """Report an exact figure for a concept and period.

    Always shows the tag that answered and the basis. A figure whose source tag
    is hidden cannot safely be compared against another company's.
    """
    settings = _load_settings()

    if basis not in (AS_REPORTED, AS_RESTATED):
        typer.secho(
            f"--basis must be {AS_RESTATED!r} or {AS_REPORTED!r}, got {basis!r}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        corpus = Corpus.load(settings.corpus_path)
        registry = ConceptRegistry.load(settings.concepts_path or DEFAULT_CONCEPTS_PATH)
        companies = tuple(corpus.by_ticker(t) for t in ticker)
        registry[concept]  # fail early on an unknown concept, before querying
    except (CorpusError, ConceptError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not settings.facts_dir.exists():
        typer.secho(
            f"No fact table at {settings.facts_dir}. Run 'fc ingest-facts' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    with DuckDBBackend(settings.facts_dir) as backend:
        result = query_financials(
            registry,
            backend,
            corpus,
            concept_name=concept,
            companies=companies,
            period_year=fy,
            basis=basis,
        )

    _print_financials(result)
    if not result.values:
        raise typer.Exit(code=1)


def _print_financials(result: FinancialsResult) -> None:
    typer.echo(f"{result.concept.label}  ({result.concept.name})")
    typer.echo(f"period year {result.period_year} · basis {result.basis}\n")

    header = (
        f"{'ticker':<8}{'value':>20}  {'unit':<6}{'period end':<13}" f"{'tag':<54}{'filed':<12}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for value in result.values:
        flag = "  RESTATED" if value.restated else ""
        typer.echo(
            f"{value.ticker:<8}{value.val:>20,.0f}  {value.unit:<6}{value.end!s:<13}"
            f"{value.tag:<54}{value.filed!s:<12}{flag}"
        )

    for missing in result.unavailable:
        typer.secho(
            f"{missing.ticker:<8}{'unavailable':>20}  -- {missing.reason}",
            fg=typer.colors.YELLOW,
        )

    restated = [v for v in result.values if v.restated]
    if restated:
        typer.echo("")
        for value in restated:
            typer.echo(
                f"  {value.ticker} {value.concept} was restated: "
                f"as_reported {value.as_reported:,.0f} -> as_restated {value.as_restated:,.0f}"
            )

    for warning in result.warnings:
        typer.echo("")
        typer.secho(f"  {warning}", fg=typer.colors.YELLOW)
