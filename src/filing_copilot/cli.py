"""``fc`` — command line entry point for filing-copilot-mcp."""

from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated

import typer

from .config import ConfigError, Settings, get_settings
from .edgar import EdgarClient, TickerResolver, UnknownTickerError, normalize_cik
from .edgar.endpoints import company_tickers_url, companyfacts_bulk_url, submissions_url
from .edgar.identifiers import Company, InvalidCIKError
from .embed import (
    NOMIC_EMBED_TEXT,
    EmbeddingCache,
    EmbeddingModel,
    EncoderError,
    OllamaEncoder,
    embed_missing,
    plan_corpus,
)
from .filings import (
    CHUNKERS,
    TARGET_ITEMS,
    CoverageReport,
    FilingCoverage,
    FilingDocument,
    FilingText,
    build_filing_text,
    build_report,
    download_corpus,
    load_filing_texts,
    verify_offsets,
)
from .filings.download import MAX_HISTORY_PAGES, CompanyDownload, download_company
from .filings.index import IncompleteFilingIndexError
from .index import (
    IndexBuildError,
    OpenSearchClient,
    OpenSearchError,
    build_index,
    index_name,
    manifest_path,
    read_manifest,
    write_manifest,
)
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


# --- Stage 2: filing text ------------------------------------------------


@app.command("fetch-filings")
def fetch_filings(
    years: Annotated[int, typer.Option("--years", help="Annual filings per company.")] = 3,
    max_pages: Annotated[
        int, typer.Option("--max-pages", help="History pages to walk back per company.")
    ] = MAX_HISTORY_PAGES,
    refresh: RefreshOpt = False,
) -> None:
    """Download the corpus's 10-K primary documents.

    Index completeness is asserted per company *before* any document is
    fetched. A short corpus that downloads cleanly is the failure mode this
    guards against -- the gap would otherwise surface as a thin answer several
    stages later, far from its cause.
    """
    settings = _load_settings()

    try:
        corpus = Corpus.load(settings.corpus_path)
    except CorpusError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"{'ticker':<8} {'filings':>7} {'fetched':>7} {'MiB':>7}  years")
    typer.echo("-" * 52)

    def report_company(result: CompanyDownload) -> None:
        years_seen = sorted({d.ref.fiscal_year for d in result.documents})
        span = f"{years_seen[0]}-{years_seen[-1]}" if years_seen else "--"
        size = sum(d.size_bytes for d in result.documents) / 1_048_576
        typer.echo(
            f"{result.ticker:<8} {len(result.documents):>7} {result.fetched:>7} "
            f"{size:>7,.1f}  {span}"
        )

    with EdgarClient(settings) as client:
        try:
            report = download_corpus(
                client,
                corpus,
                years=years,
                refresh=refresh,
                max_pages=max_pages,
                on_company=report_company,
            )
        except IncompleteFilingIndexError as exc:
            typer.secho(f"\n{exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        typer.echo("-" * 52)
        typer.echo(
            f"{'total':<8} {len(report.documents):>7} "
            f"{'':>7} {report.total_bytes / 1_048_576:>7,.1f}"
        )
        typer.echo(f"requests: {client.request_count}")

    if report.failures:
        typer.echo("")
        for failure in report.failures:
            typer.secho(f"  FAILED  {failure}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("sections")
def sections(
    years: Annotated[int, typer.Option("--years", help="Annual filings per company.")] = 3,
    threshold: Annotated[
        float, typer.Option("--threshold", help="Required share with all core items.")
    ] = 0.90,
    chunker: Annotated[
        str, typer.Option("--chunker", help="Chunker to size: item_aware or fixed_window.")
    ] = "item_aware",
) -> None:
    """Parser coverage report over the downloaded filings.

    Reads only from the cache and makes no network requests when warm. Every
    shortfall is listed by name -- the point of this report is the failures,
    not the percentage.
    """
    settings = _load_settings()
    _check_chunker(chunker)
    docs = _load_corpus_texts(settings, years)

    typer.echo(f"{'ticker':<8}{'FY':<6}{'chars':>9}  {'strategy':<15}{'state':<9}items")
    typer.echo("-" * 72)

    def report_filing(filing: FilingCoverage) -> None:
        colour = {"located": None, "partial": typer.colors.YELLOW, "failed": typer.colors.RED}
        typer.secho(
            f"{filing.ticker:<8}{filing.fiscal_year:<6}{filing.total_chars:>9,}  "
            f"{filing.strategy:<15}{filing.state:<9}{','.join(filing.found) or '--'}",
            fg=colour[filing.state],
        )

    report = build_report(docs, on_filing=report_filing)
    _print_coverage_summary(report, docs, threshold, chunker)

    if not report.meets(threshold):
        raise typer.Exit(code=1)


def _load_corpus_texts(settings: Settings, years: int) -> list[FilingText]:
    """Every corpus filing as :class:`FilingText`, read from the download cache.

    Shared by ``sections`` and ``embed`` so the coverage report describes the
    exact text that gets indexed. Makes no requests when the cache is warm, and
    says so when it was not.
    """
    try:
        corpus = Corpus.load(settings.corpus_path)
    except CorpusError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    with EdgarClient(settings) as client:
        try:
            downloaded = download_corpus(client, corpus, years=years)
        except IncompleteFilingIndexError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        if client.request_count:
            typer.secho(
                f"note: {client.request_count} request(s) -- the cache was cold.",
                fg=typer.colors.YELLOW,
            )

    docs = load_filing_texts(downloaded, corpus)
    if not docs:
        typer.secho(
            "No filings on disk. Run 'fc fetch-filings' first.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)
    return docs


def _print_coverage_summary(
    report: CoverageReport, docs: list[FilingText], threshold: float, chunker: str
) -> None:
    typer.echo("-" * 72)
    counts = "  ".join(f"{name}={n}" for name, n in sorted(report.by_strategy.items()))
    typer.echo(
        f"core items {', '.join(('1A', '7', '9A'))} located in "
        f"{len(report.located)}/{len(report.filings)} ({report.rate:.0%})   {counts}"
    )

    for filing in report.shortfalls:
        typer.secho(
            f"\n  {filing.state.upper():<8} {filing.ticker} FY{filing.fiscal_year} "
            f"({filing.document})",
            fg=typer.colors.YELLOW if filing.state == "partial" else typer.colors.RED,
        )
        typer.echo(f"    {filing.reason}")
        if filing.missing:
            typer.echo(f"    not located: {', '.join(filing.missing)}")
        if filing.implausible:
            typer.echo(f"    too short to be real: {', '.join(filing.implausible)}")
        if filing.incorporated:
            typer.echo(f"    incorporated by reference: {', '.join(filing.incorporated)}")

    problems = [p for doc in docs for p in verify_offsets(doc)]
    typer.echo("")
    if problems:
        for problem in problems:
            typer.secho(f"  OFFSETS  {problem}", fg=typer.colors.RED, err=True)
    else:
        typer.echo(f"offsets round-trip: OK across {len(docs)} filings")

    split = CHUNKERS[chunker]
    chunks = [c for doc in docs for c in split(doc)]
    if chunks:
        tokens = sorted(c.tokens for c in chunks)
        typer.echo(
            f"{chunker}: {len(chunks):,} chunks  "
            f"median {tokens[len(tokens) // 2]} tokens  "
            f"p95 {tokens[int(len(tokens) * 0.95)]} tokens"
        )

    if not report.meets(threshold):
        typer.secho(
            f"\nBelow the {threshold:.0%} threshold. Every shortfall is listed above.",
            fg=typer.colors.RED,
        )


@app.command("show")
def show(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Corpus ticker, e.g. COF.")],
    item: Annotated[str, typer.Option("--item", "-i", help="Item number, e.g. 1A.")] = "1A",
    fy: Annotated[
        int | None, typer.Option("--fy", help="Fiscal year. Defaults to the most recent.")
    ] = None,
    chars: Annotated[int, typer.Option("--chars", help="Characters shown at each end.")] = 800,
    full: Annotated[bool, typer.Option("--full", help="Print the whole section.")] = False,
    offsets: Annotated[bool, typer.Option("--offsets", help="Print offsets only.")] = False,
) -> None:
    """Print a located section's text, with the offsets it came from.

    The inspection tool for everything Stage 2 asserts. 'Located' only means a
    span of plausible length was found -- whether it holds the right content is
    a question only reading it can answer.
    """
    settings = _load_settings()
    item = item.strip().upper()

    try:
        corpus = Corpus.load(settings.corpus_path)
        company = corpus.by_ticker(ticker)
    except CorpusError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    with EdgarClient(settings) as client:
        try:
            downloaded = download_company(client, company, years=_SHOW_YEARS)
        except IncompleteFilingIndexError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    document = _pick_filing(downloaded.documents, company.ticker, fy)
    filing = build_filing_text(document, company.name)
    text, sections = filing.text, filing.sections

    located = sections.get(item)
    if located is None:
        found = ", ".join(i for i in TARGET_ITEMS if i in sections) or "none"
        typer.secho(
            f"{company.ticker} FY{document.ref.fiscal_year}: Item {item} was not located.\n"
            f"Located: {found}.\nRun 'fc sections' for why it was missed.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"{company.ticker} FY{document.ref.fiscal_year} · {document.ref.form} · "
        f"Item {item} · {located.strategy}"
    )
    typer.echo(
        f"chars {located.char_start:,}-{located.char_end:,} "
        f"({located.length:,} long) of {len(text):,} · {document.ref.accession}"
    )
    typer.echo(f"document: {filing.document_at(located.char_start)}")
    for extra_start, extra_end in located.extra_spans:
        # Only a crossref section has these, and they are the least obvious thing
        # about its output -- a filer declaring "8 - 24, 80 - 85" gets two blocks.
        typer.echo(
            f"  also: chars {extra_start:,}-{extra_end:,} "
            f"({extra_end - extra_start:,} long, {filing.document_at(extra_start)})"
        )

    if offsets:
        return

    body = text[located.char_start : located.char_end]
    typer.echo("-" * 72)
    if full or len(body) <= chars * 2:
        typer.echo(body)
        return
    typer.echo(body[:chars])
    typer.secho(f"\n[... {len(body) - chars * 2:,} characters elided ...]\n", dim=True)
    typer.echo(body[-chars:])


# Enough years that --fy can select a back year; a warm cache makes this free.
_SHOW_YEARS = 3


def _pick_filing(documents: list[FilingDocument], ticker: str, fy: int | None) -> FilingDocument:
    """The requested fiscal year, or the most recent filing on disk."""
    if not documents:
        typer.secho(
            f"No cached filings for {ticker}. Run 'fc fetch-filings' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if fy is None:
        return max(documents, key=lambda d: d.ref.fiscal_year)

    for document in documents:
        if document.ref.fiscal_year == fy:
            return document

    years = ", ".join(
        str(d.ref.fiscal_year) for d in sorted(documents, key=lambda d: -d.ref.fiscal_year)
    )
    typer.secho(
        f"{ticker} has no FY{fy} filing on disk. Available: {years}.", fg=typer.colors.RED, err=True
    )
    raise typer.Exit(code=2)


# --- Stage 3: embeddings --------------------------------------------------------


def _embedding_model(settings: Settings) -> EmbeddingModel:
    """The configured model, refusing any whose prefix behaviour is unverified.

    ``needs_task_prefix`` and ``max_input_chars`` were measured for nomic on this
    Ollama build. Accepting another name here would reuse those measurements for a
    model they were never taken on -- a silent retrieval regression (ADR-0009).
    """
    if settings.embedding_model != NOMIC_EMBED_TEXT.name:
        typer.secho(
            f"EMBEDDING_MODEL={settings.embedding_model!r} is not verified. Only "
            f"{NOMIC_EMBED_TEXT.name!r} has measured prefix and input-length behaviour "
            f"(see ADR-0009 and tests/test_encoder.py).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return replace(NOMIC_EMBED_TEXT, dimensions=settings.embedding_dimensions)


def _check_chunker(chunker: str) -> None:
    if chunker not in CHUNKERS:
        typer.secho(
            f"--chunker must be one of {', '.join(sorted(CHUNKERS))}, got {chunker!r}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)


ChunkerOpt = Annotated[
    str, typer.Option("--chunker", help="item_aware (default) or fixed_window, the A/B baseline.")
]


@app.command("embed")
def embed(
    chunker: ChunkerOpt = "item_aware",
    years: Annotated[int, typer.Option("--years", help="Annual filings per company.")] = 3,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Count cache hits and misses; embed nothing.")
    ] = False,
) -> None:
    """Chunk the corpus, embed only what the cache lacks, and write the manifest.

    Safe to re-run: a second run over an unchanged corpus embeds nothing. An
    interrupted run keeps every completed batch.
    """
    settings = _load_settings()
    _check_chunker(chunker)
    model = _embedding_model(settings)

    docs = _load_corpus_texts(settings, years)
    plan = plan_corpus(docs, CHUNKERS[chunker], model)
    cache = EmbeddingCache(settings.embeddings_dir)
    missing = len(plan.inputs.keys() - cache.cached_digests(model))

    typer.echo(
        f"{chunker}: {len(docs)} filings, {len(plan.rows):,} chunks, "
        f"{len(plan.inputs):,} distinct inputs for {model.cache_key}"
    )
    typer.echo(f"cached: {len(plan.inputs) - missing:,}   to embed: {missing:,}")
    if dry_run:
        return

    started = time.monotonic()

    def report(done: int, total: int) -> None:
        elapsed = time.monotonic() - started
        remaining = elapsed / done * (total - done)
        typer.echo(f"  {done:,}/{total:,}  {elapsed:,.0f}s elapsed, ~{remaining:,.0f}s left")

    try:
        with OllamaEncoder(host=settings.ollama_host, model=model) as encoder:
            result = embed_missing(encoder, cache, plan.inputs, on_batch=report)
    except EncoderError as exc:
        typer.secho(
            f"{exc}\nCompleted batches are saved; re-run to resume.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from exc

    path = manifest_path(settings.manifests_dir, chunker, model.cache_key)
    write_manifest(plan.rows, path)

    stats = cache.stats(model)
    typer.echo(
        f"embedded {result.embedded:,} in {time.monotonic() - started:,.0f}s; "
        f"{result.already_cached:,} were already cached"
    )
    typer.echo(
        f"cache:    {stats.vectors:,} vectors, {stats.parts} part(s), "
        f"{stats.total_bytes / 1_048_576:.1f} MiB  ({cache.directory_for(model)})"
    )
    typer.echo(f"manifest: {len(plan.rows):,} rows  ({path})")


@app.command("build-index")
def build_index_command(chunker: ChunkerOpt = "item_aware") -> None:
    """Rebuild the search index from the manifest and the embedding cache.

    Makes no EDGAR requests and no Ollama calls: everything it indexes was stored
    by ``fc embed``. If anything is missing it stops before the old index is
    touched.
    """
    settings = _load_settings()
    _check_chunker(chunker)
    model = _embedding_model(settings)

    path = manifest_path(settings.manifests_dir, chunker, model.cache_key)
    if not path.exists():
        typer.secho(
            f"No manifest at {path}. Run 'fc embed --chunker {chunker}' first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    started = time.monotonic()
    rows = read_manifest(path)
    name = index_name(settings.opensearch_index_prefix, chunker)
    typer.echo(f"{name}: {len(rows):,} rows from {path}")

    def report(done: int, total: int) -> None:
        typer.echo(f"  {done:,}/{total:,}  {time.monotonic() - started:,.1f}s")

    try:
        with OpenSearchClient(settings.opensearch_url) as client:
            typer.echo(f"OpenSearch {client.ping()} at {settings.opensearch_url}")
            result = build_index(
                rows, EmbeddingCache(settings.embeddings_dir), model, client, name, on_batch=report
            )
    except (IndexBuildError, OpenSearchError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    verb = "rebuilt" if result.replaced_existing else "created"
    typer.echo(
        f"{verb} {result.index}: {result.documents:,} documents verified by _count "
        f"in {time.monotonic() - started:,.1f}s -- 0 embeddings computed"
    )
