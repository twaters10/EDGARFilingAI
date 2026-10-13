"""``fc`` — command line entry point for filing-copilot-mcp."""

from __future__ import annotations

import datetime as dt
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Annotated, NoReturn

import typer
import yaml

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
from .filings.chunkers import ITEM_TITLES
from .filings.download import MAX_HISTORY_PAGES, CompanyDownload, download_company
from .filings.index import IncompleteFilingIndexError
from .index import (
    IndexBuildError,
    ManifestRow,
    OpenSearchClient,
    OpenSearchError,
    build_index,
    index_name,
    manifest_path,
    read_manifest,
    write_manifest,
)
from .retrieval import (
    MODES,
    FanOutResult,
    Hit,
    Mode,
    SearchFilters,
    bm25_search,
    cosine_from_knn_score,
    encode_query,
    fan_out_peers,
    hybrid_search,
    vector_search,
)
from .retrieval.adjudicate import (
    MIN_RECHECK_DAYS,
    agreement,
    build_pool,
    narrow,
    recent_judgments,
    recheck_sample,
    render,
)
from .retrieval.evalset import (
    FAILURES_FILE,
    JUDGMENTS_FILE,
    QUESTIONS_FILE,
    RECHECKS_FILE,
    CorpusVocabulary,
    EvalSetError,
    FailureTag,
    Judgment,
    Question,
    Recheck,
    append_judgment,
    append_recheck,
    check_gold,
    load_failures,
    load_judgments,
    load_questions,
    load_rechecks,
)
from .retrieval.evaluate import (
    Config,
    QuestionResult,
    SearchFn,
    Strategy,
    failure_table,
    grid,
    live_search_fn,
    run_eval,
    summarize,
    untagged_failures,
    win_loss,
    write_csv,
)
from .retrieval.labeling import find_passages, gold_entry
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


# --- Stage 4: retrieval ---------------------------------------------------------

MultiTickerOpt = Annotated[
    list[str] | None, typer.Option("--ticker", "-t", help="Filter to a ticker. Repeatable.")
]
MultiYearOpt = Annotated[
    list[int] | None, typer.Option("--fy", help="Filter to a fiscal year. Repeatable.")
]
MultiItemOpt = Annotated[
    list[str] | None, typer.Option("--item", "-i", help="Filter to an item, e.g. 1A. Repeatable.")
]


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="The question, as a user would type it.")],
    mode: Annotated[
        str,
        typer.Option("--mode", help="vector (kNN), bm25 (lexical), or hybrid (the two, by RRF)."),
    ] = "vector",
    chunker: ChunkerOpt = "item_aware",
    k: Annotated[int, typer.Option("-k", help="Results to show per retriever.")] = 10,
    compare: Annotated[
        bool, typer.Option("--compare", help="Run all three retrievers and show them together.")
    ] = False,
    tickers: MultiTickerOpt = None,
    fiscal_years: MultiYearOpt = None,
    items: MultiItemOpt = None,
    fan_out: Annotated[
        bool,
        typer.Option("--fan-out", help="One search per --ticker, interleaved, instead of one."),
    ] = False,
    per_company: Annotated[int, typer.Option("-n", help="Hits per company with --fan-out.")] = 3,
) -> None:
    """Search one index, and show exactly what each retriever returned.

    Vector scores are shown as cosine similarity (undoing OpenSearch's score
    transform); BM25 scores are raw; hybrid scores are RRF sums, and each hybrid
    hit shows where the vector and BM25 lists ranked it. No score is comparable
    across retrievers -- only ranks are, which is why fusion uses them.

    --ticker/--fy/--item are exact-match filters applied inside each query.
    --fan-out runs one filtered search per ticker so every company is represented.
    """
    settings = _load_settings()
    _check_chunker(chunker)
    _check_mode(mode)
    filters = SearchFilters.of(
        tickers=tickers or (), fiscal_years=fiscal_years or (), items=items or ()
    )
    if fan_out and len(filters.tickers) < 2:
        _fail("--fan-out needs at least two --ticker values.", code=2)
    name = index_name(settings.opensearch_index_prefix, chunker)
    modes: tuple[str, ...] = MODES if compare and not fan_out else (mode,)

    results: dict[str, list[Hit]] = {}
    fanned: FanOutResult | None = None
    try:
        with OpenSearchClient(settings.opensearch_url) as client:
            vector: list[float] = []
            encoder = None
            if set(modes) & {"vector", "hybrid"}:
                encoder = OllamaEncoder(host=settings.ollama_host, model=_embedding_model(settings))
                vector = encode_query(encoder, query)  # once, shared by every mode
            if fan_out:
                fanned = fan_out_peers(
                    client,
                    encoder,
                    name,
                    query,
                    filters.tickers,
                    n_per_company=per_company,
                    mode=_as_mode(mode),
                    filters=filters,
                )
                results[mode] = fanned.hits
            else:
                if "vector" in modes:
                    results["vector"] = vector_search(client, name, vector, k=k, filters=filters)
                if "bm25" in modes:
                    results["bm25"] = bm25_search(client, name, query, k=k, filters=filters)
                if "hybrid" in modes:
                    results["hybrid"] = hybrid_search(
                        client, name, query, vector, k=k, filters=filters
                    )
            if encoder is not None:
                encoder.close()
    except (EncoderError, OpenSearchError) as exc:
        _fail(f"{exc}\nIs the index built? `fc build-index --chunker {chunker}`.", code=1)

    scope = "" if filters.is_empty else f"  filters: {filters.clauses()}"
    typer.echo(f"{query!r} against {name}{scope}")
    for retriever, hits in results.items():
        _print_hits(retriever, hits)

    if fanned is not None:
        covered = [t for t in filters.tickers if fanned.per_company[t]]
        typer.echo(f"\ncompanies with hits: {', '.join(covered) or '-'}")
        if fanned.empty:
            typer.echo(f"no hits (a finding, not an error): {', '.join(fanned.empty)}")
    elif compare:
        ids = {r: {h.chunk_id for h in hits} for r, hits in results.items()}
        typer.echo(f"\nshared top-{k}:  vector & bm25 {len(ids['vector'] & ids['bm25'])}")
        for retriever, hits in results.items():
            found = ", ".join(sorted({h.ticker for h in hits})) or "-"
            typer.echo(f"{retriever:>6} companies: {found}")


def _print_hits(retriever: str, hits: list[Hit], snippet_chars: int = 100) -> None:
    label = {"vector": "cosine", "bm25": "bm25", "hybrid": "rrf"}[retriever]
    typer.echo(f"\n-- {retriever} ({len(hits)} hits) " + "-" * 50)
    if not hits:
        typer.echo("   (no hits)")
    for hit in hits:
        score = cosine_from_knn_score(hit.score) if retriever == "vector" else hit.score
        hit_items = ",".join(hit.items) or "-"
        snippet = " ".join(hit.text.split())[:snippet_chars]
        where = f"{_provenance(hit):<13} " if hit.sources else ""
        typer.echo(
            f"{hit.rank:>3}  {label} {score:6.3f}  {where}{hit.ticker:<5} FY{hit.fiscal_year} "
            f"{hit_items:<8}  {snippet}"
        )


def _provenance(hit: Hit) -> str:
    """``v#3 b#1`` -- where each fused list ranked this hit; ``-`` where it did not."""
    ranks = dict(hit.sources)
    return " ".join(
        f"{name[0]}#{ranks[name]}" if name in ranks else f"{name[0]}#-"
        for name in ("vector", "bm25")
    )


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        _fail(f"--mode must be one of {', '.join(MODES)}, got {mode!r}.", code=2)


def _as_mode(mode: str) -> Mode:
    for known in MODES:
        if known == mode:
            return known
    raise ValueError(mode)


def _fail(message: str, *, code: int) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


# --- Stage 4: the retrieval eval --------------------------------------------------


@app.command("label-span")
def label_span(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Corpus ticker, e.g. SYF.")],
    find: Annotated[str, typer.Option("--find", "-f", help="Phrase to look for (any case).")],
    fy: Annotated[
        int | None, typer.Option("--fy", help="Fiscal year. Defaults to the most recent.")
    ] = None,
    item: Annotated[
        str | None, typer.Option("--item", "-i", help="Search only this item's text.")
    ] = None,
    grade: Annotated[int, typer.Option("--grade", help="Grade to pre-fill (1 or 2).")] = 2,
    limit: Annotated[int, typer.Option("--max", help="Most passages to show.")] = 10,
) -> None:
    """Find passages to label as gold, by reading the filing -- not the retriever.

    Prints every paragraph containing --find, with the exact offsets gold spans
    need, followed by a ready-to-paste YAML entry for a question's `gold:` list.
    Check each passage against RUBRIC.md before pasting, and fix the grade.
    """
    if grade not in (1, 2):
        _fail("--grade must be 1 or 2 (a 0 is a judgment, not gold).", code=2)
    settings = _load_settings()
    filing = _load_filing(settings, ticker, fy)
    passages = find_passages(filing, find, item=item, max_results=limit)

    where = f"Item {item.upper()}" if item else "the whole filing"
    typer.echo(
        f"{filing.ticker} FY{filing.ref.fiscal_year} · {filing.ref.accession} · "
        f"{len(passages)} passage(s) containing {find!r} in {where}"
    )
    if not passages:
        if item and item.strip().upper() not in filing.sections:
            typer.echo(f"Item {item.upper()} was not located in this filing -- see `fc sections`.")
        return

    for n, passage in enumerate(passages, start=1):
        typer.echo(
            f"\n[{n}] chars {passage.char_start:,}-{passage.char_end:,} "
            f"({passage.char_end - passage.char_start:,} long) · {passage.document}"
        )
        typer.echo(passage.text.strip())
        entry = yaml.safe_dump(
            [gold_entry(filing, passage, grade=grade)], sort_keys=False, width=1000
        )
        typer.secho(
            "\n".join("      " + line for line in entry.rstrip().splitlines()), fg=typer.colors.CYAN
        )


@app.command("eval-retrieval")
def eval_retrieval(
    split: Annotated[str, typer.Option("--split", help="dev (default), blind, or all.")] = "dev",
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy", help="global (question text only), scoped (scope as filters), or both."
        ),
    ] = "global",
    chunker: Annotated[
        str | None, typer.Option("--chunker", help="One chunker only. Default: both.")
    ] = None,
) -> None:
    """Score every question under every retriever x chunker, against the labels.

    Prints means per configuration (over labelled questions), per-question
    win/loss counts, the unjudged rate, and error-analysis to-dos; writes one
    CSV row per question x configuration under data/processed/eval/.
    """
    settings = _load_settings()
    if split not in ("dev", "blind", "all"):
        _fail("--split must be dev, blind or all.", code=2)
    if strategy not in ("global", "scoped", "both"):
        _fail("--strategy must be global, scoped or both.", code=2)
    chunkers = [chunker] if chunker else sorted(CHUNKERS)
    for name in chunkers:
        _check_chunker(name)

    questions, judgments, _ = _load_eval(settings)
    chosen = [q for q in questions if split == "all" or q.split == split]
    if split in ("blind", "all"):
        _record_blind_run(settings, split)

    strategies: list[Strategy] = (
        ["global", "scoped"]
        if strategy == "both"
        else ["scoped" if strategy == "scoped" else "global"]
    )
    configs = [c for s in strategies for c in grid(chunkers, strategy=s)]
    typer.echo(
        f"{len(chosen)} {split} question(s), {sum(q.is_labelled for q in chosen)} labelled · "
        f"{len(judgments)} judgment(s) · {len(configs)} configuration(s)"
    )

    results = _run_eval_live(settings, chosen, configs, judgments, chunkers)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = settings.eval_runs_dir / f"retrieval-{split}-{strategy}-{stamp}.csv"
    write_csv(results, out)

    _print_eval_summary(results, configs, chunkers)
    tags = load_failures(settings.retrieval_evals_dir / FAILURES_FILE)
    _print_failure_analysis(results, tags, configs)
    agreed = agreement(
        questions, judgments, load_rechecks(settings.retrieval_evals_dir / RECHECKS_FILE)
    )
    if agreed is not None:
        typer.echo(
            f"\nlabel agreement (recheck): {agreed.pairs} pairs · "
            f"{agreed.percent:.0%} agree · Cohen's kappa {agreed.kappa:.2f}"
        )
    typer.echo(f"\nper-question results: {out}")


@app.command("adjudicate")
def adjudicate(
    question_id: Annotated[
        str | None, typer.Option("--question", "-q", help="One question only.")
    ] = None,
    split: Annotated[str, typer.Option("--split", help="dev (default) or blind.")] = "dev",
    depth: Annotated[int, typer.Option("--depth", help="Top-k pooled per configuration.")] = 10,
    recheck: Annotated[
        bool, typer.Option("--recheck", help="Blindly re-grade a sample of judged spans.")
    ] = False,
    fraction: Annotated[
        float, typer.Option("--fraction", help="Share of judged spans to recheck.")
    ] = 0.15,
) -> None:
    """Judge pooled, unjudged hits -- blind to which system found them.

    Pools every configuration's top --depth for each labelled question, drops
    anything already judged, deduplicates, shuffles, and asks for a grade per
    RUBRIC.md. Answers are appended to judgments.yaml as you go; quit any time
    and re-run to resume.

    With --recheck: re-grade a random sample of already-judged spans without
    seeing the original grades, and report agreement (Cohen's kappa).
    """
    settings = _load_settings()
    questions, judgments, _ = _load_eval(settings)
    if recheck:
        _run_recheck(settings, questions, judgments, fraction)
        return

    chosen = [q for q in questions if (q.id == question_id if question_id else q.split == split)]
    if question_id and not chosen:
        _fail(f"No question {question_id!r} in questions.yaml.", code=2)
    unlabelled = [q.id for q in chosen if not q.is_labelled]
    if unlabelled:
        typer.secho(
            f"skipping {len(unlabelled)} question(s) with no gold yet: {', '.join(unlabelled)}\n"
            f"  Label them from the filing first (`fc label-span`) -- adjudicating a question "
            f"with no gold would build its labels from retrieval output alone.",
            fg=typer.colors.YELLOW,
        )
    chosen = [q for q in chosen if q.is_labelled]
    if not chosen:
        return

    configs = grid(sorted(CHUNKERS))
    path = settings.retrieval_evals_dir / JUDGMENTS_FILE
    judged = 0
    with _live_search(settings, sorted(CHUNKERS)) as search_fn:
        for question in chosen:
            results = {c: search_fn(question, c, depth) for c in configs}
            pool = build_pool(question, results, load_judgments(path), depth=depth)
            if not pool:
                continue
            typer.secho(f"\n=== {question.id}: {len(pool)} to judge", bold=True)
            for n, item in enumerate(pool, start=1):
                typer.echo(f"\n[{n}/{len(pool)}] " + render(item, question))
                answer = _ask_grade()
                if answer == "q":
                    typer.echo(f"\nstopped. {judged} judgment(s) saved to {path}")
                    return
                if answer == "s":
                    continue
                start, end = item.char_start, item.char_end
                if int(answer) > 0:
                    phrase = typer.prompt(
                        "  phrase from the answering paragraph (Enter = whole chunk)",
                        default="",
                        show_default=False,
                    )
                    if phrase and (narrowed := narrow(item, phrase)) is not None:
                        start, end = narrowed
                    elif phrase:
                        typer.echo("  phrase not found -- recording the whole chunk")
                append_judgment(
                    path,
                    Judgment(
                        question=question.id,
                        accession=item.accession,
                        char_start=start,
                        char_end=end,
                        grade=int(answer),
                        judged_on=dt.date.today(),
                        document=item.document,
                    ),
                )
                judged += 1
    typer.echo(f"\ndone. {judged} judgment(s) saved to {path}")


def _ask_grade() -> str:
    while True:
        answer = str(typer.prompt("  grade 0/1/2, s=skip, q=quit")).strip().lower()
        if answer in ("0", "1", "2", "s", "q"):
            return answer
        typer.echo("  0, 1, 2, s or q")


def _run_recheck(
    settings: Settings, questions: list[Question], judgments: list[Judgment], fraction: float
) -> None:
    path = settings.retrieval_evals_dir / RECHECKS_FILE
    fresh = recent_judgments(questions, judgments, dt.date.today())
    if fresh:
        typer.secho(
            f"{fresh} label(s) are under {MIN_RECHECK_DAYS} days old -- a recheck this soon "
            f"measures memory, not judgment.",
            fg=typer.colors.YELLOW,
        )
        if not typer.confirm("Recheck anyway?", default=False):
            return
    sample = recheck_sample(questions, judgments, load_rechecks(path), fraction=fraction)
    if not sample:
        typer.echo("Nothing to recheck: no judged spans, or all already rechecked.")
        return

    by_id = {q.id: q for q in questions}
    texts = _span_texts(settings, {item.accession for item in sample})
    typer.echo(f"{len(sample)} span(s) to re-grade. Original grades are not shown.")
    for n, item in enumerate(sample, start=1):
        text = texts[item.accession][item.char_start : item.char_end]
        typer.echo(
            f"\n[{n}/{len(sample)}] {item.question}: {by_id[item.question].question}\n"
            f"{'-' * 72}\n{text.strip()}\n{'-' * 72}"
        )
        answer = _ask_grade()
        if answer == "q":
            break
        if answer == "s":
            continue
        append_recheck(
            path,
            Recheck(
                question=item.question,
                accession=item.accession,
                char_start=item.char_start,
                char_end=item.char_end,
                regrade=int(answer),
                rechecked_on=dt.date.today(),
            ),
        )
    agreed = agreement(questions, judgments, load_rechecks(path))
    if agreed is not None:
        typer.echo(
            f"\nagreement over {agreed.pairs} rechecked span(s): {agreed.percent:.0%} · "
            f"Cohen's kappa {agreed.kappa:.2f}"
        )


def _load_eval(settings: Settings) -> tuple[list[Question], list[Judgment], list[ManifestRow]]:
    """Questions (validated), judgments, and the manifest the gold is checked against."""
    model = _embedding_model(settings)
    path = manifest_path(settings.manifests_dir, "item_aware", model.cache_key)
    if not path.exists():
        _fail(f"No manifest at {path}. Run 'fc embed' first.", code=2)
    manifest = read_manifest(path)
    try:
        corpus = Corpus.load(settings.corpus_path)
        vocabulary = CorpusVocabulary(
            tickers=frozenset(c.ticker for c in corpus),
            fiscal_years=frozenset(r.fiscal_year for r in manifest),
            items=frozenset(ITEM_TITLES),
        )
        questions = load_questions(settings.retrieval_evals_dir / QUESTIONS_FILE, vocabulary)
        judgments = load_judgments(settings.retrieval_evals_dir / JUDGMENTS_FILE)
    except (CorpusError, EvalSetError) as exc:
        _fail(str(exc), code=2)
    if problems := check_gold(questions, manifest):
        _fail("Gold spans out of step with the index:\n  - " + "\n  - ".join(problems), code=1)
    return questions, judgments, manifest


@contextmanager
def _live_search(settings: Settings, chunkers: list[str]) -> Iterator[SearchFn]:
    """OpenSearch and Ollama opened once, as a SearchFn over every chunker's index."""
    index_for = {c: index_name(settings.opensearch_index_prefix, c) for c in chunkers}
    with (
        OpenSearchClient(settings.opensearch_url) as client,
        OllamaEncoder(host=settings.ollama_host, model=_embedding_model(settings)) as encoder,
    ):
        yield live_search_fn(client, encoder, index_for)


def _run_eval_live(
    settings: Settings,
    questions: list[Question],
    configs: list[Config],
    judgments: list[Judgment],
    chunkers: list[str],
) -> list[QuestionResult]:
    try:
        with _live_search(settings, chunkers) as search_fn:
            return run_eval(questions, configs, search_fn, judgments)
    except (EncoderError, OpenSearchError) as exc:
        _fail(f"{exc}\nIs OpenSearch up and are the indexes built?", code=1)


def _print_eval_summary(
    results: list[QuestionResult], configs: list[Config], chunkers: list[str]
) -> None:
    def fmt(value: float | None) -> str:
        return "   -  " if value is None else f"{value:6.3f}"

    typer.echo(
        f"\n{'configuration':<30} {'labelled':>8} {'R@5':>6} {'R@10':>6} {'MRR':>6} "
        f"{'nDCG@10':>7} {'unjudged@10':>11} {'peer cov@10':>11}"
    )
    for s in summarize(results):
        typer.echo(
            f"{s.config.label:<30} {s.labelled:>4}/{s.questions:<3} {fmt(s.recall_5)} "
            f"{fmt(s.recall_10)} {fmt(s.mrr)} {fmt(s.ndcg_10):>7} {s.unjudged_10:>11.1f} "
            f"{fmt(s.peer_coverage_10):>11}"
        )
    if not any(r.labelled for r in results):
        typer.secho(
            "\nNo labelled questions yet: every metric is '-'. Add gold spans with "
            "`fc label-span`. (Peer coverage needs no labels.)",
            fg=typer.colors.YELLOW,
        )
        return

    typer.echo("\nper-question nDCG@10, wins-losses-ties:")
    for strategy in sorted({c.strategy for c in configs}):
        for name in chunkers:
            hybrid = Config("hybrid", name, strategy)
            for other in ("vector", "bm25"):
                wl = win_loss(results, hybrid, Config(_as_mode(other), name, strategy))
                typer.echo(f"  {hybrid.label:<30} vs {other:<7} {wl.wins}-{wl.losses}-{wl.ties}")
        if len(chunkers) == 2:
            for mode in MODES:
                a, b = Config(mode, "item_aware", strategy), Config(mode, "fixed_window", strategy)
                wl = win_loss(results, a, b)
                typer.echo(f"  {a.label:<30} vs fixed_window {wl.wins}-{wl.losses}-{wl.ties}")


def _print_failure_analysis(
    results: list[QuestionResult], tags: list[FailureTag], configs: list[Config]
) -> None:
    table = failure_table(tags)
    if table:
        columns = sorted({c for counts in table.values() for c in counts})
        typer.echo("\nfailure causes (failures.yaml):")
        typer.echo(f"  {'cause':<24}" + "".join(f"{c:>26}" for c in columns))
        for cause, counts in table.items():
            typer.echo(f"  {cause:<24}" + "".join(f"{counts.get(c, 0):>26}" for c in columns))
    for config in configs:
        if config.mode != "hybrid":
            continue
        todo = untagged_failures(results, tags, config)
        if todo:
            typer.echo(f"\nto tag in failures.yaml ({config.label} missed): {', '.join(todo)}")


def _record_blind_run(settings: Settings, split: str) -> None:
    """The blind split is run once. Keep a log, and say so when it is not the first time."""
    log = settings.eval_runs_dir / "blind_runs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    previous = log.read_text().splitlines() if log.exists() else []
    if previous:
        typer.secho(
            f"The blind split has been run {len(previous)} time(s) before "
            f"(last: {previous[-1]}). Its numbers stop being blind once decisions follow them.",
            fg=typer.colors.YELLOW,
        )
    with log.open("a") as handle:
        handle.write(f"{dt.datetime.now().isoformat(timespec='seconds')} split={split}\n")


def _load_filing(settings: Settings, ticker: str, fy: int | None) -> FilingText:
    try:
        company = Corpus.load(settings.corpus_path).by_ticker(ticker)
    except CorpusError as exc:
        _fail(str(exc), code=2)
    with EdgarClient(settings) as client:
        try:
            downloaded = download_company(client, company, years=_SHOW_YEARS)
        except IncompleteFilingIndexError as exc:
            _fail(str(exc), code=1)
    document = _pick_filing(downloaded.documents, company.ticker, fy)
    return build_filing_text(document, company.name)


def _span_texts(settings: Settings, accessions: set[str]) -> dict[str, str]:
    """accession -> normalized filing text, loading only the companies needed."""
    model = _embedding_model(settings)
    manifest = read_manifest(manifest_path(settings.manifests_dir, "item_aware", model.cache_key))
    ticker_of = {r.accession: r.ticker for r in manifest}
    texts: dict[str, str] = {}
    for ticker in sorted({ticker_of[a] for a in accessions}):
        company = Corpus.load(settings.corpus_path).by_ticker(ticker)
        with EdgarClient(settings) as client:
            downloaded = download_company(client, company, years=_SHOW_YEARS)
        for document in downloaded.documents:
            if document.ref.accession in accessions:
                texts[document.ref.accession] = build_filing_text(document, company.name).text
    return texts


# --- entry point ------------------------------------------------------------------
# Last in the file on purpose: `python -m filing_copilot.cli` runs this block as soon
# as it is reached, so every @app.command above must already be registered.


def main() -> None:  # pragma: no cover - thin wrapper
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
