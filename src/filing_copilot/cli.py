"""``fc`` — command line entry point for filing-copilot-mcp."""

from __future__ import annotations

import sys
from typing import Annotated

import typer

from .config import ConfigError, Settings, get_settings
from .edgar import EdgarClient, TickerResolver, UnknownTickerError, normalize_cik
from .edgar.endpoints import company_tickers_url, submissions_url
from .edgar.identifiers import Company, InvalidCIKError

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
