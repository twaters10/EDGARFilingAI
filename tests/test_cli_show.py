"""``fc show`` -- the inspection command for what the sectioner extracted.

No network: the EDGAR client is driven by the same scripted MockTransport the
rest of the suite uses, so the command walks its real path (submissions ->
filing selection -> document fetch -> normalize -> section) against bytes the
test supplies.

The assertion that matters is the round trip: what ``--offsets`` reports must
select exactly what ``--full`` prints. That is the property every Stage 5
citation rests on, checked here through the surface a person actually uses.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from filing_copilot import cli
from filing_copilot.edgar.cache import ResponseCache
from filing_copilot.edgar.client import EdgarClient
from filing_copilot.edgar.throttle import RateLimiter

CORPUS = """
companies:
  - ticker: SYF
    cik: "0001601712"
    name: Synchrony Financial
    group: card_issuer
    lender: true
"""

# A filing with two locatable items, each long enough to clear the sectioner's
# plausibility floor.
FILING = (
    "<div><p>Item 1A. Risk Factors</p>"
    + "<p>We face substantial credit risk in our card portfolio.</p>" * 40
    + "<p>Item 7. Management's Discussion and Analysis</p>"
    + "<p>Net interest income rose against the prior year.</p>" * 40
    + "</div>"
).encode()


def submissions(years: list[int]) -> bytes:
    recent = {
        "form": ["10-K"] * len(years),
        "accessionNumber": [f"0001601712-{y % 100 + 1:02d}-00000{i}" for i, y in enumerate(years)],
        "filingDate": [f"{y + 1}-02-06" for y in years],
        "reportDate": [f"{y}-12-31" for y in years],
        "primaryDocument": [f"syf-{y}1231.htm" for y in years],
        "isXBRL": [1] * len(years),
    }
    return json.dumps({"filings": {"recent": recent, "files": []}}).encode()


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CliRunner]:
    """A CLI wired to a scripted transport and a one-company corpus."""
    corpus_path = tmp_path / "corpus.yaml"
    corpus_path.write_text(CORPUS, encoding="utf-8")

    settings = cli.Settings(  # type: ignore[call-arg]
        edgar_user_agent="filing-copilot-test test@example.com",
        data_dir=tmp_path,
        corpus_path=corpus_path,
    )
    monkeypatch.setattr(cli, "_load_settings", lambda: settings)

    def handler(request: httpx.Request) -> httpx.Response:
        if "submissions" in str(request.url):
            return httpx.Response(200, content=submissions([2023, 2024, 2025]))
        return httpx.Response(200, content=FILING)

    def build(*_args: object, **_kwargs: object) -> EdgarClient:
        return EdgarClient(
            settings,
            cache=ResponseCache(settings.raw_dir),
            limiter=RateLimiter(1000.0, sleep=lambda _: None),
            transport=httpx.MockTransport(handler),
            sleep=lambda _: None,
        )

    monkeypatch.setattr(cli, "EdgarClient", build)
    yield CliRunner()


def run(runner: CliRunner, *args: str) -> tuple[int, str]:
    result = runner.invoke(cli.app, ["show", *args])
    return result.exit_code, result.stdout


def test_prints_the_section_with_its_offsets(runner: CliRunner) -> None:
    code, out = run(runner, "--ticker", "SYF", "--item", "1A")
    assert code == 0
    assert "SYF FY2025" in out
    assert "Item 1A" in out
    assert "item_heading" in out
    assert "credit risk" in out


def test_defaults_to_the_most_recent_filing(runner: CliRunner) -> None:
    _, out = run(runner, "--ticker", "SYF", "--item", "1A")
    assert "FY2025" in out


def test_a_back_year_can_be_selected(runner: CliRunner) -> None:
    code, out = run(runner, "--ticker", "SYF", "--fy", "2023", "--item", "1A")
    assert code == 0
    assert "FY2023" in out


def test_offsets_round_trip_into_the_full_text(runner: CliRunner) -> None:
    """What --offsets reports must select exactly what --full prints."""
    _, head = run(runner, "--ticker", "SYF", "--item", "1A", "--offsets")
    _, body = run(runner, "--ticker", "SYF", "--item", "1A", "--full")

    span = head.split("chars ")[1].split(" (")[0]
    start, end = (int(v.replace(",", "")) for v in span.split("-"))

    # Everything after the rule is the section verbatim, plus the single newline
    # typer.echo appends. Stripping instead would eat the section's own trailing
    # newline and quietly hide an off-by-one.
    printed = body.split("-" * 72 + "\n", 1)[1]
    assert len(printed) - 1 == end - start


def test_offsets_only_prints_no_body(runner: CliRunner) -> None:
    _, out = run(runner, "--ticker", "SYF", "--item", "1A", "--offsets")
    assert "credit risk" not in out


def test_a_long_section_is_elided_by_default(runner: CliRunner) -> None:
    _, out = run(runner, "--ticker", "SYF", "--item", "1A", "--chars", "100")
    assert "characters elided" in out


def test_full_prints_everything(runner: CliRunner) -> None:
    _, out = run(runner, "--ticker", "SYF", "--item", "1A", "--full")
    assert "characters elided" not in out


def test_the_item_is_case_insensitive(runner: CliRunner) -> None:
    assert run(runner, "--ticker", "SYF", "--item", "1a")[0] == 0


def test_an_unlocated_item_says_which_were_found(runner: CliRunner) -> None:
    """A KeyError here would be the unhelpful version of this message."""
    result = runner.invoke(cli.app, ["show", "--ticker", "SYF", "--item", "9A"])
    assert result.exit_code == 1
    assert "was not located" in result.output
    assert "1A" in result.output


def test_an_unknown_ticker_exits_two(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["show", "--ticker", "NOTREAL", "--item", "1A"])
    assert result.exit_code == 2


def test_an_unavailable_fiscal_year_lists_what_is_on_disk(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["show", "--ticker", "SYF", "--fy", "1999", "--item", "1A"])
    assert result.exit_code == 2
    assert "2025" in result.output
