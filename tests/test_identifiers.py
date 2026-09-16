"""CIK normalization is the contract every later stage depends on.

Losing zero-padding here surfaces as a 404 in Stage 1, far from the cause — so the
three renderings get exhaustive coverage.
"""

from __future__ import annotations

import json

import pytest

from filing_copilot.edgar.identifiers import (
    Company,
    InvalidCIKError,
    TickerResolver,
    UnknownTickerError,
    normalize_cik,
    to_archives_cik,
    to_data_api_cik,
)

SYF_CIK = "0001601712"


@pytest.mark.parametrize(
    "value",
    [
        1601712,
        "1601712",
        "0001601712",
        "CIK0001601712",
        "cik0001601712",
        "  CIK0001601712  ",
    ],
)
def test_every_sec_rendering_normalizes_to_canonical(value: str | int) -> None:
    assert normalize_cik(value) == SYF_CIK


def test_canonical_form_is_ten_digits() -> None:
    assert len(normalize_cik(320193)) == 10
    assert normalize_cik(320193) == "0000320193"


def test_data_api_rendering_is_prefixed_and_padded() -> None:
    assert to_data_api_cik(1601712) == "CIK0001601712"
    assert to_data_api_cik("CIK0001601712") == "CIK0001601712"


def test_archives_rendering_is_unpadded() -> None:
    assert to_archives_cik("0001601712") == "1601712"
    assert to_archives_cik(1601712) == "1601712"


def test_renderings_round_trip() -> None:
    for start in (1601712, 320193, 1):
        canonical = normalize_cik(start)
        assert normalize_cik(to_data_api_cik(canonical)) == canonical
        assert normalize_cik(to_archives_cik(canonical)) == canonical


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "abc", "CIK", "12345678901", "-1", "0", 0, -5, True, "0000000000"],
)
def test_invalid_values_are_rejected(bad: object) -> None:
    with pytest.raises(InvalidCIKError):
        normalize_cik(bad)  # type: ignore[arg-type]


# --- TickerResolver ---------------------------------------------------------

TICKERS_FIXTURE = json.dumps(
    {
        "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "546": {"cik_str": 1601712, "ticker": "SYF", "title": "Synchrony Financial"},
    }
).encode()


def test_resolver_parses_sec_index_keyed_object() -> None:
    resolver = TickerResolver.from_json(TICKERS_FIXTURE)
    assert len(resolver) == 3


def test_resolver_normalizes_the_integer_cik_str_field() -> None:
    """cik_str is an int despite the name; it must not reach callers unpadded."""
    company = TickerResolver.from_json(TICKERS_FIXTURE).by_ticker("SYF")
    assert company == Company(cik=SYF_CIK, ticker="SYF", title="Synchrony Financial")
    assert isinstance(company.cik, str)


@pytest.mark.parametrize("symbol", ["SYF", "syf", " Syf "])
def test_resolver_is_case_and_space_insensitive(symbol: str) -> None:
    assert TickerResolver.from_json(TICKERS_FIXTURE).by_ticker(symbol).cik == SYF_CIK


def test_unknown_ticker_raises_with_actionable_message() -> None:
    resolver = TickerResolver.from_json(TICKERS_FIXTURE)
    with pytest.raises(UnknownTickerError, match="--cik"):
        resolver.by_ticker("NOTATICKER")
