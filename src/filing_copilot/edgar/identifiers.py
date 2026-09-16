"""Central Index Key (CIK) handling and ticker resolution.

SEC uses *three* renderings of the same CIK, and mixing them produces silent 404s:

===========================================  ==========================  ==============
Context                                      Form                        Example
===========================================  ==========================  ==============
``company_tickers.json`` ``cik_str`` field   integer (despite the name)  ``1601712``
``data.sec.gov`` submissions / companyfacts  10-digit padded, prefixed   ``CIK0001601712``
``sec.gov/Archives/edgar/data/``             unpadded integer            ``1601712``
===========================================  ==========================  ==============

The canonical *internal* form is a 10-digit zero-padded string with no prefix
(``"0001601712"``). Convert outward at the boundary with :func:`to_data_api_cik`
and :func:`to_archives_cik`. Never pass a bare int across a module boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

CIK_WIDTH = 10

# The separator is only permitted *after* a CIK prefix, so a bare "-1" is rejected
# rather than parsed as 1.
_CIK_PATTERN = re.compile(r"^\s*(?:CIK[-_]?)?0*(\d{1,10})\s*$", re.IGNORECASE)


class InvalidCIKError(ValueError):
    """A value could not be interpreted as a CIK."""


class UnknownTickerError(KeyError):
    """A ticker symbol is not present in SEC's company_tickers.json."""

    def __str__(self) -> str:
        # KeyError.__str__ reprs its argument, which would print the message in
        # quotes when the CLI surfaces it. Return it plainly.
        return str(self.args[0]) if self.args else ""


def normalize_cik(value: str | int) -> str:
    """Return the canonical 10-digit zero-padded CIK, without a ``CIK`` prefix.

    Accepts every form SEC emits: bare int, padded string, ``CIK``-prefixed.

    >>> normalize_cik(1601712)
    '0001601712'
    >>> normalize_cik("CIK0001601712")
    '0001601712'
    >>> normalize_cik("1601712")
    '0001601712'
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        raise InvalidCIKError(f"Not a CIK: {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise InvalidCIKError(f"CIK must be positive, got {value!r}")
        return str(value).zfill(CIK_WIDTH)

    match = _CIK_PATTERN.match(value)
    if match is None:
        raise InvalidCIKError(f"Not a CIK: {value!r}")
    digits = match.group(1)
    if int(digits) == 0:
        raise InvalidCIKError(f"CIK must be positive, got {value!r}")
    return digits.zfill(CIK_WIDTH)


def to_data_api_cik(value: str | int) -> str:
    """Render for ``data.sec.gov`` paths: ``CIK0001601712``."""
    return f"CIK{normalize_cik(value)}"


def to_archives_cik(value: str | int) -> str:
    """Render for ``sec.gov/Archives/edgar/data/`` paths: ``1601712`` (unpadded)."""
    return str(int(normalize_cik(value)))


@dataclass(frozen=True, slots=True)
class Company:
    """One row of SEC's company_tickers.json, with the CIK normalized."""

    cik: str
    ticker: str
    title: str


class TickerResolver:
    """Maps ticker symbols to CIKs using SEC's ``company_tickers.json``.

    Constructed from raw bytes rather than fetching itself, so the fetching policy
    (throttle, cache, retry) stays in :class:`~filing_copilot.edgar.client.EdgarClient`
    and this class stays trivially testable.
    """

    def __init__(self, companies: dict[str, Company]) -> None:
        self._by_ticker = companies

    @classmethod
    def from_json(cls, raw: bytes | str) -> TickerResolver:
        """Parse SEC's company_tickers.json.

        The document is an object keyed by *row index* (``"0"``, ``"1"``, ...), each
        value being ``{"cik_str": int, "ticker": str, "title": str}``.
        """
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("company_tickers.json did not contain a JSON object")

        companies: dict[str, Company] = {}
        for row in payload.values():
            ticker = str(row["ticker"]).strip().upper()
            companies[ticker] = Company(
                cik=normalize_cik(row["cik_str"]),
                ticker=ticker,
                title=str(row["title"]).strip(),
            )
        return cls(companies)

    def by_ticker(self, ticker: str) -> Company:
        """Look up a company by ticker, case-insensitively."""
        key = ticker.strip().upper()
        try:
            return self._by_ticker[key]
        except KeyError as exc:
            raise UnknownTickerError(
                f"Ticker {ticker!r} is not in SEC's company_tickers.json. "
                "Check the symbol, or pass --cik directly."
            ) from exc

    def __len__(self) -> int:
        return len(self._by_ticker)
