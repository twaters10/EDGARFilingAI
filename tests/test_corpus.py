"""Corpus loading and its validation rules.

No filesystem and no network: every case but the shipped-file test parses YAML
from a string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from filing_copilot.structured import Corpus, CorpusError

VALID = """
companies:
  - ticker: SYF
    cik: "0001601712"
    name: Synchrony Financial
    group: card_issuer
    lender: true
  - ticker: V
    cik: "0001403161"
    name: Visa Inc.
    group: card_network
    lender: false
"""


def test_parses_a_valid_corpus() -> None:
    corpus = Corpus.from_yaml(VALID)
    assert len(corpus) == 2
    assert [c.ticker for c in corpus] == ["SYF", "V"]
    assert corpus.ciks == ("0001601712", "0001403161")


def test_lenders_excludes_non_lenders() -> None:
    corpus = Corpus.from_yaml(VALID)
    assert [c.ticker for c in corpus.lenders] == ["SYF"]


def test_lookup_is_case_insensitive() -> None:
    corpus = Corpus.from_yaml(VALID)
    assert corpus.by_ticker("syf").name == "Synchrony Financial"
    assert corpus.by_ticker("  Syf  ").ticker == "SYF"


def test_cik_lookup_accepts_every_sec_rendering() -> None:
    corpus = Corpus.from_yaml(VALID)
    for rendering in ("0001601712", "1601712", 1601712, "CIK0001601712"):
        assert corpus.by_cik(rendering).ticker == "SYF"


def test_unpadded_cik_in_yaml_is_normalized() -> None:
    """A CIK written without leading zeros is plausible-looking and fatal later."""
    corpus = Corpus.from_yaml("""
        companies:
          - {ticker: AXP, cik: 4962, name: American Express, group: g, lender: true}
        """)
    assert corpus.by_ticker("AXP").cik == "0000004962"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("companies: []", "non-empty list"),
        ("{}", "top-level 'companies:' list"),
        ("companies:\n  - [not, a, mapping]", "not a mapping"),
        (
            "companies:\n  - {ticker: X, cik: '1', name: n, group: g}",
            # match= is a regex, so "key(s)" would parse as a group.
            r"missing required key\(s\): lender",
        ),
        (
            "companies:\n  - {ticker: X, cik: '1', name: n, group: g, lender: yes_please}",
            "must be true or false",
        ),
        (
            "companies:\n  - {ticker: X, cik: 'not-a-cik', name: n, group: g, lender: true}",
            "is not a CIK",
        ),
    ],
)
def test_malformed_corpus_is_rejected(raw: str, expected: str) -> None:
    with pytest.raises(CorpusError, match=expected):
        Corpus.from_yaml(raw)


def test_duplicate_ticker_is_rejected() -> None:
    """A duplicate silently halves a peer set and the matrix still reads full."""
    with pytest.raises(CorpusError, match="Duplicate ticker"):
        Corpus.from_yaml("""
            companies:
              - {ticker: SYF, cik: '1601712', name: a, group: g, lender: true}
              - {ticker: SYF, cik: '927628', name: b, group: g, lender: true}
            """)


def test_duplicate_cik_is_rejected_across_renderings() -> None:
    """Padded and unpadded spellings of one CIK are the same company."""
    with pytest.raises(CorpusError, match="Duplicate cik"):
        Corpus.from_yaml("""
            companies:
              - {ticker: SYF,  cik: '0001601712', name: a, group: g, lender: true}
              - {ticker: SYF2, cik: 1601712,      name: b, group: g, lender: true}
            """)


def test_unknown_ticker_lists_what_is_available() -> None:
    corpus = Corpus.from_yaml(VALID)
    with pytest.raises(CorpusError, match="SYF, V"):
        corpus.by_ticker("NOPE")


def test_missing_file_names_the_path_and_the_fix() -> None:
    with pytest.raises(CorpusError, match="CORPUS_PATH"):
        Corpus.load(Path("does/not/exist.yaml"))


def test_shipped_corpus_file_is_valid() -> None:
    """The real config/corpus.yaml must load -- it is the ingestion input."""
    corpus = Corpus.load(Path("config/corpus.yaml"))
    assert len(corpus) == 20
    assert len(corpus.lenders) == 17
    # Non-December fiscal year ends are required for the alignment warning to be
    # demonstrable at all. Their FYEs are derived from filed data, not asserted here.
    for ticker in ("V", "JEF", "AAPL"):
        assert corpus.by_ticker(ticker)
