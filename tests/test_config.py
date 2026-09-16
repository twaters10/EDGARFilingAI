"""The User-Agent validator is what stops us introducing ourselves to SEC under a
fake identity. It is cheap to test and expensive to get wrong."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from filing_copilot.config import PLACEHOLDER_USER_AGENT, Settings


def make(**overrides: object) -> Settings:
    base: dict[str, object] = {"edgar_user_agent": "filing-copilot me@example.com"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_accepts_a_real_contact() -> None:
    assert make().edgar_user_agent == "filing-copilot me@example.com"


def test_rejects_the_example_placeholder() -> None:
    with pytest.raises(ValidationError, match="placeholder"):
        make(edgar_user_agent=PLACEHOLDER_USER_AGENT)


@pytest.mark.parametrize("value", ["", "   ", "no contact here", "missing@domain"])
def test_rejects_user_agents_without_a_contact_address(value: str) -> None:
    with pytest.raises(ValidationError):
        make(edgar_user_agent=value)


def test_user_agent_is_stripped() -> None:
    assert make(edgar_user_agent="  a@b.co  ").edgar_user_agent == "a@b.co"


def test_rate_limit_cannot_exceed_sec_ceiling() -> None:
    """SEC permits 10 req/s. Configuration must not be able to exceed it."""
    with pytest.raises(ValidationError):
        make(edgar_rate_limit=25.0)


@pytest.mark.parametrize("rate", [0, -1.0])
def test_rate_limit_must_be_positive(rate: float) -> None:
    with pytest.raises(ValidationError):
        make(edgar_rate_limit=rate)


def test_defaults_are_conservative() -> None:
    settings = make()
    assert settings.edgar_rate_limit == 5.0, "default should leave headroom under SEC's 10"
    assert settings.data_dir == Path("data")


def test_raw_dir_is_derived_from_data_dir(tmp_path: Path) -> None:
    assert make(data_dir=tmp_path).raw_dir == tmp_path / "raw"


def test_aws_settings_default_to_none_until_stage_8() -> None:
    settings = make()
    assert settings.aws_region is None
    assert settings.s3_bucket is None
