"""Single source of configuration.

No credential, bucket name, or region is hardcoded anywhere else in this project.
Everything comes from the environment or a git-ignored ``.env`` file.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The literal value shipped in .env.example. If it survives into a real run we refuse
# to start, rather than introduce ourselves to SEC under a fake identity.
PLACEHOLDER_USER_AGENT = "Your Name your.email@example.com"

_CONTACT_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class ConfigError(RuntimeError):
    """Configuration is missing or unusable. Carries a message meant for a human."""


class Settings(BaseSettings):
    """Runtime settings, loaded from the environment then ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- EDGAR -------------------------------------------------------------
    edgar_user_agent: str = Field(
        description="Sent on every SEC request. SEC requires a real contact address."
    )
    edgar_rate_limit: float = Field(
        default=5.0,
        gt=0,
        le=10.0,
        description="Requests per second. SEC permits 10; we default to 5 for headroom.",
    )
    edgar_max_retries: int = Field(default=5, ge=0, le=10)
    edgar_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Storage -----------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    corpus_path: Path = Field(
        default=Path("config/corpus.yaml"),
        description="The companies to ingest. Corpus growth is a config change.",
    )
    concepts_path: Path | None = Field(
        default=None,
        description="Override the packaged concepts.yaml. Unset uses the shipped file.",
    )

    # --- AWS (unused until Stage 8; declared so nothing is ever hardcoded) --
    aws_region: str | None = None
    s3_bucket: str | None = None

    @field_validator("edgar_user_agent")
    @classmethod
    def _reject_placeholder_user_agent(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("EDGAR_USER_AGENT is empty.")
        if cleaned == PLACEHOLDER_USER_AGENT:
            raise ValueError(
                "EDGAR_USER_AGENT is still the .env.example placeholder. "
                "SEC requires a real contact address — edit .env before fetching."
            )
        if not _CONTACT_PATTERN.search(cleaned):
            raise ValueError(
                "EDGAR_USER_AGENT must contain a contact email address, "
                f"e.g. {PLACEHOLDER_USER_AGENT!r}."
            )
        return cleaned

    @property
    def raw_dir(self) -> Path:
        """Root of the on-disk response cache. Exactly as SEC served it."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Scratch space for work in progress. Safe to delete and rebuild."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Query-ready artifacts -- the Parquet fact tree lives here."""
        return self.data_dir / "processed"

    @property
    def facts_dir(self) -> Path:
        """Root of the partitioned fact table: cik=<cik>/fiscal_year=<yyyy>/."""
        return self.processed_dir / "facts"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process.

    Raises:
        ConfigError: with a message suitable for printing to a user, not a traceback.
    """
    try:
        return Settings()  # type: ignore[call-arg]  # pydantic-settings fills from env
    except Exception as exc:
        raise ConfigError(
            f"Could not load configuration.\n\n{exc}\n\n"
            "Copy .env.example to .env and fill in EDGAR_USER_AGENT."
        ) from exc
