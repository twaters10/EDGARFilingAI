"""Enforces the Stage 0 acceptance criterion: no secret, contact address, or bucket
name appears in a tracked file.

This repository is PUBLIC, so anything committed is world-readable immediately.

Scope note, so this test is not mistaken for stronger protection than it gives: it
inspects the *contents of tracked files*. It does not and cannot scrub commit
metadata — the author email in git history is separate, normal, and not a leak.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files whose whole purpose is to discuss these patterns.
EXEMPT = {"tests/test_no_secrets.py"}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("AWS secret access key", re.compile(r"aws_secret_access_key\s*=\s*\S+", re.IGNORECASE)),
    ("Anthropic/OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("hardcoded s3:// bucket", re.compile(r"s3://[a-z0-9][a-z0-9.-]{2,}")),
    # The operator's real contact belongs in .env only.
    ("personal contact address", re.compile(r"waterstaylor80@hotmail\.com")),
)


def committable_files() -> list[str]:
    """Every file that ``git add -A`` would stage: tracked plus untracked-not-ignored.

    Checking only ``git ls-files`` would make this suite pass trivially before the first
    commit — exactly when a leak is easiest to introduce and cheapest to prevent.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted({line for line in result.stdout.splitlines() if line})


def test_git_is_available_and_repo_has_files() -> None:
    assert committable_files(), "expected a git repository with committable files"


def test_dotenv_is_not_committable() -> None:
    assert ".env" not in committable_files(), ".env must never be committed"


def test_dotenv_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, ".gitignore must exclude .env"


def test_dotenv_example_is_present_for_onboarding() -> None:
    assert (REPO_ROOT / ".env.example").is_file()


def test_data_directory_is_ignored() -> None:
    """Cached SEC responses are large and regenerable; they must not be committed."""
    result = subprocess.run(
        ["git", "check-ignore", "data/raw/submissions/CIK0001601712.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, ".gitignore must exclude data/"


@pytest.mark.parametrize(
    "label,pattern",
    SECRET_PATTERNS,
    ids=[label for label, _ in SECRET_PATTERNS],
)
def test_no_tracked_file_contains_secrets(label: str, pattern: re.Pattern[str]) -> None:
    offenders: list[str] = []
    for relative in committable_files():
        if relative in EXEMPT:
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; nothing to scan
        if pattern.search(text):
            offenders.append(relative)

    assert not offenders, f"{label} found in committable files: {offenders}"


def test_example_env_contains_only_the_placeholder() -> None:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "your.email@example.com" in text
    assert "hotmail" not in text.lower()
