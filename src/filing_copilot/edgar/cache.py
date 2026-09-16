"""On-disk response cache, keyed by URL.

The cache path is *derived from the URL path* rather than hashed, so the tree stays
readable::

    data/raw/
    ├── reference/company_tickers.json
    ├── submissions/CIK0001601712.json
    ├── companyfacts/CIK0001601712.json
    └── filings/1601712/000160171225000012/index.json

You will be reading these files constantly in Stages 1-2. A hashed-filename cache is
undebuggable at exactly the moment you need to debug it.

Every entry gets a ``<name>.meta.json`` sidecar recording the source URL, fetch time,
HTTP status, content type, ETag, and a SHA-256 of the body — so a cached file can
always be traced back to its origin, which is what makes citations verifiable later.

Entries never expire. The corpus is deliberately frozen; use ``refresh=True`` to
re-fetch (ADR-0017).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

_META_SUFFIX = ".meta.json"

# Ordered (pattern, replacement) rules mapping a URL path to a cache-relative path.
# First match wins; anything unmatched falls back to misc/<host>/<path>.
_PATH_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^/files/(?P<name>company_tickers\.json)$"), r"reference/\g<name>"),
    (re.compile(r"^/submissions/(?P<name>CIK\d{10}\.json)$"), r"submissions/\g<name>"),
    (
        re.compile(r"^/api/xbrl/companyfacts/(?P<name>CIK\d{10}\.json)$"),
        r"companyfacts/\g<name>",
    ),
    (
        re.compile(r"^/Archives/edgar/data/(?P<cik>\d+)/(?P<rest>.+)$"),
        r"filings/\g<cik>/\g<rest>",
    ),
    (re.compile(r"^/Archives/edgar/daily-index/(?P<rest>.+)$"), r"bulk/\g<rest>"),
)

_UNSAFE = re.compile(r"[^A-Za-z0-9._/-]")


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Summary of what is on disk."""

    entries: int
    total_bytes: int
    oldest: datetime | None
    newest: datetime | None


class ResponseCache:
    """Content-addressed-by-URL store for fetched SEC responses."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, url: str) -> Path:
        """Deterministic cache path for a URL. Pure — touches no disk."""
        parsed = urlparse(url)
        path = parsed.path
        for pattern, replacement in _PATH_RULES:
            match = pattern.match(path)
            if match is not None:
                return self._root / _sanitize(match.expand(replacement))
        host = parsed.netloc or "unknown-host"
        return self._root / "misc" / _sanitize(host) / _sanitize(path.lstrip("/"))

    def get(self, url: str) -> bytes | None:
        """Return the cached body, or ``None`` on a miss."""
        path = self.path_for(url)
        if not path.is_file():
            return None
        return path.read_bytes()

    def put(
        self,
        url: str,
        body: bytes,
        *,
        status: int,
        content_type: str | None = None,
        etag: str | None = None,
    ) -> Path:
        """Write the body and its sidecar. Returns the body path."""
        path = self.path_for(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

        meta = {
            "url": url,
            "fetched_at": datetime.now(UTC).isoformat(),
            "status": status,
            "content_type": content_type,
            "etag": etag,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        self._meta_path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def meta_for(self, url: str) -> dict[str, object] | None:
        """Read the sidecar for a cached URL, if present."""
        meta_path = self._meta_path(self.path_for(url))
        if not meta_path.is_file():
            return None
        loaded: dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
        return loaded

    def stats(self) -> CacheStats:
        """Walk the cache and summarize it."""
        entries = 0
        total = 0
        times: list[datetime] = []
        if not self._root.exists():
            return CacheStats(0, 0, None, None)

        for item in self._root.rglob("*"):
            if not item.is_file() or item.name.endswith(_META_SUFFIX):
                continue
            entries += 1
            total += item.stat().st_size
            meta_path = self._meta_path(item)
            if meta_path.is_file():
                raw = json.loads(meta_path.read_text(encoding="utf-8")).get("fetched_at")
                if isinstance(raw, str):
                    times.append(datetime.fromisoformat(raw))

        return CacheStats(
            entries=entries,
            total_bytes=total,
            oldest=min(times) if times else None,
            newest=max(times) if times else None,
        )

    @staticmethod
    def _meta_path(body_path: Path) -> Path:
        return body_path.with_name(body_path.name + _META_SUFFIX)


def _sanitize(segment: str) -> str:
    """Strip path traversal and characters that have no business in a filename."""
    cleaned = _UNSAFE.sub("_", segment)
    parts = [p for p in cleaned.split("/") if p not in ("", ".", "..")]
    return "/".join(parts)
