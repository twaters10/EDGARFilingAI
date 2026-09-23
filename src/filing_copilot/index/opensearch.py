"""A deliberately small OpenSearch client: five REST calls over ``httpx``.

**Tradeoff, stated:** ``opensearch-py`` would do all of this, plus SigV4 request
signing for Stage 8. It is not used here because the point of Stage 3 is to see
what building an index actually *is* -- a ``PUT`` with a mapping, a ``POST`` of
newline-delimited JSON, a refresh, a count -- and a client library hides exactly
that. It also keeps tests on ``httpx.MockTransport``, like every other HTTP test
in this repo.

The cost lands in Stage 8: Serverless requires SigV4-signed requests, which this
client will need an ``httpx.Auth`` for (botocore's ``SigV4Auth`` over the
request). That is a constructor argument, not a rewrite.

Every method raises :class:`OpenSearchError` with the server's own explanation,
because OpenSearch's error bodies are usually more precise than anything we could
paraphrase -- a mapping conflict names the field.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any

import httpx


class OpenSearchError(RuntimeError):
    """An OpenSearch request failed. Carries the server's reason."""


class OpenSearchClient:
    """Index lifecycle and bulk loading. Search arrives in Stage 4."""

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout_seconds)

    def ping(self) -> str:
        """The server's version string. Fails with a hint when nothing is listening."""
        return str(self._request("GET", "")["version"]["number"])

    def delete_index(self, name: str) -> bool:
        """Delete ``name`` if it exists. Returns whether there was one to delete."""
        response = self._send("DELETE", name)
        if response.status_code == 404:
            return False
        self._raise_for(response, "DELETE", name)
        return True

    def create_index(self, name: str, body: Mapping[str, Any]) -> None:
        self._request("PUT", name, json=body)

    def bulk_index(self, name: str, documents: Iterable[tuple[str, Mapping[str, Any]]]) -> int:
        """Index ``(id, document)`` pairs in one ``_bulk`` request. Returns the count.

        The body is NDJSON: an action line naming the ``_id``, then the document,
        per document, with a trailing newline OpenSearch insists on. Using the
        chunk's own id as ``_id`` makes re-indexing overwrite rather than
        duplicate.

        ``_bulk`` returns HTTP 200 even when individual documents fail -- the
        failures are inside the body under ``errors: true``. Checking only the
        status code would report a partial load as a success.
        """
        lines: list[str] = []
        for doc_id, document in documents:
            lines.append(json.dumps({"index": {"_index": name, "_id": doc_id}}))
            lines.append(json.dumps(document))
        if not lines:
            return 0

        result = self._request(
            "POST",
            "_bulk",
            content="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        if result.get("errors"):
            failed = [
                item["index"] for item in result["items"] if item["index"].get("error") is not None
            ]
            first = failed[0]
            raise OpenSearchError(
                f"{len(failed)} of {len(lines) // 2} documents failed to index. "
                f"First: _id={first.get('_id')} {first['error']}"
            )
        return len(lines) // 2

    def refresh(self, name: str) -> None:
        """Make everything indexed so far visible to search and ``_count``."""
        self._request("POST", f"{name}/_refresh")

    def count(self, name: str) -> int:
        return int(self._request("GET", f"{name}/_count")["count"])

    # --- transport ------------------------------------------------------------

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(method, f"{self._url}/{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise OpenSearchError(
                f"Could not reach OpenSearch at {self._url}: {exc}\n"
                f"Is it running? `colima start`, then `docker compose up -d`."
            ) from exc

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._send(method, path, **kwargs)
        self._raise_for(response, method, path)
        body: dict[str, Any] = response.json()
        return body

    @staticmethod
    def _raise_for(response: httpx.Response, method: str, path: str) -> None:
        if response.status_code < 400:
            return
        try:
            reason = response.json()["error"]
        except (json.JSONDecodeError, KeyError, TypeError):
            reason = response.text[:300]
        raise OpenSearchError(f"{method} /{path} returned {response.status_code}: {reason}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenSearchClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
