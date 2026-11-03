"""Reader clients shared by inference and RL; the historical import path is retained."""

from __future__ import annotations
from urllib.parse import unquote, urlparse
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
import sys
import time
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _trace_timing_enabled() -> bool:
    return os.environ.get("SYNTHESIS_TRACE_TIMING", "0") != "0"


def _trace_timing(message: str) -> None:
    if _trace_timing_enabled():
        print(f"[trace]{message}", file=sys.stderr, flush=True)


def _reader_retry_debug(*, target: str, sleep_seconds: float, error: object) -> None:
    """Always expose the Enhanced Reader's built-in retry."""

    print(
        "[tool-retry]"
        f" tool='enhanced_reader' retry_attempt=1 max_attempts=2"
        f" sleep_seconds={float(sleep_seconds):.3f}"
        f" error={str(error)[:1000]!r} url={target!r}",
        file=sys.stderr,
        flush=True,
    )


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(slots=True)
class ReaderDocument:
    """Content returned by a webpage reader."""

    url: str
    title: str | None = None
    content: str = ""
    raw_markdown: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class ReaderClient(Protocol):
    """Minimal webpage reader interface."""

    def read(self, url: str, **kwargs: Any) -> ReaderDocument:
        """Read a webpage into main-content text or markdown."""


class EnhancedReaderClient:
    """HTTP client for the local enhanced reader service."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8004",
        timeout_s: float = 180.0,
        retry_502: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retry_502 = bool(retry_502)

    def read(self, url: str, **kwargs: Any) -> ReaderDocument:
        del kwargs
        target = url if url.startswith(("http://", "https://")) else f"https://{url}"
        started_at = time.perf_counter()
        _trace_timing(f"[reader] phase=start target={target!r} base_url={self.base_url!r}")
        request = Request(
            f"{self.base_url}/{target}",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code != 502 or not self.retry_502:
                raise
            _reader_retry_debug(target=target, sleep_seconds=15, error=exc)
            _trace_timing(f"[reader] phase=retry_502 target={target!r} sleep_s=15")
            time.sleep(15)
            request = Request(
                f"{self.base_url}/{target}",
                headers={"Accept": "application/json"},
            )
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read().decode("utf-8")
        elapsed_s = time.perf_counter() - started_at
        _trace_timing(f"[reader] phase=done target={target!r} elapsed_s={elapsed_s:.3f}")
        data = json.loads(payload)
        document = data.get("data") or {}
        return ReaderDocument(
            url=document.get("url") or target,
            title=document.get("title") or None,
            content=document.get("content") or "",
            raw_markdown=document.get("raw_markdown") or None,
            raw=data,
        )


class RawMarkdownReaderClient:
    """Minimal client for the raw 8003 markdown reader endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8003",
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def read(self, url: str, **kwargs: Any) -> ReaderDocument:
        del kwargs
        target = url if url.startswith(("http://", "https://")) else f"https://{url}"
        request = Request(
            f"{self.base_url}/{target}",
            headers={"Accept": "text/plain"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            markdown = response.read().decode("utf-8")
        return ReaderDocument(
            url=target,
            title=_title_from_url(target),
            content=markdown,
            raw_markdown=markdown,
            raw={"reader": "raw_markdown_reader"},
        )


def _title_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.path.startswith("/wiki/"):
        return None
    return unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
