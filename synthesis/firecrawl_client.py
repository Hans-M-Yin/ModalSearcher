"""Firecrawl scraping and Browser image downloads through the official SDK.

Configure ``FIRECRAWL_API_KEY`` to use one API key for this process. Browser
sessions, download retries, and Wikimedia image variants are managed locally.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import unquote, urlsplit

from synthesis.retry_monitor import tracked_sleep
from synthesis.url_utils import normalize_http_referer, normalize_http_url

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    fcntl = None


_MODULE_DIR = Path(__file__).resolve().parent
_FIRECRAWL_API_URL = "https://api.firecrawl.dev"
_WIKIMEDIA_UPLOAD_HOST = "upload.wikimedia.org"
_WIKIMEDIA_VARIANT_MAX_LONG_EDGE = 1000
_WIKIMEDIA_VARIANT_METADATA_TIMEOUT_S = 8.0
_WIKIMEDIA_VARIANT_CACHE_ENV = "SFT_WIKIMEDIA_CACHE_DIR"
_WIKIMEDIA_VARIANT_CACHE_SUBDIR = "browser_variants"
_WIKIMEDIA_VARIANT_MEMORY_LIMIT = 4096
_WIKIMEDIA_VARIANT_LOCK = threading.Lock()
_WIKIMEDIA_VARIANT_MEMORY_CACHE: dict[str, dict[str, Any]] = {}
_PROCESS_USAGE_LOCK = threading.Lock()
_PROCESS_USAGE: dict[str, int] = {
    "requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "credits_used": 0,
}


def _status_code_from_exception(exc: BaseException) -> int | None:
    """Extract an HTTP status from an SDK/API exception without parsing its text."""
    candidates: list[Any] = [exc, getattr(exc, "response", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("status_code", "status"):
            value = getattr(candidate, attribute, None)
            try:
                status_code = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= status_code <= 599:
                return status_code
    return None


_FIRECRAWL_CREDIT_ERROR_MARKERS = (
    "insufficient credits",
    "not enough credits",
    "no credits remaining",
    "credits exhausted",
    "credit limit exceeded",
    "quota exhausted",
    "quota exceeded",
    "credit balance",
)


def _is_firecrawl_credit_exhausted(
    *,
    status_code: int | None = None,
    response_body: str = "",
) -> bool:
    """Recognize API-level Firecrawl credit exhaustion.

    A root-level HTTP 402 is treated as a credit/billing exhaustion response.
    Other statuses require an explicit credit/quota marker so target-page
    403/404/5xx responses do not disable a shared API key.
    """

    if status_code == 402:
        return True
    if status_code is not None and status_code >= 500:
        return False
    message = str(response_body or "").lower()
    return any(marker in message for marker in _FIRECRAWL_CREDIT_ERROR_MARKERS)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _record_process_usage(*, success: bool, credits_used: int = 0) -> None:
    """Track Firecrawl usage for this Python process, independently of the key pool."""
    with _PROCESS_USAGE_LOCK:
        _PROCESS_USAGE["requests"] += 1
        _PROCESS_USAGE["successful_requests" if success else "failed_requests"] += 1
        _PROCESS_USAGE["credits_used"] += max(0, int(credits_used))


def _firecrawl_error_debug(event: str, **details: object) -> None:
    """Emit Firecrawl backend context only when a call fails."""

    suffix = " ".join(f"{key}={value!r}" for key, value in details.items())
    print(
        f"[firecrawl] {event}{(' ' + suffix) if suffix else ''}",
        file=sys.stderr,
        flush=True,
    )


def get_firecrawl_usage_snapshot() -> dict[str, int]:
    """Return a thread-safe snapshot of Firecrawl calls made by this process."""
    with _PROCESS_USAGE_LOCK:
        return dict(_PROCESS_USAGE)


def _local_browser_lock_path_for_state(state_path: Path) -> Path:
    """Return a local lock path for Browser state stored on HDFS/shared storage."""

    state_hash = hashlib.sha256(str(state_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"firecrawl_browser_state_{state_hash}.lock"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _is_wikimedia_upload_url(url: str) -> bool:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower() == _WIKIMEDIA_UPLOAD_HOST
    except (TypeError, ValueError):
        return False


def _wikimedia_upload_file_spec(url: str) -> dict[str, str] | None:
    """Extract the MediaWiki API endpoint and file title from an upload URL."""

    if not _is_wikimedia_upload_url(url):
        return None
    try:
        parts = urlsplit(url)
    except (TypeError, ValueError):
        return None
    segments = parts.path.strip("/").split("/")
    if len(segments) < 5 or segments[0].lower() != "wikipedia":
        return None
    project = segments[1].strip().lower()
    if not project:
        return None
    if segments[2].lower() == "thumb":
        if len(segments) < 7:
            return None
        filename_segment = segments[5]
    else:
        filename_segment = segments[4]
    filename = unquote(filename_segment).replace("_", " ").strip()
    if not filename:
        return None
    api_host = "commons.wikimedia.org" if project == "commons" else f"{project}.wikipedia.org"
    return {
        "api_url": f"https://{api_host}/w/api.php",
        "file_title": f"File:{filename}",
    }


def _wikimedia_variant_cache_root() -> Path:
    configured = str(os.environ.get(_WIKIMEDIA_VARIANT_CACHE_ENV) or "").strip()
    root = Path(configured).expanduser() if configured else _MODULE_DIR / ".ignore" / "wiki_caches"
    return root / _WIKIMEDIA_VARIANT_CACHE_SUBDIR


def _wikimedia_variant_cache_key(url: str, max_long_edge: int) -> str:
    return hashlib.sha256(
        f"{url}\nmax_long_edge={int(max_long_edge)}".encode("utf-8")
    ).hexdigest()


def _wikimedia_variant_cache_path(url: str, max_long_edge: int) -> Path:
    digest = _wikimedia_variant_cache_key(url, max_long_edge)
    return _wikimedia_variant_cache_root() / digest[:2] / f"{digest}.json"


def _valid_wikimedia_variant_record(
    record: object,
    *,
    requested_url: str,
    max_long_edge: int,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    if str(record.get("requested_url") or "") != requested_url:
        return None
    try:
        if int(record.get("max_long_edge") or 0) != int(max_long_edge):
            return None
    except (TypeError, ValueError):
        return None
    download_url = str(record.get("download_url") or "").strip()
    if not download_url or not download_url.startswith(("http://", "https://")):
        return None
    if not bool(record.get("resolved")):
        return None
    return dict(record)


def _read_wikimedia_variant_cache(url: str, max_long_edge: int) -> dict[str, Any] | None:
    key = _wikimedia_variant_cache_key(url, max_long_edge)
    with _WIKIMEDIA_VARIANT_LOCK:
        cached = _valid_wikimedia_variant_record(
            _WIKIMEDIA_VARIANT_MEMORY_CACHE.get(key),
            requested_url=url,
            max_long_edge=max_long_edge,
        )
        if cached is not None:
            return cached
    path = _wikimedia_variant_cache_path(url, max_long_edge)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    cached = _valid_wikimedia_variant_record(
        record,
        requested_url=url,
        max_long_edge=max_long_edge,
    )
    if cached is not None:
        with _WIKIMEDIA_VARIANT_LOCK:
            if len(_WIKIMEDIA_VARIANT_MEMORY_CACHE) >= _WIKIMEDIA_VARIANT_MEMORY_LIMIT:
                _WIKIMEDIA_VARIANT_MEMORY_CACHE.pop(next(iter(_WIKIMEDIA_VARIANT_MEMORY_CACHE)))
            _WIKIMEDIA_VARIANT_MEMORY_CACHE[key] = dict(cached)
    return cached


def _write_wikimedia_variant_cache(record: dict[str, Any]) -> None:
    requested_url = str(record.get("requested_url") or "").strip()
    try:
        max_long_edge = int(record.get("max_long_edge") or 0)
    except (TypeError, ValueError):
        return
    validated = _valid_wikimedia_variant_record(
        record,
        requested_url=requested_url,
        max_long_edge=max_long_edge,
    )
    if validated is None:
        return
    key = _wikimedia_variant_cache_key(requested_url, max_long_edge)
    with _WIKIMEDIA_VARIANT_LOCK:
        if len(_WIKIMEDIA_VARIANT_MEMORY_CACHE) >= _WIKIMEDIA_VARIANT_MEMORY_LIMIT:
            _WIKIMEDIA_VARIANT_MEMORY_CACHE.pop(next(iter(_WIKIMEDIA_VARIANT_MEMORY_CACHE)))
        _WIKIMEDIA_VARIANT_MEMORY_CACHE[key] = dict(validated)
    path = _wikimedia_variant_cache_path(requested_url, max_long_edge)
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(validated, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError:
        # The in-memory entry remains useful even when the optional durable
        # cache is temporarily unavailable.
        pass
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


class _FirecrawlKeyProvider(Protocol):
    """Credential interface used by scraping and Browser sessions."""

    def acquire_key(self) -> tuple[str, dict[str, Any]]: ...

    def record_result(self, key_id: str, **kwargs: Any) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...

    def active_key_ids(self) -> list[str]: ...

    def key_for_id(self, key_id: str) -> str: ...


def _firecrawl_fixed_client_key(
    *,
    explicit_key: str | None = None,
    legacy_api_keys: list[str] | None = None,
) -> str:
    """Return the one Firecrawl key used by a worker process.

    The environment variable is authoritative for normal workers.  The
    legacy list argument is accepted only so diagnostic callers can probe one
    key explicitly; it is never rotated or persisted by the worker.
    """

    explicit = str(explicit_key or "").strip()
    if explicit:
        return explicit
    value = str(
        os.environ.get("FIRECRAWL_API_KEY")
        or (legacy_api_keys[0] if legacy_api_keys else "")
        or ""
    ).strip()
    return value


class _FixedFirecrawlKeyProvider:
    """Worker-side fixed-key provider with no state file or lock."""

    def __init__(
        self,
        *,
        explicit_key: str | None = None,
        legacy_api_keys: list[str] | None = None,
    ) -> None:
        self.api_key = _firecrawl_fixed_client_key(
            explicit_key=explicit_key,
            legacy_api_keys=legacy_api_keys,
        )
        self.key_id_value = self.key_id(self.api_key) if self.api_key else "fixed-client"
        self.keys = [self.api_key] if self.api_key else []
        self.state_path = None
        self.lock_path = None

    @classmethod
    def key_id(cls, key: str) -> str:
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]

    def acquire_key(self) -> tuple[str, dict[str, Any]]:
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY must be set for the fixed-key Firecrawl client.")
        return self.api_key, {
            "key_id": self.key_id_value,
            "masked_key": "fixed-client",
            "managed_by": "fixed_api_key",
        }

    def record_result(self, key_id: str, **kwargs: Any) -> dict[str, Any]:
        # Account-level credit accounting stays with the provider. The client
        # keeps no credential state files.
        del key_id, kwargs
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "available_key_count": 1 if self.api_key else 0,
            "total_key_count": 1 if self.api_key else 0,
            "remaining_credits_total": -1,
            "credits_consumed": -1,
            "default_credits": -1,
            "successful_requests": 0,
            "failed_requests": 0,
            "managed_by": "fixed_api_key",
        }

    def active_key_ids(self) -> list[str]:
        return [self.key_id_value] if self.api_key else []

    def key_for_id(self, key_id: str) -> str:
        if key_id != self.key_id_value or not self.api_key:
            raise KeyError(f"Unknown fixed Firecrawl key id: {key_id}")
        return self.api_key


class FirecrawlClient:
    """Synchronous Firecrawl scraper using one fixed worker key."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        pool_state_path: str | Path | None = None,
        app_factory: Callable[..., Any] | None = None,
    ) -> None:
        # Keep historical arguments for source compatibility. Normal callers
        # use api_key or FIRECRAWL_API_KEY; a legacy list contributes one key.
        del pool_state_path
        self.key_pool: _FirecrawlKeyProvider = _FixedFirecrawlKeyProvider(
            explicit_key=api_key,
            legacy_api_keys=api_keys,
        )
        self._app_factory = app_factory
        self.last_pool_metadata: dict[str, Any] | None = None

    def scrape(
        self,
        url: str,
        *,
        only_main_content: bool = True,
        max_age: int | None = 172800000,
        parsers: list[str] | None = None,
        formats: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Scrape ``url`` and return Firecrawl's unmodified response dictionary."""
        api_key, pool_metadata = self.key_pool.acquire_key()
        self.last_pool_metadata = pool_metadata
        request_kwargs: dict[str, Any] = {"only_main_content": only_main_content}
        if max_age is not None:
            request_kwargs["max_age"] = max_age
        if parsers is not None:
            request_kwargs["parsers"] = parsers
        if formats is not None:
            request_kwargs["formats"] = formats
        request_kwargs.update(kwargs)
        try:
            result = self._app(api_key).scrape(url, **request_kwargs)
            raw = self._response_as_dict(result)
        except Exception as exc:  # SDK transport and provider exceptions are recorded identically.
            api_status_code = _status_code_from_exception(exc)
            key_credits_exhausted = _is_firecrawl_credit_exhausted(
                status_code=api_status_code,
                response_body=str(exc),
            )
            _firecrawl_error_debug(
                "scrape_failed",
                url=url,
                error_type=exc.__class__.__name__,
                status_code=api_status_code,
                error=str(exc),
            )
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=False,
                error=str(exc),
                status_code=api_status_code,
                key_auth_failed=api_status_code == 401,
                key_credits_exhausted=key_credits_exhausted,
            )
            _record_process_usage(success=False)
            raise RuntimeError(f"Firecrawl scrape failed for {url}: {exc}") from exc
        root_status_code = self._root_status_code(raw)
        if root_status_code == 401:
            error = str(raw.get("error") or "Firecrawl API authentication failed")
            _firecrawl_error_debug(
                "scrape_failed",
                url=url,
                status_code=root_status_code,
                error=error,
            )
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=False,
                error=error,
                status_code=root_status_code,
                key_auth_failed=True,
            )
            _record_process_usage(success=False)
            return {"error": error}
        response_metadata = self._response_metadata(raw)
        credits_used = self._non_negative_int(
            response_metadata.get("creditsUsed", response_metadata.get("credits_used"))
        ) or 0
        status_code = self._non_negative_int(
            response_metadata.get("statusCode", response_metadata.get("status_code"))
        )
        root_error = str(raw.get("error") or raw.get("message") or "")
        if _is_firecrawl_credit_exhausted(
            status_code=root_status_code,
            response_body=root_error,
        ):
            error = root_error or self._status_code_error(
                raw,
                response_metadata,
                root_status_code or 402,
            )
            _firecrawl_error_debug(
                "scrape_failed",
                url=url,
                status_code=root_status_code,
                credits_used=credits_used,
                error=error,
                key_credits_exhausted=True,
            )
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=False,
                error=error,
                credits_used=credits_used,
                status_code=root_status_code,
                key_credits_exhausted=True,
            )
            _record_process_usage(success=False, credits_used=credits_used)
            return {"error": error}
        if status_code is not None and status_code != 200:
            error = self._status_code_error(raw, response_metadata, status_code)
            _firecrawl_error_debug(
                "scrape_failed",
                url=url,
                status_code=status_code,
                credits_used=credits_used,
                error=error,
            )
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=False,
                error=error,
                credits_used=credits_used,
                status_code=status_code,
            )
            _record_process_usage(success=False, credits_used=credits_used)
            return {"error": error}
        if raw.get("success") is True or (
            "success" not in raw and self._is_direct_success_response(raw)
        ):
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=True,
                credits_used=credits_used,
                status_code=status_code,
            )
            _record_process_usage(success=True, credits_used=credits_used)
            return raw
        error = str(raw.get("error") or "Firecrawl returned an unsuccessful response")
        _firecrawl_error_debug(
            "scrape_failed",
            url=url,
            status_code=status_code,
            credits_used=credits_used,
            error=error,
        )
        self.key_pool.record_result(
            pool_metadata["key_id"],
            success=False,
            error=error,
            credits_used=credits_used,
            status_code=status_code,
        )
        _record_process_usage(success=False, credits_used=credits_used)
        return raw


    def _app(self, api_key: str) -> Any:
        if self._app_factory is not None:
            return self._app_factory(api_key=api_key)
        try:
            from firecrawl import Firecrawl
        except ImportError as exc:  # Keep the backend importable without the optional SDK.
            raise RuntimeError(
                "Firecrawl SDK import failed. Install the firecrawl package in the Python "
                f"environment running this process. Original ImportError: {exc}"
            ) from exc
        app = Firecrawl(api_key=api_key, api_url=_FIRECRAWL_API_URL)
        if not callable(getattr(app, "scrape", None)):
            raise RuntimeError(
                "Installed Firecrawl SDK is incompatible: Firecrawl(api_key=...) has no callable "
                ".scrape method. Upgrade the firecrawl package."
            )
        return app

    @staticmethod
    def _response_as_dict(response: Any) -> dict[str, Any]:
        normalized = FirecrawlClient._normalize_response_value(response)
        if not isinstance(normalized, dict):
            raise TypeError(f"Unexpected Firecrawl response type: {type(response).__name__}")
        return normalized

    @staticmethod
    def _normalize_response_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: FirecrawlClient._normalize_response_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FirecrawlClient._normalize_response_value(item) for item in value]
        if hasattr(value, "model_dump"):
            return FirecrawlClient._normalize_response_value(value.model_dump())
        if hasattr(value, "dict"):
            return FirecrawlClient._normalize_response_value(value.dict())
        return value

    @staticmethod
    def _response_metadata(response: dict[str, Any]) -> dict[str, Any]:
        payload = FirecrawlClient._content_payload(response)
        if isinstance(payload.get("metadata"), dict):
            return payload["metadata"]
        metadata = response.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _root_status_code(response: dict[str, Any]) -> int | None:
        """Return an API response status, excluding nested page metadata."""
        for key in ("statusCode", "status_code"):
            value = response.get(key)
            try:
                status_code = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= status_code <= 599:
                return status_code
        return None

    @staticmethod
    def _content_payload(response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data")
        return data if isinstance(data, dict) else response

    @staticmethod
    def _markdown_length(response: dict[str, Any]) -> int:
        markdown = FirecrawlClient._content_payload(response).get("markdown")
        return len(markdown) if isinstance(markdown, str) else 0

    @staticmethod
    def _is_direct_success_response(response: dict[str, Any]) -> bool:
        """Support SDK versions that return the scrape document without `success/data`."""
        return "error" not in response and isinstance(response.get("metadata"), dict)

    @staticmethod
    def _non_negative_int(value: Any) -> int | None:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _status_code_error(
        response: dict[str, Any], metadata: dict[str, Any], status_code: int,
    ) -> str:
        provider_error = metadata.get("error") or response.get("error")
        if provider_error:
            return f"Firecrawl scrape returned statusCode {status_code}: {provider_error}"
        source_url = metadata.get("sourceURL") or metadata.get("url")
        suffix = f" for {source_url}" if source_url else ""
        return f"Firecrawl scrape returned statusCode {status_code}{suffix}."


class FirecrawlBrowserError(RuntimeError):
    """Base class for failures in the managed Firecrawl Browser image backend."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FirecrawlBrowserTransportError(FirecrawlBrowserError):
    """A Browser API request failed without proving the session is broken."""


class FirecrawlBrowserTargetRequestError(FirecrawlBrowserError):
    """The remote Browser could not complete the request to the target URL."""


class FirecrawlBrowserSessionError(FirecrawlBrowserError):
    """A Browser session could not be created or is no longer usable."""


class FirecrawlBrowserHttpError(FirecrawlBrowserError):
    """The remote browser reached the URL but received a non-success status."""


class FirecrawlBrowserNonImageError(FirecrawlBrowserError):
    """The remote URL resolved successfully but did not return image bytes."""


@dataclass(slots=True)
class FirecrawlBrowserImageDownload:
    """Raw image bytes obtained through one managed Firecrawl Browser session."""

    payload: bytes
    content_type: str
    requested_url: str
    resolved_url: str
    status_code: int
    session_id: str
    key_id: str
    download_url: str = ""
    variant_source: str = "original"
    variant_resolved: bool = False
    original_width: int | None = None
    original_height: int | None = None
    download_width: int | None = None
    download_height: int | None = None
    variant_resolution_error: str = ""


@dataclass(slots=True)
class _BrowserSessionLease:
    manager: "FirecrawlBrowserSessionManager"
    slot_id: str
    session_id: str
    key_id: str
    lease_id: str
    request_url: str = ""
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.manager.release(self)
            self.released = True

    def invalidate(self, reason: str) -> None:
        if not self.released:
            self.manager.invalidate(self, reason=reason)
            self.released = True

    def __enter__(self) -> "_BrowserSessionLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class FirecrawlBrowserSessionManager:
    """File-backed, lease-based pool of reusable Firecrawl Browser sessions.

    It owns Browser session creation, selection, expiry, and best-effort
    cleanup. Sessions use the same configured API key; the provider manages
    account-level authentication and billing.
    The JSON registry stores opaque key ids rather than API secrets.

    The registry may live on HDFS/shared storage, but its default lock lives on
    the local filesystem because HDFS does not provide the POSIX ``flock``
    semantics needed here.  The default therefore coordinates processes on
    one host.  A multi-host deployment must provide a shared POSIX lock path
    through ``FIRECRAWL_BROWSER_SESSION_LOCK_FILE`` or put this manager behind
    one gateway process.
    """

    _STATE_VERSION = 2

    def __init__(
        self,
        *,
        key_pool: _FirecrawlKeyProvider,
        app_factory: Callable[..., Any] | None = None,
        state_path: str | Path | None = None,
        session_ttl_s: int = 300,
        activity_ttl_s: int = 300,
        max_sessions: int = 4,
        expiry_safety_s: float = 30.0,
        create_timeout_s: float = 90.0,
        api_timeout_s: float = 60.0,
    ) -> None:
        self.key_pool = key_pool
        self._app_factory = app_factory
        configured_path = state_path or os.environ.get("FIRECRAWL_BROWSER_SESSION_STATE_FILE")
        default_path = _MODULE_DIR / ".ignore" / "firecrawl_browser_sessions.json"
        self.state_path = Path(configured_path) if configured_path else default_path
        if not self.state_path.is_absolute():
            self.state_path = Path.cwd() / self.state_path
        configured_lock_path = os.environ.get("FIRECRAWL_BROWSER_SESSION_LOCK_FILE")
        self.lock_path = (
            Path(configured_lock_path)
            if configured_lock_path
            else _local_browser_lock_path_for_state(self.state_path)
        )
        if not self.lock_path.is_absolute():
            self.lock_path = Path.cwd() / self.lock_path
        self.session_ttl_s = max(30, min(int(session_ttl_s), 3600))
        self.activity_ttl_s = max(10, min(int(activity_ttl_s), 3600))
        self.max_sessions = max(1, int(max_sessions))
        self.expiry_safety_s = max(1.0, float(expiry_safety_s))
        self.create_timeout_s = max(10.0, float(create_timeout_s))
        self.api_timeout_s = max(1.0, float(api_timeout_s))

    @classmethod
    def from_environment(
        cls,
        *,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        app_factory: Callable[..., Any] | None = None,
        pool_state_path: str | Path | None = None,
        session_state_path: str | Path | None = None,
    ) -> "FirecrawlBrowserSessionManager":
        del pool_state_path
        key_pool = _FixedFirecrawlKeyProvider(
            explicit_key=api_key,
            legacy_api_keys=api_keys,
        )
        return cls(
            key_pool=key_pool,
            app_factory=app_factory,
            state_path=session_state_path,
            session_ttl_s=_env_int("FIRECRAWL_BROWSER_SESSION_TTL_S", 300),
            activity_ttl_s=_env_int("FIRECRAWL_BROWSER_SESSION_ACTIVITY_TTL_S", 300),
            max_sessions=_env_int("FIRECRAWL_BROWSER_MAX_SESSIONS", 4),
            expiry_safety_s=_env_float("FIRECRAWL_BROWSER_EXPIRY_SAFETY_S", 30.0),
            create_timeout_s=_env_float("FIRECRAWL_BROWSER_CREATE_TIMEOUT_S", 90.0),
            api_timeout_s=_env_float("FIRECRAWL_BROWSER_API_TIMEOUT_S", 60.0),
        )

    def acquire(
        self,
        *,
        acquire_timeout_s: float,
        lease_timeout_s: float,
        request_url: str | None = None,
    ) -> _BrowserSessionLease:
        """Lease an active session, creating one lazily when the pool can grow."""

        acquire_timeout_s = max(1.0, float(acquire_timeout_s))
        deadline = time.monotonic() + acquire_timeout_s
        lease_timeout_s = max(5.0, float(lease_timeout_s))
        last_error: Exception | None = None
        waiter_id = uuid.uuid4().hex
        self._register_waiter(
            waiter_id,
            expires_at=time.time() + acquire_timeout_s + 5.0,
        )
        try:
            while True:
                reservation = self._reserve(lease_timeout_s=lease_timeout_s)
                kind = reservation["kind"]
                if kind == "leased":
                    lease = self._lease_from_record(reservation["record"], request_url=request_url)
                    self._remove_waiter(waiter_id)
                    self._debug_pool_status(
                        "lease_acquired",
                        session_id=lease.session_id,
                        key_id=lease.key_id,
                        url=lease.request_url,
                    )
                    return lease
                if kind == "create":
                    try:
                        lease = self._create_reserved_session(reservation["record"], request_url=request_url)
                        self._remove_waiter(waiter_id)
                        self._debug_pool_status(
                            "session_created_and_leased",
                            session_id=lease.session_id,
                            key_id=lease.key_id,
                            url=lease.request_url,
                        )
                        return lease
                    except Exception as exc:
                        last_error = exc
                        if time.monotonic() >= deadline:
                            raise FirecrawlBrowserSessionError(
                                f"Firecrawl Browser session creation failed: {exc}",
                                status_code=_status_code_from_exception(exc),
                            ) from exc
                elif kind == "no_key":
                    raise FirecrawlBrowserSessionError("No active Firecrawl API key is available for Browser sessions.")

                if time.monotonic() >= deadline:
                    suffix = f"; last_error={last_error}" if last_error is not None else ""
                    raise FirecrawlBrowserSessionError(
                        f"Timed out waiting for an idle Firecrawl Browser session{suffix}"
                    )
                tracked_sleep(
                    0.1,
                    reason="firecrawl_pool_wait",
                    tool="firecrawl_browser",
                    url=request_url,
                    error_type="FirecrawlBrowserPoolWait",
                    kind="resource_wait",
                    fallback_sleep=time.sleep,
                )
        finally:
            self._remove_waiter(waiter_id)

    def execute(
        self,
        lease: _BrowserSessionLease,
        *,
        code: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Execute code using a leased Browser session and normalize SDK output."""

        timeout_s = max(1, min(int(timeout_s), 300))
        try:
            app = self._app(lease.key_id)
            execute_kwargs: dict[str, Any] = {
                "language": "node",
                "timeout": timeout_s,
            }
            response = app.browser_execute(lease.session_id, code, **execute_kwargs)
            raw = FirecrawlClient._response_as_dict(response)
        except Exception as exc:
            self._record_key_result(lease.key_id, success=False, error=str(exc), exc=exc)
            status_code = _status_code_from_exception(exc)
            message = f"Firecrawl Browser execute failed: {exc}"
            if _is_browser_session_failure(status_code=status_code, message=message):
                raise FirecrawlBrowserSessionError(message, status_code=status_code) from exc
            raise FirecrawlBrowserTransportError(message, status_code=status_code) from exc
        if raw.get("success") is False or raw.get("error") or int(raw.get("exit_code") or raw.get("exitCode") or 0) != 0:
            error = str(raw.get("error") or raw.get("stderr") or "browser code execution failed")
            self._record_key_result(lease.key_id, success=False, error=error)
            if _is_browser_session_failure(status_code=None, message=error):
                raise FirecrawlBrowserSessionError(error)
            raise FirecrawlBrowserTransportError(error)
        self._record_key_result(lease.key_id, success=True)
        return raw

    def release(self, lease: _BrowserSessionLease) -> None:
        """Return a lease to the pool and retire sessions near their hard TTL."""

        retire = self._release_or_remove(lease, reason=None)
        if retire is not None:
            credits_billed = self._delete_remote_session(retire["session_id"], retire["key_id"])
            self._record_session_retired(retire, credits_billed=credits_billed)
        self._debug_pool_status(
            "lease_released",
            session_id=lease.session_id,
            key_id=lease.key_id,
            url=lease.request_url,
        )

    def invalidate(self, lease: _BrowserSessionLease, *, reason: str) -> None:
        """Remove a broken session before another worker can lease it."""

        retired = self._release_or_remove(lease, reason=reason)
        if retired is not None:
            credits_billed = self._delete_remote_session(retired["session_id"], retired["key_id"])
            self._record_session_retired(retired, credits_billed=credits_billed)
        self._debug_pool_status(
            "lease_invalidated",
            session_id=lease.session_id,
            key_id=lease.key_id,
            url=lease.request_url,
            reason=reason,
        )

    def _reserve(self, *, lease_timeout_s: float) -> dict[str, Any]:
        now = time.time()
        owner = f"pid={os.getpid()}:thread={threading.get_ident()}"

        def update(state: dict[str, Any]) -> dict[str, Any]:
            self._cleanup_state(state, now=now)
            sessions = state["sessions"]
            for record in sessions:
                if record.get("state") != "idle":
                    continue
                if float(record.get("expires_at") or 0.0) <= now + self.expiry_safety_s:
                    continue
                lease_id = uuid.uuid4().hex
                record.update(
                    {
                        "state": "leased",
                        "lease_id": lease_id,
                        "lease_owner": owner,
                        "lease_until": now + lease_timeout_s,
                        "last_used_at": now,
                    }
                )
                self._record_session_metric(state, key_id=str(record.get("key_id") or ""), metric="leases")
                self._record_session_metric(state, key_id=str(record.get("key_id") or ""), metric="reuses")
                return {"kind": "leased", "record": dict(record)}

            active_count = sum(
                record.get("state") in {"idle", "leased", "creating"}
                for record in sessions
            )
            if active_count >= self.max_sessions:
                return {"kind": "wait"}

            active_key_ids = self.key_pool.active_key_ids()
            if not active_key_ids:
                return {"kind": "no_key"}
            offset = int(state.get("next_key_index") or 0) % len(active_key_ids)
            key_id = active_key_ids[offset]
            state["next_key_index"] = (offset + 1) % len(active_key_ids)
            lease_id = uuid.uuid4().hex
            record = {
                "slot_id": uuid.uuid4().hex,
                "state": "creating",
                "key_id": key_id,
                "creation_token": uuid.uuid4().hex,
                "create_until": now + self.create_timeout_s,
                "lease_id": lease_id,
                "lease_owner": owner,
                "lease_until": now + lease_timeout_s,
                "created_at": now,
                "last_used_at": now,
            }
            sessions.append(record)
            return {"kind": "create", "record": dict(record)}

        return self._with_locked_state(update)

    def _create_reserved_session(
        self,
        reservation: dict[str, Any],
        *,
        request_url: str | None = None,
    ) -> _BrowserSessionLease:
        key_id = str(reservation["key_id"])
        creation_token = str(reservation["creation_token"])
        try:
            response = self._app(key_id).browser(
                ttl=self.session_ttl_s,
                activity_ttl=self.activity_ttl_s,
                stream_web_view=False,
            )
            raw = FirecrawlClient._response_as_dict(response)
            if raw.get("success") is False or not raw.get("id"):
                raise FirecrawlBrowserSessionError(
                    str(raw.get("error") or "Browser session creation returned no id."),
                    status_code=FirecrawlClient._root_status_code(raw),
                )
        except Exception as exc:
            self._record_key_result(key_id, success=False, error=str(exc), exc=exc)
            self._drop_creation(creation_token)
            raise

        session_id = str(raw["id"])
        expires_at = _browser_expiry_timestamp(raw.get("expires_at") or raw.get("expiresAt"))
        expires_at = expires_at if expires_at is not None else time.time() + self.session_ttl_s

        def publish(state: dict[str, Any]) -> dict[str, Any] | None:
            self._cleanup_state(state, now=time.time())
            for record in state["sessions"]:
                if record.get("creation_token") != creation_token:
                    continue
                record.update(
                    {
                        "state": "leased",
                        "session_id": session_id,
                        "expires_at": expires_at,
                        "created_at": time.time(),
                        "last_used_at": time.time(),
                    }
                )
                record.pop("creation_token", None)
                record.pop("create_until", None)
                return dict(record)
            return None

        published = self._with_locked_state(publish)
        if published is None:
            # The creator lost its reservation (for example after a long pause).
            # Do not leak the new remote session.
            self._delete_remote_session(session_id, key_id)
            raise FirecrawlBrowserSessionError("Browser session reservation expired before it could be published.")
        self._record_session_metric_for_key(key_id, metric="created")
        self._record_session_metric_for_key(key_id, metric="leases")
        self._record_key_result(key_id, success=True)
        return self._lease_from_record(published, request_url=request_url)

    def _drop_creation(self, creation_token: str) -> None:
        def update(state: dict[str, Any]) -> None:
            state["sessions"] = [
                record
                for record in state["sessions"]
                if record.get("creation_token") != creation_token
            ]
            return None

        self._with_locked_state(update)

    def _release_or_remove(
        self,
        lease: _BrowserSessionLease,
        *,
        reason: str | None,
    ) -> dict[str, str] | None:
        now = time.time()

        def update(state: dict[str, Any]) -> dict[str, str] | None:
            self._cleanup_state(state, now=now)
            for index, record in enumerate(state["sessions"]):
                if record.get("slot_id") != lease.slot_id or record.get("lease_id") != lease.lease_id:
                    continue
                if reason or float(record.get("expires_at") or 0.0) <= now + self.expiry_safety_s:
                    state["sessions"].pop(index)
                    return {
                        "session_id": str(record.get("session_id") or ""),
                        "key_id": str(record.get("key_id") or ""),
                        "reason": reason or "expiry_safety_window",
                    }
                record.update(
                    {
                        "state": "idle",
                        "last_used_at": now,
                    }
                )
                record.pop("lease_id", None)
                record.pop("lease_owner", None)
                record.pop("lease_until", None)
                return None
            return None

        return self._with_locked_state(update)

    def _cleanup_state(self, state: dict[str, Any], *, now: float) -> None:
        normalized = self._normalize_state(state)
        kept: list[dict[str, Any]] = []
        for record in normalized["sessions"]:
            status = str(record.get("state") or "")
            expires_at = float(record.get("expires_at") or 0.0)
            if status == "creating":
                if float(record.get("create_until") or 0.0) > now:
                    kept.append(record)
                else:
                    self._record_session_metric(
                        normalized,
                        key_id=str(record.get("key_id") or ""),
                        metric="creation_abandoned",
                    )
                continue
            if status == "leased" and float(record.get("lease_until") or 0.0) <= now:
                if expires_at > now + self.expiry_safety_s:
                    record.update({"state": "idle", "last_used_at": now})
                    record.pop("lease_id", None)
                    record.pop("lease_owner", None)
                    record.pop("lease_until", None)
                    kept.append(record)
                else:
                    self._record_session_metric(
                        normalized,
                        key_id=str(record.get("key_id") or ""),
                        metric="expired",
                    )
                continue
            if status == "idle" and expires_at > now + self.expiry_safety_s:
                kept.append(record)
                continue
            if status == "leased":
                # Do not delete a still-leased session merely because it crossed
                # the nominal TTL; the lease owner is allowed to finish its
                # bounded execute call and will retire it on release.
                kept.append(record)
            elif status == "idle":
                self._record_session_metric(
                    normalized,
                    key_id=str(record.get("key_id") or ""),
                    metric="expired",
                )
        normalized["sessions"] = kept
        normalized["waiters"] = [
            waiter
            for waiter in normalized["waiters"]
            if float(waiter.get("expires_at") or 0.0) > now
        ]
        state.clear()
        state.update(normalized)

    @staticmethod
    def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(state or {})
        sessions = normalized.get("sessions")
        normalized["sessions"] = [dict(record) for record in sessions if isinstance(record, dict)] if isinstance(sessions, list) else []
        waiters = normalized.get("waiters")
        normalized["waiters"] = [dict(waiter) for waiter in waiters if isinstance(waiter, dict)] if isinstance(waiters, list) else []
        metrics = dict(normalized.get("metrics") or {})
        metrics.setdefault("created", 0)
        metrics.setdefault("leases", 0)
        metrics.setdefault("reuses", 0)
        metrics.setdefault("retired", 0)
        metrics.setdefault("expired", 0)
        metrics.setdefault("creation_abandoned", 0)
        metrics.setdefault("credits_billed", 0)
        metrics.setdefault("credit_reports", 0)
        metrics["by_key"] = dict(metrics.get("by_key") or {})
        normalized["metrics"] = metrics
        normalized["version"] = FirecrawlBrowserSessionManager._STATE_VERSION
        normalized.setdefault("next_key_index", 0)
        normalized["updated_at"] = time.time()
        return normalized

    @staticmethod
    def _record_session_metric(
        state: dict[str, Any],
        *,
        key_id: str,
        metric: str,
        amount: int = 1,
    ) -> None:
        """Update persistent aggregate and per-key Browser session metrics."""

        metrics = state.setdefault("metrics", {})
        amount = max(0, int(amount))
        metrics[metric] = max(0, int(metrics.get(metric) or 0)) + amount
        if not key_id:
            return
        by_key = metrics.setdefault("by_key", {})
        key_metrics = dict(by_key.get(key_id) or {})
        key_metrics[metric] = max(0, int(key_metrics.get(metric) or 0)) + amount
        by_key[key_id] = key_metrics

    def _record_session_metric_for_key(self, key_id: str, *, metric: str, amount: int = 1) -> None:
        def update(state: dict[str, Any]) -> None:
            self._cleanup_state(state, now=time.time())
            self._record_session_metric(state, key_id=key_id, metric=metric, amount=amount)
            return None

        self._with_locked_state(update)

    def _record_session_retired(
        self,
        retired: dict[str, str],
        *,
        credits_billed: int | None,
    ) -> None:
        key_id = str(retired.get("key_id") or "")

        def update(state: dict[str, Any]) -> None:
            self._cleanup_state(state, now=time.time())
            self._record_session_metric(state, key_id=key_id, metric="retired")
            if credits_billed is not None:
                self._record_session_metric(
                    state,
                    key_id=key_id,
                    metric="credits_billed",
                    amount=credits_billed,
                )
                self._record_session_metric(state, key_id=key_id, metric="credit_reports")
            return None

        self._with_locked_state(update)

    def _register_waiter(self, waiter_id: str, *, expires_at: float) -> None:
        owner = f"pid={os.getpid()}:thread={threading.get_ident()}"

        def update(state: dict[str, Any]) -> None:
            self._cleanup_state(state, now=time.time())
            state["waiters"] = [
                waiter
                for waiter in state["waiters"]
                if waiter.get("waiter_id") != waiter_id
            ]
            state["waiters"].append(
                {
                    "waiter_id": waiter_id,
                    "owner": owner,
                    "created_at": time.time(),
                    "expires_at": expires_at,
                }
            )
            return None

        self._with_locked_state(update)

    def _remove_waiter(self, waiter_id: str) -> None:
        def update(state: dict[str, Any]) -> None:
            self._cleanup_state(state, now=time.time())
            state["waiters"] = [
                waiter
                for waiter in state["waiters"]
                if waiter.get("waiter_id") != waiter_id
            ]
            return None

        self._with_locked_state(update)

    def pool_snapshot(self) -> dict[str, Any]:
        """Return a cross-worker view of sessions, queue depth, and credits."""

        def read(state: dict[str, Any]) -> dict[str, Any]:
            self._cleanup_state(state, now=time.time())
            sessions = list(state["sessions"])
            metrics = dict(state["metrics"])
            return {
                "max_sessions": self.max_sessions,
                "in_use": sum(record.get("state") == "leased" for record in sessions),
                "idle": sum(record.get("state") == "idle" for record in sessions),
                "creating": sum(record.get("state") == "creating" for record in sessions),
                "waiting": len(state["waiters"]),
                "metrics": metrics,
            }

        snapshot = self._with_locked_state(read)
        snapshot["key_pool"] = self.key_pool.status()
        return snapshot

    def _debug_pool_status(self, event: str, **details: object) -> None:
        # Suppress normal session/lease lifecycle noise. Only broken leases or
        # failed downloads should expose the pool snapshot.
        if event not in {"lease_invalidated", "download_failed"}:
            return
        if not _firecrawl_browser_debug_enabled():
            return
        snapshot = self.pool_snapshot()
        metrics = snapshot["metrics"]
        key_pool = snapshot["key_pool"]
        by_key = metrics.get("by_key") if isinstance(metrics.get("by_key"), dict) else {}
        per_key = ",".join(
            f"{key_id[:8]}:created={int(values.get('created') or 0)},"
            f"retired={int(values.get('retired') or 0)},"
            f"billed={int(values.get('credits_billed') or 0)}"
            for key_id, values in sorted(by_key.items())
            if isinstance(values, dict)
        ) or "-"
        context = " ".join(f"{key}={value!r}" for key, value in details.items())
        suffix = f" {context}" if context else ""
        print(
            "[firecrawl-browser] "
            f"event={event} in_use={snapshot['in_use']}/{snapshot['max_sessions']} "
            f"idle={snapshot['idle']} creating={snapshot['creating']} waiting={snapshot['waiting']} "
            f"created={int(metrics.get('created') or 0)} leases={int(metrics.get('leases') or 0)} "
            f"reuses={int(metrics.get('reuses') or 0)} retired={int(metrics.get('retired') or 0)} "
            f"expired={int(metrics.get('expired') or 0)} "
            f"browser_credits_billed={int(metrics.get('credits_billed') or 0)} "
            f"pool_credits_consumed={int(key_pool.get('credits_consumed') or 0)} "
            f"pool_credits_remaining={int(key_pool.get('remaining_credits_total') or 0)} "
            f"key_usage='{per_key}'{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def _lease_from_record(
        self,
        record: dict[str, Any],
        *,
        request_url: str | None = None,
    ) -> _BrowserSessionLease:
        session_id = str(record.get("session_id") or "")
        if not session_id:
            raise FirecrawlBrowserSessionError("Leased Browser session has no session id.")
        return _BrowserSessionLease(
            manager=self,
            slot_id=str(record["slot_id"]),
            session_id=session_id,
            key_id=str(record["key_id"]),
            lease_id=str(record["lease_id"]),
            request_url=str(request_url or ""),
        )

    def _app(self, key_id: str) -> Any:
        api_key = self.key_pool.key_for_id(key_id)
        if self._app_factory is not None:
            return self._app_factory(api_key=api_key)
        try:
            from firecrawl.v2 import FirecrawlClient as FirecrawlV2Client
        except ImportError as exc:
            raise RuntimeError(
                "Firecrawl Browser requires SDK v2 support. Install or upgrade the firecrawl package. "
                f"Original ImportError: {exc}"
            ) from exc
        return FirecrawlV2Client(
            api_key=api_key, api_url=_FIRECRAWL_API_URL, timeout=self.api_timeout_s
        )

    def _record_key_result(
        self,
        key_id: str,
        *,
        success: bool,
        error: str | None = None,
        exc: BaseException | None = None,
        credits_used: int = 0,
    ) -> None:
        status_code = _status_code_from_exception(exc) if exc is not None else None
        self.key_pool.record_result(
            key_id,
            success=success,
            error=error,
            credits_used=max(0, int(credits_used)),
            status_code=status_code,
            key_auth_failed=status_code == 401,
            key_credits_exhausted=_is_firecrawl_credit_exhausted(
                status_code=status_code,
                response_body=error or str(exc or ""),
            ),
        )

    def _delete_remote_session(self, session_id: str, key_id: str) -> int | None:
        if not session_id or not key_id:
            return None
        try:
            response = self._app(key_id).delete_browser(session_id)
            raw = FirecrawlClient._response_as_dict(response)
            if raw.get("success") is False:
                return None
            credits_billed = _non_negative_int(raw.get("credits_billed", raw.get("creditsBilled")))
            self._record_key_result(key_id, success=True, credits_used=credits_billed or 0)
            return credits_billed
        except Exception as exc:
            # Session cleanup is best-effort: it may already have expired, and
            # a cleanup failure must not make image evidence unavailable.
            self._record_key_result(key_id, success=False, error=f"Browser session cleanup failed: {exc}", exc=exc)
            return None

    @contextmanager
    def _state_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _with_locked_state(self, callback: Callable[[dict[str, Any]], Any]) -> Any:
        with self._state_lock():
            state = self._read_state()
            result = callback(state)
            self._write_state(state)
            return result

    def _read_state(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            raw = {}
        return self._normalize_state(raw if isinstance(raw, dict) else {})

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_path.parent,
                prefix=f".{self.state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(state, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.state_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


class FirecrawlBrowserImageDownloader:
    """Download raw image bytes through a reusable Firecrawl Browser pool.

    This is the only image-facing interface exposed by the Firecrawl backend.
    Agent code receives ordinary ``bytes`` and a MIME type; it never manages a
    key, session id, Browser API request, or Base64 payload.
    """

    def __init__(
        self,
        *,
        session_manager: FirecrawlBrowserSessionManager,
        max_image_bytes: int = 20 * 1024 * 1024,
        acquire_timeout_s: float = 60.0,
        retries: int = 1,
        retry_sleep_s: float = 0.5,
        timeout_cap_s: float | None = None,
        same_session_timeout_retries: int = 0,
        same_session_retry_sleep_s: float = 1.0,
        wikimedia_variant_resolution: bool = True,
        wikimedia_max_long_edge: int = _WIKIMEDIA_VARIANT_MAX_LONG_EDGE,
        wikimedia_metadata_timeout_s: float = _WIKIMEDIA_VARIANT_METADATA_TIMEOUT_S,
    ) -> None:
        self.session_manager = session_manager
        self.max_image_bytes = max(1024, int(max_image_bytes))
        self.acquire_timeout_s = max(1.0, float(acquire_timeout_s))
        self.retries = max(0, int(retries))
        self.retry_sleep_s = max(0.0, float(retry_sleep_s))
        self.timeout_cap_s = (
            max(1.0, min(float(timeout_cap_s), 300.0))
            if timeout_cap_s is not None and float(timeout_cap_s) > 0
            else None
        )
        self.same_session_timeout_retries = max(0, int(same_session_timeout_retries))
        self.same_session_retry_sleep_s = max(0.0, float(same_session_retry_sleep_s))
        self.wikimedia_variant_resolution = bool(wikimedia_variant_resolution)
        self.wikimedia_max_long_edge = max(1, int(wikimedia_max_long_edge))
        self.wikimedia_metadata_timeout_s = max(1.0, float(wikimedia_metadata_timeout_s))

    @classmethod
    def from_environment(
        cls,
        *,
        api_keys: list[str] | None = None,
        app_factory: Callable[..., Any] | None = None,
        pool_state_path: str | Path | None = None,
        session_state_path: str | Path | None = None,
    ) -> "FirecrawlBrowserImageDownloader":
        return cls(
            session_manager=FirecrawlBrowserSessionManager.from_environment(
                api_keys=api_keys,
                app_factory=app_factory,
                pool_state_path=pool_state_path,
                session_state_path=session_state_path,
            ),
            max_image_bytes=_env_int("FIRECRAWL_BROWSER_IMAGE_MAX_BYTES", 20 * 1024 * 1024),
            acquire_timeout_s=_env_float("FIRECRAWL_BROWSER_POOL_ACQUIRE_TIMEOUT_S", 60.0),
            # Retry a timeout on the same session when configured, then allow
            # one download retry with a fresh session.
            retries=_env_int("FIRECRAWL_BROWSER_IMAGE_RETRIES", 1),
            retry_sleep_s=_env_float("FIRECRAWL_BROWSER_IMAGE_RETRY_SLEEP_S", 2.0),
            timeout_cap_s=(
                _env_float("FIRECRAWL_BROWSER_IMAGE_TIMEOUT_S", 0.0)
                or None
            ),
            same_session_timeout_retries=_env_int(
                "FIRECRAWL_BROWSER_SAME_SESSION_TIMEOUT_RETRIES", 0
            ),
            same_session_retry_sleep_s=_env_float(
                "FIRECRAWL_BROWSER_SAME_SESSION_RETRY_SLEEP_S", 1.0
            ),
            wikimedia_variant_resolution=_env_flag(
                "FIRECRAWL_BROWSER_WIKIMEDIA_VARIANT_RESOLUTION", True
            ),
            wikimedia_max_long_edge=_env_int(
                "FIRECRAWL_BROWSER_WIKIMEDIA_MAX_LONG_EDGE",
                _WIKIMEDIA_VARIANT_MAX_LONG_EDGE,
            ),
            wikimedia_metadata_timeout_s=_env_float(
                "FIRECRAWL_BROWSER_WIKIMEDIA_METADATA_TIMEOUT_S",
                _WIKIMEDIA_VARIANT_METADATA_TIMEOUT_S,
            ),
        )

    def download(
        self,
        url: str,
        *,
        referer_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> FirecrawlBrowserImageDownload:
        requested_url = str(url or "").strip()
        if not requested_url:
            raise FirecrawlBrowserError("Image URL is required.")
        try:
            requested_url = normalize_http_url(requested_url)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise FirecrawlBrowserError("Image URL is invalid.") from exc
        target_url, variant_resolution = self._prepare_wikimedia_download(requested_url)
        timeout_s = max(1.0, min(float(timeout_s), 300.0))
        if self.timeout_cap_s is not None:
            timeout_s = min(timeout_s, self.timeout_cap_s)
        # Let the target request fail inside our JS catch block before the
        # outer Browser Execute timeout kills the whole script.
        request_timeout_s = max(1.0, timeout_s - min(5.0, timeout_s * 0.1))
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            lease: _BrowserSessionLease | None = None
            same_session_retry = 0
            while True:
                try:
                    if lease is None:
                        lease = self.session_manager.acquire(
                            acquire_timeout_s=self.acquire_timeout_s,
                            lease_timeout_s=max(
                                timeout_s + 30.0,
                                self.session_manager.api_timeout_s + 30.0,
                            ),
                            request_url=requested_url,
                        )
                    response = self.session_manager.execute(
                        lease,
                        code=self._download_code(
                            target_url,
                            referer_url=referer_url,
                            request_timeout_ms=int(request_timeout_s * 1000),
                            wikimedia_resolution=variant_resolution,
                        ),
                        timeout_s=timeout_s,
                    )
                    result = self._decode_download_response(
                        response,
                        requested_url=requested_url,
                        session_id=lease.session_id,
                        key_id=lease.key_id,
                    )
                    if (
                        variant_resolution is not None
                        and variant_resolution.get("mode") == "lookup"
                        and result.variant_resolved
                    ):
                        _write_wikimedia_variant_cache(
                            {
                                "requested_url": requested_url,
                                "download_url": result.download_url or target_url,
                                "max_long_edge": self.wikimedia_max_long_edge,
                                "variant_source": result.variant_source,
                                "original_width": result.original_width,
                                "original_height": result.original_height,
                                "download_width": result.download_width,
                                "download_height": result.download_height,
                                "resolved": True,
                                "resolved_at": time.time(),
                            }
                        )
                    self.session_manager._debug_pool_status(
                        "download_succeeded",
                        session_id=lease.session_id,
                        key_id=lease.key_id,
                        url=requested_url,
                        download_url=result.download_url or target_url,
                        variant_source=result.variant_source,
                        attempt=attempt + 1,
                        session_retry=same_session_retry,
                        http_status=result.status_code,
                        content_type=result.content_type,
                        byte_count=len(result.payload),
                        resolved_url=result.resolved_url,
                    )
                    lease.release()
                    return result
                except Exception as exc:
                    last_error = exc
                    status_code = getattr(exc, "status_code", None)
                    timeout_error = (
                        isinstance(exc, (FirecrawlBrowserTransportError, FirecrawlBrowserTargetRequestError))
                        and _is_timeout_message(str(exc))
                    )
                    retry_same_session = timeout_error and same_session_retry < self.same_session_timeout_retries
                    retry_new_session = (
                        not retry_same_session
                        and attempt < self.retries
                        and self._is_retryable(exc)
                    )
                    self.session_manager._debug_pool_status(
                        "download_failed",
                        session_id=lease.session_id if lease is not None else "",
                        key_id=lease.key_id if lease is not None else "",
                        url=requested_url,
                        download_url=target_url,
                        attempt=attempt + 1,
                        session_retry=same_session_retry,
                        http_status=status_code,
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                        retry_scope=(
                            "same_session"
                            if retry_same_session
                            else "new_session"
                            if retry_new_session
                            else None
                        ),
                        will_retry=retry_same_session or retry_new_session,
                    )
                    if retry_same_session:
                        same_session_retry += 1
                        tracked_sleep(
                            self.same_session_retry_sleep_s,
                            reason="firecrawl_browser_same_session_retry",
                            tool="firecrawl_browser",
                            error=exc,
                            url=requested_url,
                            error_type=type(exc).__name__,
                            fallback_sleep=time.sleep,
                        )
                        continue

                    if lease is not None:
                        # An outer transport timeout is unsafe to reuse. A
                        # structured target-request timeout is reusable unless
                        # its configured same-session retry has already failed.
                        destroy_after_same_session_retry = (
                            isinstance(exc, FirecrawlBrowserTargetRequestError)
                            and same_session_retry > 0
                        )
                        if (
                            isinstance(exc, FirecrawlBrowserSessionError)
                            or isinstance(exc, FirecrawlBrowserTransportError)
                            and timeout_error
                            or destroy_after_same_session_retry
                        ):
                            lease.invalidate(reason=str(exc))
                        else:
                            lease.release()
                    if not retry_new_session:
                        raise
                    tracked_sleep(
                        self.retry_sleep_s * (attempt + 1),
                        reason="firecrawl_browser_retry",
                        tool="firecrawl_browser",
                        error=exc,
                        url=requested_url,
                        error_type=type(exc).__name__,
                        fallback_sleep=time.sleep,
                    )
                    break
        assert last_error is not None
        raise last_error

    def _prepare_wikimedia_download(
        self,
        requested_url: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Prepare a Wikimedia URL and its in-Browser variant resolver.

        The MediaWiki API lookup is deliberately carried out by the Browser
        JavaScript below.  This keeps Wikimedia traffic out of the worker and
        lets the existing Firecrawl egress handle both metadata and image
        requests.  A durable mapping is used only after a successful image
        download, so a transient API failure never poisons the cache.
        """

        if not self.wikimedia_variant_resolution or not _is_wikimedia_upload_url(requested_url):
            return requested_url, None
        cached = _read_wikimedia_variant_cache(
            requested_url,
            self.wikimedia_max_long_edge,
        )
        if cached is not None:
            return str(cached["download_url"]), {
                "mode": "cached",
                "download_url": str(cached["download_url"]),
                "variant_source": str(cached.get("variant_source") or "cache"),
                "original_width": cached.get("original_width"),
                "original_height": cached.get("original_height"),
                "download_width": cached.get("download_width"),
                "download_height": cached.get("download_height"),
            }
        file_spec = _wikimedia_upload_file_spec(requested_url)
        if file_spec is None:
            return requested_url, None
        widths = [self.wikimedia_max_long_edge]
        for width in (
            1536,
            1440,
            1280,
            1200,
            1152,
            1080,
            1024,
            960,
            800,
            768,
            720,
            640,
            512,
            480,
            400,
            330,
            320,
            256,
        ):
            if width <= self.wikimedia_max_long_edge and width not in widths:
                widths.append(width)
        # MediaWiki returns the closest available thumbnail at or above the
        # requested size.  In practice the first few standard widths find the
        # largest valid variant; cap the probes so a metadata outage cannot
        # consume the whole Browser execution timeout.
        widths = widths[:6]
        return requested_url, {
            "mode": "lookup",
            "api_url": file_spec["api_url"],
            "file_title": file_spec["file_title"],
            "max_long_edge": self.wikimedia_max_long_edge,
            "request_widths": widths,
            "metadata_timeout_ms": int(self.wikimedia_metadata_timeout_s * 1000),
        }

    def _download_code(
        self,
        url: str,
        *,
        referer_url: str | None,
        request_timeout_ms: int,
        wikimedia_resolution: dict[str, Any] | None = None,
    ) -> str:
        # Referer is optional. If it contains raw Unicode or control
        # characters, omit it instead of making Playwright reject the whole
        # request and incorrectly treating the Browser session as broken.
        safe_referer = normalize_http_referer(referer_url) or ""
        safe_url = normalize_http_url(url)
        payload = json.dumps(
            {
                "url": safe_url,
                "referer": safe_referer,
                "timeoutMs": max(1000, min(int(request_timeout_ms), 300000)),
                "wikimediaResolution": wikimedia_resolution,
            },
            ensure_ascii=True,
        )
        return f"""
await (async () => {{
const input = {payload};
const executionStartedAt = Date.now();
const remainingExecutionTimeoutMs = () => Math.max(
  1000,
  Number(input.timeoutMs) - (Date.now() - executionStartedAt)
);
const headers = {{
  "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
  "Accept-Language": "en-US,en;q=0.9",
}};
if (input.referer) headers["Referer"] = input.referer;
const maxBytes = {self.max_image_bytes};
let downloadUrl = input.url;
let variantSource = "original";
let variantResolved = false;
let originalWidth = null;
let originalHeight = null;
let downloadWidth = null;
let downloadHeight = null;
let variantResolutionError = null;

const asPositiveNumber = (value) => {{
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}};

const variantFromCache = (resolution) => {{
  if (!resolution || resolution.mode !== "cached" || !resolution.download_url) return null;
  return {{
    download_url: String(resolution.download_url),
    variant_source: String(resolution.variant_source || "cache"),
    resolved: true,
    original_width: asPositiveNumber(resolution.original_width),
    original_height: asPositiveNumber(resolution.original_height),
    download_width: asPositiveNumber(resolution.download_width),
    download_height: asPositiveNumber(resolution.download_height),
  }};
}};

const parseImageInfo = (body) => {{
  const pages = body && body.query && body.query.pages;
  const pageInfo = Array.isArray(pages)
    ? pages[0]
    : pages && Object.values(pages)[0];
  const imageInfo = pageInfo && Array.isArray(pageInfo.imageinfo)
    ? pageInfo.imageinfo[0]
    : null;
  return imageInfo || null;
}};

const resolveWikimediaVariant = async (resolution) => {{
  const cached = variantFromCache(resolution);
  if (cached) return cached;
  if (!resolution || resolution.mode !== "lookup") return null;
  const apiUrl = String(resolution.api_url || "");
  const fileTitle = String(resolution.file_title || "");
  const maxLongEdge = asPositiveNumber(resolution.max_long_edge);
  const widths = Array.isArray(resolution.request_widths)
    ? resolution.request_widths
    : [];
  if (!apiUrl || !fileTitle || !maxLongEdge || !widths.length) return null;
  const metadataHeaders = {{
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "OpenSearch-VL-FirecrawlBrowserImageDownloader/1.0",
  }};
  for (const requestedWidth of widths) {{
    const width = asPositiveNumber(requestedWidth);
    if (!width) continue;
    let metadataResponse = null;
    try {{
      const query = new URLSearchParams({{
        action: "query",
        format: "json",
        formatversion: "2",
        titles: fileTitle,
        prop: "imageinfo",
        iiprop: "url|size|mime|thumbmime",
        iiurlwidth: String(Math.round(width)),
        iiurlheight: String(Math.round(width)),
        maxlag: "5",
      }}).toString();
      metadataResponse = await page.request.get(`${{apiUrl}}?${{query}}`, {{
        failOnStatusCode: false,
        headers: metadataHeaders,
        timeout: Math.max(
          1000,
          Math.min(
            Number(resolution.metadata_timeout_ms) || 8000,
            remainingExecutionTimeoutMs()
          ),
        ),
      }});
      const status = metadataResponse.status();
      if (status < 200 || status >= 300) {{
        throw new Error(`imageinfo_http_status_${{status}}`);
      }}
      const body = JSON.parse((await metadataResponse.body()).toString("utf8"));
      if (body && body.error) {{
        const errorCode = String(body.error.code || "api_error");
        throw new Error(`imageinfo_api_error_${{errorCode}}`);
      }}
      const info = parseImageInfo(body);
      if (!info) continue;
      const sourceWidth = asPositiveNumber(info.width);
      const sourceHeight = asPositiveNumber(info.height);
      if (sourceWidth && sourceHeight) {{
        originalWidth = sourceWidth;
        originalHeight = sourceHeight;
      }}
      const sourceLongEdge = sourceWidth && sourceHeight
        ? Math.max(sourceWidth, sourceHeight)
        : null;
      if (sourceLongEdge && sourceLongEdge <= maxLongEdge) {{
        return {{
          download_url: String(info.url || input.url),
          variant_source: "imageinfo_original",
          resolved: true,
          original_width: sourceWidth,
          original_height: sourceHeight,
          download_width: sourceWidth,
          download_height: sourceHeight,
        }};
      }}
      const thumbnailUrl = String(info.thumburl || "");
      if (!thumbnailUrl || !/^https?:\\/\\//i.test(thumbnailUrl)) continue;
      let thumbnailWidth = asPositiveNumber(info.thumbwidth);
      let thumbnailHeight = asPositiveNumber(info.thumbheight);
      if (!thumbnailWidth || !thumbnailHeight) {{
        const match = thumbnailUrl.match(/(\\d+)px-[^/]+(?:\\?.*)?$/i);
        if (match) thumbnailWidth = asPositiveNumber(match[1]);
      }}
      if (!thumbnailHeight && thumbnailWidth && sourceWidth && sourceHeight) {{
        thumbnailHeight = Math.max(1, Math.round(thumbnailWidth * sourceHeight / sourceWidth));
      }}
      if (!thumbnailWidth && thumbnailHeight && sourceWidth && sourceHeight) {{
        thumbnailWidth = Math.max(1, Math.round(thumbnailHeight * sourceWidth / sourceHeight));
      }}
      const thumbnailLongEdge = thumbnailWidth && thumbnailHeight
        ? Math.max(thumbnailWidth, thumbnailHeight)
        : null;
      if (thumbnailLongEdge && thumbnailLongEdge <= maxLongEdge) {{
        return {{
          download_url: thumbnailUrl,
          variant_source: "imageinfo_thumbnail",
          resolved: true,
          original_width: sourceWidth,
          original_height: sourceHeight,
          download_width: thumbnailWidth,
          download_height: thumbnailHeight,
        }};
      }}
    }} finally {{
      if (metadataResponse) {{
        try {{ await metadataResponse.dispose(); }} catch (_) {{}}
      }}
    }}
  }}
  return null;
}};

try {{
  const resolvedVariant = await resolveWikimediaVariant(input.wikimediaResolution);
  if (resolvedVariant && resolvedVariant.download_url) {{
    downloadUrl = resolvedVariant.download_url;
    variantSource = String(resolvedVariant.variant_source || "imageinfo");
    variantResolved = Boolean(resolvedVariant.resolved);
    originalWidth = resolvedVariant.original_width || null;
    originalHeight = resolvedVariant.original_height || null;
    downloadWidth = resolvedVariant.download_width || null;
    downloadHeight = resolvedVariant.download_height || null;
  }}
}} catch (error) {{
  variantResolutionError = String(error && error.message ? error.message : error || "variant resolution failed").slice(0, 500);
}}

let response = null;
let phase = "request";
const output = {{
  status: null,
  resolved_url: downloadUrl,
  download_url: downloadUrl,
  variant_source: variantSource,
  variant_resolved: variantResolved,
  original_width: originalWidth,
  original_height: originalHeight,
  download_width: downloadWidth,
  download_height: downloadHeight,
  variant_resolution_error: variantResolutionError,
  content_type: "",
  body_base64: null,
  byte_count: 0,
  error: null,
  target_error_type: null,
  target_error_message: null,
  target_phase: phase,
}};
try {{
  response = await page.request.get(downloadUrl, {{
    failOnStatusCode: false,
    headers,
    timeout: remainingExecutionTimeoutMs(),
  }});
  phase = "response";
  const responseHeaders = response.headers();
  const contentType = String(responseHeaders["content-type"] || "").split(";", 1)[0].trim().toLowerCase();
  const status = response.status();
  const resolvedUrl = response.url();
  const declaredLength = Number(responseHeaders["content-length"] || 0);
  const likelyImage = contentType.startsWith("image/") || !contentType || contentType === "application/octet-stream" || contentType === "binary/octet-stream";
  output.status = status;
  output.resolved_url = resolvedUrl;
  output.content_type = contentType;
  output.target_phase = phase;
  if (status < 200 || status >= 300) {{
    output.error = `http_status_${{status}}`;
  }} else if (!likelyImage) {{
    output.error = "non_image_content_type";
  }} else if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {{
    output.error = "image_too_large";
  }} else {{
    phase = "body";
    output.target_phase = phase;
    const body = await response.body();
    output.byte_count = body.length;
    if (body.length > maxBytes) {{
      output.error = "image_too_large";
    }} else {{
      output.body_base64 = body.toString("base64");
    }}
  }}
}} catch (error) {{
  const message = String(error && error.message ? error.message : error || "target request failed");
  const normalized = message.toLowerCase();
  output.target_phase = phase;
  output.target_error_type = normalized.includes("timeout") || normalized.includes("timed out")
    ? "target_request_timeout"
    : "target_request_error";
  output.target_error_message = message.slice(0, 500);
  output.error = output.target_error_type;
}} finally {{
  if (response) {{
    try {{ await response.dispose(); }} catch (_) {{}}
  }}
}}
return JSON.stringify(output);
}})();
""".strip()

    def _decode_download_response(
        self,
        response: dict[str, Any],
        *,
        requested_url: str,
        session_id: str,
        key_id: str,
    ) -> FirecrawlBrowserImageDownload:
        raw_output = response.get("result") or response.get("stdout") or response.get("output")
        if isinstance(raw_output, dict):
            payload = raw_output
        else:
            try:
                payload = json.loads(str(raw_output or ""))
            except json.JSONDecodeError as exc:
                raise FirecrawlBrowserSessionError(
                    f"Firecrawl Browser returned an unparseable image response: {raw_output!r}"
                ) from exc
        if not isinstance(payload, dict):
            raise FirecrawlBrowserSessionError("Firecrawl Browser returned a non-object image response.")
        try:
            status_code = int(payload.get("status") or 0)
        except (TypeError, ValueError):
            status_code = 0
        resolved_url = str(payload.get("resolved_url") or requested_url)
        content_type = _normalized_content_type(payload.get("content_type"))
        error = str(payload.get("error") or "")
        target_error_type = str(payload.get("target_error_type") or "")
        target_error_message = str(payload.get("target_error_message") or "")
        target_phase = str(payload.get("target_phase") or "request")
        download_url = str(payload.get("download_url") or requested_url)
        variant_source = str(payload.get("variant_source") or "original")
        variant_resolved = payload.get("variant_resolved") is True or str(
            payload.get("variant_resolved") or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        original_width = _positive_int(payload.get("original_width"))
        original_height = _positive_int(payload.get("original_height"))
        download_width = _positive_int(payload.get("download_width"))
        download_height = _positive_int(payload.get("download_height"))
        variant_resolution_error = str(payload.get("variant_resolution_error") or "")
        if error in {"target_request_timeout", "target_request_error"} or target_error_type:
            detail = target_error_message or error or target_error_type
            raise FirecrawlBrowserTargetRequestError(
                f"Firecrawl Browser target request failed during {target_phase} "
                f"for {requested_url}: {detail}",
                status_code=status_code or None,
            )
        if status_code < 200 or status_code >= 300:
            raise FirecrawlBrowserHttpError(
                f"Firecrawl Browser image request returned HTTP {status_code} for {requested_url}",
                status_code=status_code or None,
            )
        if error == "non_image_content_type":
            raise FirecrawlBrowserNonImageError(
                f"Firecrawl Browser URL did not return an image ({content_type or 'unknown'}): {requested_url}",
                status_code=status_code,
            )
        if error:
            raise FirecrawlBrowserError(
                f"Firecrawl Browser image download failed for {requested_url}: {error}",
                status_code=status_code or None,
            )
        encoded = payload.get("body_base64")
        if not isinstance(encoded, str) or not encoded:
            raise FirecrawlBrowserNonImageError(
                f"Firecrawl Browser returned no image payload for {requested_url}",
                status_code=status_code,
            )
        if len(encoded) > ((self.max_image_bytes + 2) // 3) * 4 + 8:
            raise FirecrawlBrowserError(
                f"Firecrawl Browser Base64 payload exceeds configured image limit for {requested_url}",
                status_code=status_code,
            )
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise FirecrawlBrowserSessionError(
                f"Firecrawl Browser returned invalid Base64 image data for {requested_url}"
            ) from exc
        if not image_bytes:
            raise FirecrawlBrowserNonImageError(
                f"Firecrawl Browser returned an empty image payload for {requested_url}",
                status_code=status_code,
            )
        sniffed_content_type = _sniff_image_content_type(image_bytes)
        if not content_type.startswith("image/"):
            content_type = sniffed_content_type
        if not content_type.startswith("image/"):
            raise FirecrawlBrowserNonImageError(
                f"Firecrawl Browser payload is not a recognized image for {requested_url}",
                status_code=status_code,
            )
        return FirecrawlBrowserImageDownload(
            payload=image_bytes,
            content_type=content_type,
            requested_url=requested_url,
            resolved_url=resolved_url,
            status_code=status_code,
            session_id=session_id,
            key_id=key_id,
            download_url=download_url,
            variant_source=variant_source,
            variant_resolved=variant_resolved,
            original_width=original_width,
            original_height=original_height,
            download_width=download_width,
            download_height=download_height,
            variant_resolution_error=variant_resolution_error,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, FirecrawlBrowserNonImageError):
            return False
        status_code = getattr(exc, "status_code", None)
        # A destroyed remote session is reported as a 410 by the Browser
        # execute endpoint.  This is not the target image's HTTP 410 and can
        # succeed on a newly acquired session.  Keep ordinary target 410s
        # non-retryable.
        if isinstance(exc, FirecrawlBrowserSessionError):
            message = str(exc).lower()
            if "session destroyed" in message or "browser session" in message and "destroyed" in message:
                return True
        return status_code is None or status_code in {408, 429, 500, 502, 503, 504}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return float(default)


def _firecrawl_browser_debug_enabled() -> bool:
    """Whether to emit one structured pool line per Browser lease lifecycle."""

    raw = str(os.environ.get("FIRECRAWL_BROWSER_DEBUG") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _non_negative_int(value: object) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _browser_expiry_timestamp(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def _is_browser_session_failure(*, status_code: int | None, message: str) -> bool:
    """Distinguish a dead remote Browser session from a transient transport error."""

    if status_code in {401, 404, 410}:
        return True
    normalized = str(message or "").lower()
    return (
        "session destroyed" in normalized
        or "browser session has been destroyed" in normalized
        or ("browser session" in normalized and "not found" in normalized)
    )


def _is_timeout_message(message: str) -> bool:
    normalized = str(message or "").lower()
    return "timeout" in normalized or "timed out" in normalized


def _normalized_content_type(value: object) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def _sniff_image_content_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith(b"BM"):
        return "image/bmp"
    if len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    stripped = payload.lstrip().lower()
    if stripped.startswith(b"<svg") or b"<svg" in stripped[:256]:
        return "image/svg+xml"
    return ""


def acquire_firecrawl_api_key() -> tuple[str, dict[str, Any]]:
    """Return the explicitly configured Firecrawl API key."""

    return _FixedFirecrawlKeyProvider().acquire_key()
