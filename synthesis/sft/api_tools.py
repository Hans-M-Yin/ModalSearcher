"""Shared runtime context, image handling, and tool dispatcher for inference and RL."""

from __future__ import annotations
import base64
from copy import deepcopy
import contextvars
import io
import json
import logging
import os
import threading
import time
import uuid
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import requests
from PIL import Image, ImageOps
from synthesis.retry_monitor import tracked_sleep
from . import tools


logger = logging.getLogger(__name__)


I2I_UPLOAD_MAX_LONG_EDGE = 1000


_SFT_TIMING_ENABLED_ENV = "SFT_TIMING_DEBUG"


def _sft_timing_enabled() -> bool:
    return str(os.environ.get(_SFT_TIMING_ENABLED_ENV, "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass(slots=True)
class _TrajectoryTiming:
    events: dict[str, tuple[float, int]] = field(default_factory=dict)

    def add(self, name: str, elapsed_s: float) -> None:
        total_s, calls = self.events.get(name, (0.0, 0))
        self.events[name] = (total_s + max(0.0, float(elapsed_s)), calls + 1)


_ACTIVE_TRAJECTORY_TIMING: contextvars.ContextVar[_TrajectoryTiming | None] = contextvars.ContextVar(
    "sft_active_trajectory_timing",
    default=None,
)


def _record_active_timing(name: str, elapsed_s: float) -> None:
    if not _sft_timing_enabled():
        return
    active = _ACTIVE_TRAJECTORY_TIMING.get()
    if active is not None:
        active.add(name, elapsed_s)


def _timed_call(name: str, callback: Callable[[], Any]) -> Any:
    """Run a module call and record wall time, including internal sleeps/retries."""

    if not _sft_timing_enabled() or _ACTIVE_TRAJECTORY_TIMING.get() is None:
        return callback()
    started_at = time.perf_counter()
    try:
        return callback()
    finally:
        _record_active_timing(name, time.perf_counter() - started_at)


_COS_UPLOAD_RETRIES = 5


_COS_UPLOAD_RETRY_DELAY_S = 5


_NORMALIZED_COORD_SCALE = 1000.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _prepare_region_for_crop(
    region: object,
    image_size: tuple[int, int],
    *,
    coordinate_order: str | None = None,
) -> object:
    """Convert model-style normalized coordinates into absolute crop coordinates."""

    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return region

    try:
        coords = [float(value) for value in region]
    except (TypeError, ValueError):
        return region

    # #### START Response 0720 ####
    effective_order = (coordinate_order or "").strip().lower()
    if not effective_order:
        effective_order = "yxyx" if _env_flag("REVERSE_IMAGE_CROP_COORDS") else "xyxy"
    if effective_order == "yxyx":
        coords = [coords[1], coords[0], coords[3], coords[2]]
    # #### END Response 0720 ####

    image_width, image_height = image_size
    x1 = int(round(coords[0] / _NORMALIZED_COORD_SCALE * image_width))
    y1 = int(round(coords[1] / _NORMALIZED_COORD_SCALE * image_height))
    x2 = int(round(coords[2] / _NORMALIZED_COORD_SCALE * image_width))
    y2 = int(round(coords[3] / _NORMALIZED_COORD_SCALE * image_height))

    x1 = min(max(x1, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    x2 = min(max(x2, 0), image_width)
    y2 = min(max(y2, 0), image_height)
    return [x1, y1, x2, y2]


@dataclass(slots=True)
class ToolRuntimeContext:
    """Per-session runtime state used by the tool dispatcher."""

    working_dir: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    image_registry: dict[str, Any] = field(default_factory=dict)
    url_registry: dict[str, tools.UrlResource] = field(default_factory=dict)
    resource_registry: dict[str, tools.UrlResource] = field(default_factory=dict)
    filename_prefix: str = "sft"
    case_id: str = "sft_session"
    visual_lookup: Callable[..., object] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _image_counter: int = 0

    def __post_init__(self) -> None:
        self.working_dir = os.path.abspath(self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        if self.image_registry:
            for key in self.image_registry:
                if key.startswith("img_"):
                    try:
                        self._image_counter = max(self._image_counter, int(key.split("_", 1)[1]))
                    except (TypeError, ValueError):
                        continue

    @property
    def intermediate_dir(self) -> str:
        path = os.path.join(self.working_dir, "artifacts")
        os.makedirs(path, exist_ok=True)
        return path

    def next_image_id(self) -> str:
        self._image_counter += 1
        return f"img_{self._image_counter}"

    def register_image(self, payload: Any) -> str:
        image_id = self.next_image_id()
        self.image_registry[image_id] = payload
        return image_id

    def latest_image_reference(self) -> str | None:
        if not self.image_registry:
            return None
        return next(reversed(self.image_registry))

    def image_summary(self) -> str:
        if not self.image_registry:
            return ""
        lines = ["Available image refs:"]
        for image_id, payload in self.image_registry.items():
            lines.append(f"- {image_id}: {str(payload)[:120]}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_resource_url(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            parsed = urllib.parse.urlsplit(raw)
        except Exception:
            return raw
        if parsed.scheme not in {"http", "https"}:
            return raw
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
        netloc = hostname
        if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        return urllib.parse.urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))

    def register_url_resource(self, resource: tools.UrlResource) -> None:
        if resource.resource_id:
            existing = self.resource_registry.get(resource.resource_id)
            if existing is not None and existing.primary_url != resource.primary_url:
                raise ValueError(f"Resource ID collision: {resource.resource_id}")
            self.resource_registry[resource.resource_id] = resource
        for candidate in (
            resource.primary_url,
            resource.image_url,
            resource.thumbnail_url,
            resource.source_page_url,
        ):
            key = self._normalize_resource_url(candidate)
            if key:
                self.url_registry[key] = resource

    def resolve_url_resource(self, url: str) -> tools.UrlResource | None:
        return self.url_registry.get(self._normalize_resource_url(url))

    def resolve_resource_id(self, resource_id: str) -> tools.UrlResource | None:
        return self.resource_registry.get(str(resource_id or "").strip())

    def register_search_output(
        self,
        tool_name: str,
        output: dict[str, Any],
        *,
        timings: dict[str, float] | None = None,
    ) -> None:
        _compact, resources = tools.postprocess_search_output(
            tool_name=tool_name,
            output=output,
            timings=timings,
        )
        registration_started_at = time.monotonic()
        for resource in resources:
            self.register_url_resource(resource)
        if timings is not None:
            timings["resource_registration_s"] = time.monotonic() - registration_started_at

    def postprocess_search_output(
        self,
        tool_name: str,
        output: dict[str, Any],
        *,
        timings: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        compact, resources = tools.postprocess_search_output(
            tool_name=tool_name,
            output=output,
            timings=timings,
        )
        registration_started_at = time.monotonic()
        for resource in resources:
            self.register_url_resource(resource)
        if timings is not None:
            timings["resource_registration_s"] = time.monotonic() - registration_started_at
        return compact


@dataclass(slots=True)
class ToolExecutionResult:
    """Structured result for one tool invocation."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    output_text: str
    new_images: dict[str, Any] = field(default_factory=dict)
    # Private scalar telemetry. It is separate from ``output`` so timing
    # metadata never enters the model-visible tool observation.
    timing: dict[str, Any] = field(default_factory=dict)


_DEFAULT_FAILURE_JSONL_PATH = Path("synthesis/ignore/failure.jsonl")


_FAILURE_JSONL_ENV = "SFT_FAILURE_JSONL"


_TOOL_FAILURE_RECORDS: list[dict[str, Any]] = []


_TOOL_FAILURE_LOCK = threading.Lock()


def get_tool_failure_records() -> list[dict[str, Any]]:
    """Return a snapshot of tool failures recorded in this process."""
    with _TOOL_FAILURE_LOCK:
        return deepcopy(_TOOL_FAILURE_RECORDS)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _failure_jsonl_path() -> Path:
    configured = str(os.environ.get(_FAILURE_JSONL_ENV) or "").strip()
    return Path(configured) if configured else _DEFAULT_FAILURE_JSONL_PATH


def _record_tool_failure(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    error_output: Any,
) -> None:
    """Keep an in-memory failure list and append one durable JSONL record."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "arguments": _json_safe(arguments),
        "error_output": _json_safe(error_output),
    }
    with _TOOL_FAILURE_LOCK:
        _TOOL_FAILURE_RECORDS.append(record)
        try:
            path = _failure_jsonl_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, default=str)
                handle.write("\n")
                handle.flush()
        except Exception:
            # Failure logging must not turn the original tool failure into a
            # separate dispatcher failure.
            logger.exception("Unable to persist tool failure to %s", _failure_jsonl_path())


def _normalize_region_bbox(region: object) -> tuple[tuple[int, int, int, int] | None, str | None]:
    if region in (None, ""):
        return None, None
    if isinstance(region, dict):
        if not all(key in region for key in ("x", "y", "width", "height")):
            return None, "Region dict must contain x, y, width, and height."
        try:
            x = int(region["x"])
            y = int(region["y"])
            width = int(region["width"])
            height = int(region["height"])
        except (TypeError, ValueError):
            return None, "Region dict values must be numeric."
        if width <= 0 or height <= 0:
            return None, "Region width and height must be positive."
        return (x, y, width, height), None
    if isinstance(region, (list, tuple)):
        if len(region) != 4:
            return None, "Region list must contain exactly 4 numbers."
        try:
            x1 = int(region[0])
            y1 = int(region[1])
            x2 = int(region[2])
            y2 = int(region[3])
        except (TypeError, ValueError):
            return None, "Region list values must be numeric."
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None, "Region list must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1."
        return (x1, y1, width, height), None
    return None, "Region must be a 4-number list or a dict with x/y/width/height."


def _decode_data_url(data_url: str) -> bytes:
    payload = data_url.split("base64,", 1)[-1]
    return base64.b64decode(payload)


def _encode_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"


def _ensure_png_compatible_image(image: Image.Image) -> Image.Image:
    """Convert PIL modes unsupported by PNG to RGB/RGBA before serialization."""
    if image.mode in {"RGB", "RGBA", "L", "LA"}:
        return image
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB")


def _resize_image_for_i2i_upload(
    image: Image.Image,
    max_long_edge: int = I2I_UPLOAD_MAX_LONG_EDGE,
) -> Image.Image:
    """Downscale an i2i upload while preserving aspect ratio.

    Images at or below the limit are returned unchanged.  A new image is
    created only when downscaling is necessary, so the caller's image is not
    mutated.
    """

    max_long_edge = int(max_long_edge)
    if max_long_edge <= 0:
        raise ValueError("max_long_edge must be positive")

    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image

    scale = max_long_edge / long_edge
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    return image.resize(resized_size, resample=Image.Resampling.LANCZOS)


def _normalize_image_data_url(data: bytes) -> str:
    """Validate arbitrary image bytes and serialize one API-supported PNG.

    Search and reader backends can return AVIF/SVG/TIFF files, HTML error
    pages with an image-like extension, or content whose extension disagrees
    with its bytes.  Azure validates the actual bytes behind a data URL, so a
    guessed MIME type is insufficient.  Decoding and re-encoding here makes
    every accepted local/data/remote image a valid supported input.
    """

    try:
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            image = ImageOps.exif_transpose(decoded)
            image = _ensure_png_compatible_image(image)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
    except Exception as exc:
        raise ValueError("Image input is not a decodable raster image.") from exc
    return _encode_data_url(output.getvalue(), "image/png")


def _resolve_image_payload(source: Any, context: ToolRuntimeContext) -> Any:
    if isinstance(source, str) and source in context.image_registry:
        return context.image_registry[source]
    return source


def _i2i_empty_debug_context(
    *,
    image_source: Any,
    region: Any,
    cropped_image_oss_url: str | None,
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    """Return non-model-visible provenance for a persistent Lens empty result."""

    resolved_source = _resolve_image_payload(image_source, context)
    original_url = (
        resolved_source
        if isinstance(resolved_source, str) and resolved_source.startswith(("http://", "https://"))
        else None
    )
    reference = image_source if isinstance(image_source, str) else None
    return {
        "original_image_url": original_url,
        "original_image_reference": reference,
        "region": _json_safe(region),
        "cropped_image_oss_url": cropped_image_oss_url,
    }


def _image_source_to_model_url(source: Any, context: ToolRuntimeContext) -> str:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, str):
        if payload.startswith("data:image"):
            return _normalize_image_data_url(_decode_data_url(payload))
        if payload.startswith(("http://", "https://")):
            response = requests.get(payload, timeout=60)
            response.raise_for_status()
            return _normalize_image_data_url(response.content)
        if os.path.exists(payload):
            with open(payload, "rb") as handle:
                return _normalize_image_data_url(handle.read())
    if isinstance(payload, bytes):
        return _normalize_image_data_url(payload)
    if isinstance(payload, Image.Image):
        buffer = io.BytesIO()
        _ensure_png_compatible_image(payload).save(buffer, format="PNG")
        return _normalize_image_data_url(buffer.getvalue())
    raise ValueError(f"Unsupported image source for model input: {type(payload)!r}")


def _load_pil_image(source: Any, context: ToolRuntimeContext) -> Image.Image:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, Image.Image):
        return payload.copy()
    if isinstance(payload, bytes):
        return Image.open(io.BytesIO(payload))
    if isinstance(payload, str):
        if payload.startswith("data:image"):
            return Image.open(io.BytesIO(_decode_data_url(payload)))
        if payload.startswith(("http://", "https://")):
            response = requests.get(payload, timeout=60)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        if os.path.exists(payload):
            return Image.open(payload)
    raise ValueError(f"Unsupported image source: {type(payload)!r}")


def _persist_pil_image(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> tuple[str, str]:
    image_id = context.next_image_id()
    filename = f"{context.filename_prefix}_{context.session_id}_{tool_name}_{image_id}.png"
    save_path = os.path.join(context.intermediate_dir, filename)
    _ensure_png_compatible_image(image).save(save_path, format="PNG")
    context.image_registry[image_id] = save_path
    return image_id, save_path


def _persist_pil_image_to_cache(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> tuple[str, str]:
    image_id = context.next_image_id()
    cache_dir = Path(__file__).resolve().parents[1] / ".image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{context.filename_prefix}_{context.session_id}_{tool_name}_{image_id}.png"
    save_path = str(cache_dir / filename)
    _ensure_png_compatible_image(image).save(save_path, format="PNG")
    context.image_registry[image_id] = save_path
    return image_id, save_path


def _try_upload_pil_image(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
    *,
    max_long_edge: int | None = None,
) -> str | None:
    image = _ensure_png_compatible_image(image)
    if max_long_edge is not None:
        image = _resize_image_for_i2i_upload(image, max_long_edge)
    try:
        from opensearch_vl.opensearch_infer import cos_upload
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("COS uploader import failed: %s", exc)
        return None

    for attempt in range(1, _COS_UPLOAD_RETRIES + 2):
        try:
            uploaded_url = cos_upload.upload_pil_image(
                image,
                context.filename_prefix,
                0,
                0,
                tool_name,
            )
            if uploaded_url:
                return uploaded_url
            error = "uploader returned an empty URL"
        except Exception as exc:  # pragma: no cover - optional dependency
            error = exc
        if attempt >= _COS_UPLOAD_RETRIES + 1:
            logger.warning(
                "COS upload failed for %s after %d attempt(s): %s",
                tool_name,
                attempt,
                error,
            )
            break
        tools._tool_retry_debug(
            "cos_upload",
            attempt=attempt,
            max_attempts=_COS_UPLOAD_RETRIES + 1,
            sleep_seconds=_COS_UPLOAD_RETRY_DELAY_S * attempt,
            error=error,
            tool_name=tool_name,
        )
        logger.warning(
            "COS upload failed for %s (attempt %d/%d); retrying in %ds: %s",
            tool_name,
            attempt,
            _COS_UPLOAD_RETRIES + 1,
            _COS_UPLOAD_RETRY_DELAY_S * attempt,
            error,
        )
        tracked_sleep(
            _COS_UPLOAD_RETRY_DELAY_S * attempt,
            reason="cos_upload_error",
            tool="cos_upload",
            error=error,
            error_type=(type(error).__name__ if isinstance(error, BaseException) else "EmptyUploadURL"),
            fallback_sleep=time.sleep,
        )
    return None


def _materialize_remote_image_url(source: Any, context: ToolRuntimeContext, tool_name: str) -> tuple[str | None, str | None]:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, str) and payload.startswith(("http://", "https://")):
        return payload, None

    try:
        image = _load_pil_image(payload, context)
    except Exception as exc:
        return None, f"Failed to load image for {tool_name}: {exc}"

    uploaded_url = _try_upload_pil_image(
        image,
        context,
        tool_name,
        max_long_edge=(I2I_UPLOAD_MAX_LONG_EDGE if tool_name.startswith("i2i") else None),
    )
    if uploaded_url:
        return uploaded_url, None
    return None, (
        f"{tool_name} requires a publicly reachable image URL. "
        "Uploading the local image failed; configure the optional COS uploader first."
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _read_url_page_id(
    *,
    resource_id: str,
    resource: tools.UrlResource | None,
    url: str,
    context: ToolRuntimeContext,
) -> str:
    """Return a stable opaque page ID, registering direct legacy URLs privately."""

    page_id = resource_id or (resource.resource_id if resource is not None else "")
    if not page_id:
        page_id = tools._resource_id("page", url)
        context.register_url_resource(
            tools.UrlResource(primary_url=url, resource_id=page_id, kind="page")
        )
    return page_id


def _public_read_url_text_observation(
    *,
    output: dict[str, Any],
    page_id: str,
    goal: str,
) -> dict[str, str]:
    """Expose text-page reads without URL/backend metadata to the model.

    The underlying reader result retains provenance for runtime diagnostics, but
    tool observations must not reveal raw URLs or Firecrawl metadata.  Keep the
    public schema identical for Enhanced Reader, Firecrawl, PDFs, and failures.
    """

    if output.get("ok"):
        content = str(output.get("content") or "")
        title = str(output.get("title") or "")
    else:
        # Reader errors commonly interpolate the requested URL (and sometimes
        # redirect URLs).  Do not forward any such diagnostic text to the model.
        content = "Unable to read the requested page."
        title = ""
    return {
        "page_id": page_id,
        "title": title,
        "goal": goal,
        "content": content,
    }


def _call_with_optional_cache(
    tool_cache: Any | None,
    tool_name: str,
    arguments: dict[str, Any],
    compute: Callable[[], Any],
    timings: dict[str, float] | None = None,
) -> Any:
    """Use an explicitly injected cache without adding shared cache state."""

    cache_started_at = time.monotonic()
    compute_started_at: list[float] = []

    def timed_compute() -> Any:
        compute_started_at.append(time.monotonic())
        try:
            return compute()
        finally:
            if timings is not None:
                timings["search_tool_body_s"] = time.monotonic() - compute_started_at[0]

    if tool_cache is None:
        result = timed_compute()
        if timings is not None:
            timings["cache_enabled"] = 0.0
    else:
        result = tool_cache.get_or_compute(tool_name, arguments, timed_compute)
        if timings is not None:
            timings["cache_enabled"] = 1.0

    if timings is not None:
        cache_or_compute_s = time.monotonic() - cache_started_at
        timings["cache_or_compute_s"] = cache_or_compute_s
        timings["cache_hit"] = 1.0 if not compute_started_at else 0.0
        body_s = float(timings.get("search_tool_body_s", 0.0) or 0.0)
        timings["cache_overhead_s"] = max(0.0, cache_or_compute_s - body_s)
    return result


def _execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    context: ToolRuntimeContext,
    question_text: str = "",
    assistant_text: str = "",
    tool_goal: str = "",
    tool_cache: Any | None = None,
) -> ToolExecutionResult:
    """Execute one tool call against the runtime context."""

    params = tools.normalize_tool_arguments(name, arguments)
    if tool_cache is not None:
        set_stats_callback = getattr(tool_cache, "set_stats_callback", None)
        stats_callback = getattr(tools, "_record_tool_cache_event", None)
        if callable(set_stats_callback) and callable(stats_callback):
            set_stats_callback(stats_callback)

    if name == "t2t_search":
        search_timing: dict[str, float] = {}
        query = params.get("query") or params.get("q") or ""
        if not query:
            output = {"ok": False, "error": "query is required for t2t_search"}
        else:
            search_arguments = {
                "query": query,
                "lang": params.get("lang") or params.get("hl") or "en",
                "top_k": int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
            }
            output = _call_with_optional_cache(
                tool_cache,
                name,
                search_arguments,
                lambda: tools.t2t_search(**search_arguments, timings=search_timing),
                timings=search_timing,
            )
        if isinstance(output, dict) and output.get("ok"):
            postprocess_started_at = time.monotonic()
            output = context.postprocess_search_output(
                "t2t_search",
                output,
                timings=search_timing,
            )
            search_timing["search_postprocess_s"] = time.monotonic() - postprocess_started_at
        serialization_started_at = time.monotonic()
        output_text = _json_text(output)
        search_timing["result_serialization_s"] = time.monotonic() - serialization_started_at
        return ToolExecutionResult(
            name=name,
            arguments=params,
            output=output,
            output_text=output_text,
            timing=search_timing,
        )

    if name == "t2i_search":
        search_timing = {}
        query = params.get("query") or params.get("q") or ""
        if not query:
            output = {"ok": False, "error": "query is required for t2i_search"}
        else:
            search_arguments = {
                "query": query,
                "lang": params.get("lang") or params.get("hl") or "en",
                "top_k": int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
            }
            output = _call_with_optional_cache(
                tool_cache,
                name,
                search_arguments,
                lambda: tools.t2i_search(**search_arguments, timings=search_timing),
                timings=search_timing,
            )
        if isinstance(output, dict) and output.get("ok"):
            postprocess_started_at = time.monotonic()
            output = context.postprocess_search_output(
                "t2i_search",
                output,
                timings=search_timing,
            )
            search_timing["search_postprocess_s"] = time.monotonic() - postprocess_started_at
        serialization_started_at = time.monotonic()
        output_text = _json_text(output)
        search_timing["result_serialization_s"] = time.monotonic() - serialization_started_at
        return ToolExecutionResult(
            name=name,
            arguments=params,
            output=output,
            output_text=output_text,
            timing=search_timing,
        )

    if name == "read_url":
        resource_id = str(params.get("resource_id") or "").strip()
        direct_url = str(params.get("url") or params.get("URL") or "").strip()
        # Older/model-generated calls occasionally put a compact search
        # resource ID in ``url``. Treat it as an ID before the low-level URL
        # normalizer can turn it into ``https://image_.../``.
        if not resource_id and tools.is_compact_resource_id(direct_url):
            resource_id = direct_url
            direct_url = ""
        resource = context.resolve_resource_id(resource_id) if resource_id else None
        effective_goal = str(params.get("goal") or tool_goal or "").strip()
        if resource_id and resource is None and not direct_url:
            failure_output = {
                "ok": False,
                "error": f"resource_id not found: {resource_id}",
            }
            _record_tool_failure(
                tool_name=name,
                arguments=params,
                error_output=failure_output,
            )
            output = _public_read_url_text_observation(
                output=failure_output,
                page_id=resource_id,
                goal=effective_goal,
            )
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        url = resource.primary_url if resource is not None else direct_url
        if not url:
            failure_output = {"ok": False, "error": "URL is required."}
            _record_tool_failure(
                tool_name=name,
                arguments=params,
                error_output=failure_output,
            )
            output = _public_read_url_text_observation(
                output=failure_output,
                page_id=resource_id,
                goal=effective_goal,
            )
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        # print(
        #     "[read_url debug] "
        #     f"resource_id={resource_id or '-'} "
        #     f"url={url}",
        #     file=sys.stderr,
        #     flush=True,
        # )
        output = tools.read_url(
            url=url,
            goal=effective_goal,
            assistant_output=assistant_text,
            resource=resource or context.resolve_url_resource(url),
            raw_document_fetcher=(
                tool_cache.fetch_document if tool_cache is not None else None
            ),
        )
        read_url_timing: dict[str, Any] = {}
        if isinstance(output, dict):
            private_timing = output.pop("_timing", None)
            if isinstance(private_timing, dict):
                read_url_timing = private_timing
        if isinstance(output, dict) and output.get("ok") is False:
            _record_tool_failure(
                tool_name=name,
                arguments=params,
                error_output=output,
            )
        new_images: dict[str, Any] = {}
        if output.get("ok") and output.get("local_path"):
            image_id = context.register_image(output["local_path"])
            new_images[image_id] = output["local_path"]
            output = {"image_id": image_id}
        else:
            page_id = _read_url_page_id(
                resource_id=resource_id,
                resource=resource,
                url=url,
                context=context,
            )
            output = _public_read_url_text_observation(
                output=output,
                page_id=page_id,
                goal=effective_goal,
            )
        return ToolExecutionResult(
            name=name,
            arguments=params,
            output=output,
            output_text=_json_text(output),
            new_images=new_images,
            timing=read_url_timing,
        )

    if name == "i2i_search":
        search_timing: dict[str, float] = {}
        image_source = params.get("image") or params.get("url") or context.latest_image_reference() or ""
        requested_region = params.get("region")
        if not image_source:
            output = {
                "ok": False,
                "error": "No image is available in the current context for i2i_search.",
            }
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

        region = params.get("region")
        new_images: dict[str, Any] = {}
        if region not in (None, ""):
            image = _load_pil_image(image_source, context)
            # #### START Response 0720 ####
            region = _prepare_region_for_crop(
                region,
                image.size,
                coordinate_order=str((context.metadata or {}).get("bbox_coordinate_order") or ""),
            )
            # #### END Response 0720 ####
            bbox, err = _normalize_region_bbox(region)
            if err:
                output = {"ok": False, "error": err}
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
            assert bbox is not None
            x, y, width, height = bbox
            cropped = image.crop((x, y, x + width, y + height))
            # The COS uploader derives its object key from ``tool_name``.  Keep
            # each crop immutable so a previous Lens request never observes a
            # later crop written to the same public URL.
            uploaded_url = _try_upload_pil_image(
                cropped,
                context,
                f"i2i_region_{uuid.uuid4().hex}",
                max_long_edge=I2I_UPLOAD_MAX_LONG_EDGE,
            )
            if not uploaded_url:
                cropped_id, cropped_path = _persist_pil_image_to_cache(cropped, context, "i2i_region")
                new_images[cropped_id] = cropped_path
                output = {
                    "ok": False,
                    "error": (
                        "Cropped region was created, but reverse image search needs a public URL. "
                        "Uploading the cropped image failed, so it was saved to the local image cache instead."
                    ),
                    "cropped_image_id": cropped_id,
                    "cropped_image_path": cropped_path,
                }
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)
            context.register_image(uploaded_url)
            output = tools.i2i_search(
                image_url=uploaded_url,
                visual_lookup=context.visual_lookup,
                **({"private_result_limit": context.metadata["i2i_private_result_limit"]}
                   if "i2i_private_result_limit" in context.metadata else {}),
                top_k=int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
                timings=search_timing,
                debug_context=_i2i_empty_debug_context(
                    image_source=image_source,
                    region=requested_region,
                    cropped_image_oss_url=uploaded_url,
                    context=context,
                ),
            )
            output = dict(output)
            output["cropped_image_url"] = uploaded_url
            if output.get("ok"):
                postprocess_started_at = time.monotonic()
                output = context.postprocess_search_output(
                    "i2i_search",
                    output,
                    timings=search_timing,
                )
                search_timing["search_postprocess_s"] = time.monotonic() - postprocess_started_at
            serialization_started_at = time.monotonic()
            output_text = _json_text(output)
            search_timing["result_serialization_s"] = time.monotonic() - serialization_started_at
            return ToolExecutionResult(
                name=name,
                arguments=params,
                output=output,
                output_text=output_text,
                new_images=new_images,
                timing=search_timing,
            )

        remote_url, err = _materialize_remote_image_url(image_source, context, "i2i_search")
        if err:
            output = {"ok": False, "error": err}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        output = tools.i2i_search(
            image_url=remote_url or "",
            visual_lookup=context.visual_lookup,
            **({"private_result_limit": context.metadata["i2i_private_result_limit"]}
               if "i2i_private_result_limit" in context.metadata else {}),
            top_k=int(params.get("top_k", tools.DEFAULT_SEARCH_TOP_K)),
            timings=search_timing,
            debug_context=_i2i_empty_debug_context(
                image_source=image_source,
                region=requested_region,
                cropped_image_oss_url=None,
                context=context,
            ),
        )
        if isinstance(output, dict) and output.get("ok"):
            postprocess_started_at = time.monotonic()
            output = context.postprocess_search_output(
                "i2i_search",
                output,
                timings=search_timing,
            )
            search_timing["search_postprocess_s"] = time.monotonic() - postprocess_started_at
        serialization_started_at = time.monotonic()
        output_text = _json_text(output)
        search_timing["result_serialization_s"] = time.monotonic() - serialization_started_at
        return ToolExecutionResult(
            name=name,
            arguments=params,
            output=output,
            output_text=output_text,
            timing=search_timing,
        )

    output = {"ok": False, "error": f"Unknown tool: {name}"}
    return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))


def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    context: ToolRuntimeContext,
    question_text: str = "",
    assistant_text: str = "",
    tool_goal: str = "",
    tool_cache: Any | None = None,
) -> ToolExecutionResult:
    """Execute one tool and record failures before returning or re-raising."""
    dispatcher_started_at = time.perf_counter()
    dispatcher_core_elapsed_s = 0.0
    dispatcher_failure_record_elapsed_s = 0.0
    try:
        dispatcher_core_started_at = time.perf_counter()
        result = _timed_call(
            f"tool:{name}",
            lambda: _execute_tool_call(
                name=name,
                arguments=arguments,
                context=context,
                question_text=question_text,
                assistant_text=assistant_text,
                tool_goal=tool_goal,
                tool_cache=tool_cache,
            ),
        )
        dispatcher_core_elapsed_s = time.perf_counter() - dispatcher_core_started_at
    except Exception as exc:
        dispatcher_core_elapsed_s = max(
            0.0,
            time.perf_counter() - locals().get("dispatcher_core_started_at", dispatcher_started_at),
        )
        _record_tool_failure(
            tool_name=name,
            arguments=arguments,
            error_output={
                "ok": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            },
        )
        raise

    # Keep dispatcher-level timings separate from the tool-specific timings
    # above.  In particular, this lets the RL caller distinguish time spent
    # inside this synchronous dispatcher from time spent waiting for the
    # asyncio event loop to resume after the executor returned.
    result.timing = dict(result.timing or {})
    result.timing["dispatcher_core_s"] = max(0.0, dispatcher_core_elapsed_s)
    if isinstance(result.output, dict) and result.output.get("ok") is False:
        failure_record_started_at = time.perf_counter()
        _record_tool_failure(
            tool_name=name,
            arguments=result.arguments,
            error_output=result.output,
        )
        dispatcher_failure_record_elapsed_s = time.perf_counter() - failure_record_started_at
    result.timing["dispatcher_failure_record_s"] = max(
        0.0,
        dispatcher_failure_record_elapsed_s,
    )
    result.timing["dispatcher_total_s"] = max(
        0.0,
        time.perf_counter() - dispatcher_started_at,
    )
    return result
