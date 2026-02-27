"""Production traffic shadow capture helpers.

This module provides a lightweight FastAPI-compatible middleware factory that
captures sampled request/response metadata into canonical ``records.jsonl``
entries compatible with the deterministic harness/diff workflow.

Example
-------
```python
from fastapi import FastAPI
from insideLLMs import shadow

app = FastAPI()
app.middleware("http")(
    shadow.fastapi(output_path="./shadow/records.jsonl", sample_rate=0.01)
)
```
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from insideLLMs._serialization import stable_json_dumps
from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION

_MIDDLEWARE_PROBE_ID = "shadow_capture"
_MIDDLEWARE_PROBE_VERSION = "1.0.0"


@dataclass
class ShadowWriter:
    """Append-only writer for canonical shadow capture JSONL records."""

    output_path: Path
    strict_serialization: bool = False
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one JSON object as a stable JSONL line."""
        line = stable_json_dumps(dict(record), strict=self.strict_serialization)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sample_request(*, sample_rate: float, method: str, path: str, query: str, body: bytes) -> bool:
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    seed = (
        method.encode("utf-8")
        + b"\n"
        + path.encode("utf-8")
        + b"\n"
        + query.encode("utf-8")
        + b"\n"
        + body[:4096]
    )
    digest = hashlib.sha256(seed).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return bucket < sample_rate


def _decode_request_body(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        try:
            return raw.decode("utf-8")
        except Exception:
            return {"bytes": len(raw)}


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _request_url_parts(request: Any) -> tuple[str, str]:
    url = getattr(request, "url", None)
    if url is None:
        return "/", ""
    path = getattr(url, "path", "/")
    query = getattr(url, "query", "")
    return str(path), str(query)


async def _read_request_body(request: Any) -> bytes:
    body_fn = getattr(request, "body", None)
    if not callable(body_fn):
        return b""
    value = await body_fn()
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return b""


def _record_example_id(*, method: str, path: str, query: str, started_at: str) -> str:
    digest = hashlib.sha256(f"{method}|{path}|{query}|{started_at}".encode("utf-8")).hexdigest()
    return digest[:24]


def fastapi(
    *,
    output_path: str | Path = "records.jsonl",
    sample_rate: float = 0.01,
    run_id: str | None = None,
    model_id: str = "production-shadow",
    model_provider: str = "insideLLMs",
    probe_id: str = _MIDDLEWARE_PROBE_ID,
    dataset_id: str = "production-traffic",
    include_request_headers: bool = False,
    strict_serialization: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> Callable[[Any, Callable[[Any], Awaitable[Any]]], Awaitable[Any]]:
    """Create a FastAPI-compatible middleware that shadows sampled traffic.

    The returned callable is compatible with ``app.middleware("http")``.
    Captured entries are emitted in ``ResultRecord``-compatible shape.
    """
    if sample_rate < 0 or sample_rate > 1:
        raise ValueError("sample_rate must be within [0, 1]")

    writer = ShadowWriter(Path(output_path), strict_serialization=strict_serialization)
    now = clock or (lambda: datetime.now(timezone.utc))
    effective_run_id = run_id or f"shadow-{uuid4().hex[:12]}"

    async def middleware(request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        started_dt = _to_utc(now())
        started_at = started_dt.isoformat()
        method = str(getattr(request, "method", "GET")).upper()
        path, query = _request_url_parts(request)
        request_headers = _safe_mapping(getattr(request, "headers", {}))
        request_body = await _read_request_body(request)
        should_capture = _sample_request(
            sample_rate=sample_rate,
            method=method,
            path=path,
            query=query,
            body=request_body,
        )

        input_payload: dict[str, Any] = {
            "method": method,
            "path": path,
            "query": query,
            "body": _decode_request_body(request_body),
        }
        if include_request_headers:
            input_payload["headers"] = request_headers

        try:
            response = await call_next(request)
        except Exception as exc:
            if should_capture:
                completed_dt = _to_utc(now())
                completed_at = completed_dt.isoformat()
                latency_ms = max((completed_dt - started_dt).total_seconds() * 1000.0, 0.0)
                writer.append(
                    {
                        "schema_version": DEFAULT_SCHEMA_VERSION,
                        "run_id": effective_run_id,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "model": {
                            "model_id": model_id,
                            "provider": model_provider,
                            "params": {},
                        },
                        "probe": {
                            "probe_id": probe_id,
                            "probe_version": _MIDDLEWARE_PROBE_VERSION,
                            "params": {},
                        },
                        "example_id": _record_example_id(
                            method=method,
                            path=path,
                            query=query,
                            started_at=started_at,
                        ),
                        "dataset": {
                            "dataset_id": dataset_id,
                            "dataset_version": None,
                            "dataset_hash": None,
                            "provenance": "shadow.fastapi",
                            "params": {"sample_rate": sample_rate},
                        },
                        "input": input_payload,
                        "output": None,
                        "output_text": None,
                        "scores": {},
                        "primary_metric": None,
                        "usage": {},
                        "latency_ms": latency_ms,
                        "status": "error",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "custom": {
                            "source": "shadow.fastapi",
                            "http": {"method": method, "path": path, "query": query},
                            "sample_rate": sample_rate,
                        },
                    }
                )
            raise

        if not should_capture:
            return response

        completed_dt = _to_utc(now())
        completed_at = completed_dt.isoformat()
        latency_ms = max((completed_dt - started_dt).total_seconds() * 1000.0, 0.0)
        status_code = int(getattr(response, "status_code", 200))
        response_headers = _safe_mapping(getattr(response, "headers", {}))
        status = "error" if status_code >= 500 else "success"
        writer.append(
            {
                "schema_version": DEFAULT_SCHEMA_VERSION,
                "run_id": effective_run_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "model": {
                    "model_id": model_id,
                    "provider": model_provider,
                    "params": {},
                },
                "probe": {
                    "probe_id": probe_id,
                    "probe_version": _MIDDLEWARE_PROBE_VERSION,
                    "params": {},
                },
                "example_id": _record_example_id(
                    method=method,
                    path=path,
                    query=query,
                    started_at=started_at,
                ),
                "dataset": {
                    "dataset_id": dataset_id,
                    "dataset_version": None,
                    "dataset_hash": None,
                    "provenance": "shadow.fastapi",
                    "params": {"sample_rate": sample_rate},
                },
                "input": input_payload,
                "output": {
                    "status_code": status_code,
                    "content_type": response_headers.get("content-type"),
                    "content_length": response_headers.get("content-length"),
                },
                "output_text": None,
                "scores": {},
                "primary_metric": None,
                "usage": {},
                "latency_ms": latency_ms,
                "status": status,
                "error": f"HTTP {status_code}" if status == "error" else None,
                "error_type": "HTTPStatusError" if status == "error" else None,
                "custom": {
                    "source": "shadow.fastapi",
                    "http": {
                        "method": method,
                        "path": path,
                        "query": query,
                        "status_code": status_code,
                    },
                    "sample_rate": sample_rate,
                },
            }
        )
        return response

    return middleware


__all__ = ["ShadowWriter", "fastapi"]
