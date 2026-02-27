"""Tests for production shadow capture middleware helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.schemas import DEFAULT_SCHEMA_VERSION
from insideLLMs.shadow import fastapi


class _DummyURL:
    def __init__(self, path: str, query: str = "") -> None:
        self.path = path
        self.query = query


class _DummyRequest:
    def __init__(
        self,
        *,
        method: str = "POST",
        path: str = "/v1/chat/completions",
        query: str = "",
        body: bytes = b'{"prompt":"hello"}',
        headers: dict[str, str] | None = None,
    ) -> None:
        self.method = method
        self.url = _DummyURL(path, query)
        self._body = body
        self.headers = headers or {"content-type": "application/json"}

    async def body(self) -> bytes:
        return self._body


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/json"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_shadow_fastapi_writes_result_record_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "records.jsonl"
    middleware = fastapi(
        output_path=output_path,
        sample_rate=1.0,
        run_id="shadow-run",
        include_request_headers=True,
    )
    request = _DummyRequest()
    response = _DummyResponse(status_code=200)

    async def _call_next(_request: _DummyRequest) -> _DummyResponse:
        return response

    returned = asyncio.run(middleware(request, _call_next))
    assert returned is response

    payload = _load_jsonl(output_path)
    assert len(payload) == 1
    record = payload[0]
    assert record["schema_version"] == DEFAULT_SCHEMA_VERSION
    assert record["run_id"] == "shadow-run"
    assert record["status"] == "success"
    assert record["probe"]["probe_id"] == "shadow_capture"
    assert record["dataset"]["provenance"] == "shadow.fastapi"
    assert record["input"]["path"] == "/v1/chat/completions"
    assert record["input"]["headers"]["content-type"] == "application/json"


def test_shadow_fastapi_respects_zero_sample_rate(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "records.jsonl"
    middleware = fastapi(output_path=output_path, sample_rate=0.0)
    request = _DummyRequest()
    response = _DummyResponse(status_code=200)

    async def _call_next(_request: _DummyRequest) -> _DummyResponse:
        return response

    asyncio.run(middleware(request, _call_next))
    assert _load_jsonl(output_path) == []


def test_shadow_fastapi_logs_error_records_before_reraising(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow" / "records.jsonl"
    middleware = fastapi(output_path=output_path, sample_rate=1.0, run_id="shadow-errors")
    request = _DummyRequest()

    async def _call_next(_request: _DummyRequest) -> _DummyResponse:
        raise RuntimeError("upstream failure")

    with pytest.raises(RuntimeError, match="upstream failure"):
        asyncio.run(middleware(request, _call_next))

    payload = _load_jsonl(output_path)
    assert len(payload) == 1
    record = payload[0]
    assert record["run_id"] == "shadow-errors"
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"

