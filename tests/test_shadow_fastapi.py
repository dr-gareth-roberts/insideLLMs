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


@pytest.mark.parametrize("outcome", ["success", "http_error", "exception"])
def test_shadow_redacts_query_in_both_persisted_locations(tmp_path: Path, outcome: str) -> None:
    from urllib.parse import parse_qs

    query = "page=2&access_token=oauth-example&X-Amz-Signature=signature-example&X-Amz-Credential=credential-example&access_token=second-example&empty="
    request = _DummyRequest(query=query)
    path = tmp_path / "records.jsonl"
    middleware = fastapi(output_path=path, sample_rate=1.0)
    response = _DummyResponse(status_code=500 if outcome == "http_error" else 200)
    app_error = RuntimeError("upstream failed")

    async def call_next(received: _DummyRequest) -> _DummyResponse:
        assert received.url.query == query
        if outcome == "exception":
            raise app_error
        return response

    if outcome == "exception":
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(middleware(request, call_next))
        assert exc.value is app_error
    else:
        assert asyncio.run(middleware(request, call_next)) is response
    serialized = path.read_text()
    for secret in ("oauth-example", "signature-example", "credential-example", "second-example"):
        assert secret not in serialized
    record = _load_jsonl(path)[0]
    for stored in (record["input"]["query"], record["custom"]["http"]["query"]):
        values = parse_qs(stored, keep_blank_values=True)
        assert values["page"] == ["2"]
        assert values["empty"] == [""]
        assert values["access_token"] == ["[REDACTED]", "[REDACTED]"]
        assert values["X-Amz-Signature"] == ["[REDACTED]"]
        assert values["X-Amz-Credential"] == ["[REDACTED]"]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("q=hello%20world&tag=a&tag=b&empty=", "q=hello%20world&tag=a&tag=b&empty="),
        ("page=2#acces%73_token=fragment-example", "page=2#access_token=%5BREDACTED%5D"),
        ("access%5Ftoken=encoded-example", "access_token=%5BREDACTED%5D"),
    ],
)
def test_shadow_query_redaction_handles_encoded_keys_and_preserves_plain_queries(
    tmp_path: Path,
    query: str,
    expected: str,
) -> None:
    middleware = fastapi(output_path=tmp_path / "records.jsonl", sample_rate=1.0)

    async def call_next(request: _DummyRequest) -> _DummyResponse:
        assert request.url.query == query
        return _DummyResponse()

    asyncio.run(middleware(_DummyRequest(query=query), call_next))
    record = _load_jsonl(tmp_path / "records.jsonl")[0]
    assert record["input"]["query"] == expected
    assert record["custom"]["http"]["query"] == expected
