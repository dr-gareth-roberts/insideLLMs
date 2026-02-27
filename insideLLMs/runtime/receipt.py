"""Receipt middleware for verifiable per-call evidence (Ultimate mode).

Logs every generate/chat/stream with canonical request hash, response hash,
latency, and optional record_index/example_id. Writes to a JSONL sink
(run_dir/receipts/calls.jsonl). Must be at the top of the chain so all
calls (including cache hits) are recorded; cache hits get receipt with
cache_hit=true and response_hash of the cached response.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from insideLLMs.crypto.canonical import canonical_json_bytes, digest_bytes
from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import ChatMessage
from insideLLMs.runtime.pipeline import PassthroughMiddleware


def _request_hash(prompt: str, kwargs: dict[str, Any]) -> str:
    """Canonical hash of request (prompt + generation params)."""
    payload = {
        "prompt": prompt,
        "params": {
            k: v for k, v in sorted(kwargs.items()) if k not in ("record_index", "example_id")
        },
    }
    return digest_bytes(canonical_json_bytes(payload))


def _request_hash_chat(messages: list[Any], kwargs: dict[str, Any]) -> str:
    """Canonical hash of chat request (messages + params)."""
    msgs = [
        {
            "role": getattr(m, "role", m.get("role")),
            "content": getattr(m, "content", m.get("content")),
        }
        for m in messages
    ]
    payload = {
        "messages": msgs,
        "params": {
            k: v for k, v in sorted(kwargs.items()) if k not in ("record_index", "example_id")
        },
    }
    return digest_bytes(canonical_json_bytes(payload))


def _response_hash(text: str, usage: Optional[dict[str, Any]] = None) -> str:
    """Canonical hash of response (text + usage if present)."""
    payload = {"text": text, "usage": usage or {}}
    return digest_bytes(canonical_json_bytes(payload))


class ReceiptMiddleware(PassthroughMiddleware):
    """Middleware that appends a receipt (request/response hashes, latency) per call to a JSONL sink.

    Used in Ultimate mode to produce run_dir/receipts/calls.jsonl for
    receipts_merkle_root and attestation. When receipt_sink is None, no-op.
    """

    def __init__(self, receipt_sink: Optional[Path | str] = None) -> None:
        super().__init__()
        self._sink_path = Path(receipt_sink) if receipt_sink else None
        self._lock = threading.Lock()

    def _append_receipt(self, receipt: dict[str, Any]) -> None:
        if self._sink_path is None:
            return
        line = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self._sink_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    def process_generate(self, prompt: str, **kwargs: Any) -> str:
        start = time.perf_counter()
        request_hash = _request_hash(prompt, kwargs)
        if self.next_middleware:
            response = self.next_middleware.process_generate(prompt, **kwargs)
        elif self.model:
            response = self.model.generate(prompt, **kwargs)
        else:
            raise ModelError("No model available in pipeline")
        latency_ms = (time.perf_counter() - start) * 1000
        response_hash = _response_hash(response)
        receipt = {
            "request_hash": request_hash,
            "response_hash": response_hash,
            "latency_ms": round(latency_ms, 3),
            "record_index": kwargs.get("record_index"),
            "example_id": kwargs.get("example_id"),
            "cache_hit": kwargs.get("_receipt_cache_hit", False),
            "retry_count": kwargs.get("_receipt_retry_count"),
            "trace_id": kwargs.get("trace_id"),
        }
        self._append_receipt(receipt)
        return response

    def process_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        start = time.perf_counter()
        request_hash = _request_hash_chat(messages, kwargs)
        if self.next_middleware:
            response = self.next_middleware.process_chat(messages, **kwargs)
        elif self.model:
            response = self.model.chat(messages, **kwargs)
        else:
            raise ModelError("No model available in pipeline")
        latency_ms = (time.perf_counter() - start) * 1000
        response_hash = _response_hash(response)
        receipt = {
            "request_hash": request_hash,
            "response_hash": response_hash,
            "latency_ms": round(latency_ms, 3),
            "record_index": kwargs.get("record_index"),
            "example_id": kwargs.get("example_id"),
            "cache_hit": kwargs.get("_receipt_cache_hit", False),
            "retry_count": kwargs.get("_receipt_retry_count"),
            "trace_id": kwargs.get("trace_id"),
        }
        self._append_receipt(receipt)
        return response

    async def aprocess_generate(self, prompt: str, **kwargs: Any) -> str:
        start = time.perf_counter()
        request_hash = _request_hash(prompt, kwargs)
        if self.next_middleware:
            response = await self.next_middleware.aprocess_generate(prompt, **kwargs)
        elif self.model:
            if hasattr(self.model, "agenerate"):
                response = await self.model.agenerate(prompt, **kwargs)
            else:
                response = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.model.generate(prompt, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")
        latency_ms = (time.perf_counter() - start) * 1000
        response_hash = _response_hash(response)
        receipt = {
            "request_hash": request_hash,
            "response_hash": response_hash,
            "latency_ms": round(latency_ms, 3),
            "record_index": kwargs.get("record_index"),
            "example_id": kwargs.get("example_id"),
            "cache_hit": kwargs.get("_receipt_cache_hit", False),
            "retry_count": kwargs.get("_receipt_retry_count"),
            "trace_id": kwargs.get("trace_id"),
        }
        self._append_receipt(receipt)
        return response

    async def aprocess_chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        start = time.perf_counter()
        request_hash = _request_hash_chat(messages, kwargs)
        if self.next_middleware:
            response = await self.next_middleware.aprocess_chat(messages, **kwargs)
        elif self.model:
            if hasattr(self.model, "achat"):
                response = await self.model.achat(messages, **kwargs)
            else:
                response = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: self.model.chat(messages, **kwargs)
                )
        else:
            raise ModelError("No model available in pipeline")
        latency_ms = (time.perf_counter() - start) * 1000
        response_hash = _response_hash(response)
        receipt = {
            "request_hash": request_hash,
            "response_hash": response_hash,
            "latency_ms": round(latency_ms, 3),
            "record_index": kwargs.get("record_index"),
            "example_id": kwargs.get("example_id"),
            "cache_hit": kwargs.get("_receipt_cache_hit", False),
            "retry_count": kwargs.get("_receipt_retry_count"),
            "trace_id": kwargs.get("trace_id"),
        }
        self._append_receipt(receipt)
        return response
