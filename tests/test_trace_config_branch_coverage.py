"""Additional branch coverage for trace_config helpers."""

from __future__ import annotations

import pytest

import insideLLMs.trace_contracts as trace_contracts
from insideLLMs.trace.trace_config import (
    TracePayloadNormaliser,
    load_trace_config,
    validate_with_config,
)
from insideLLMs.trace.trace_contracts import Violation


def _violation(code: str, seq: int) -> Violation:
    return Violation(code=code, event_seq=seq, detail=code)


def test_to_contracts_maps_all_json_schema_primitive_types():
    config = load_trace_config(
        {
            "contracts": {
                "tool_payloads": {
                    "tools": {
                        "typed_tool": {
                            "args_schema": {
                                "type": "object",
                                "required": ["name", "count"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "count": {"type": "integer"},
                                    "score": {"type": "number"},
                                    "enabled": {"type": "boolean"},
                                    "items": {"type": "array"},
                                    "meta": {"type": "object"},
                                },
                            }
                        }
                    }
                }
            }
        }
    )

    compiled = config.to_contracts()
    schema = compiled["tool_schemas"]["typed_tool"]
    assert schema.arg_types["name"] is str
    assert schema.arg_types["count"] is int
    assert schema.arg_types["score"] == (int, float)
    assert schema.arg_types["enabled"] is bool
    assert schema.arg_types["items"] is list
    assert schema.arg_types["meta"] is dict


def test_normaliser_legacy_redaction_list_paths_and_stream_chunk_non_string():
    normaliser = TracePayloadNormaliser(
        redact_enabled=True,
        json_pointers=[
            "not/a/pointer",  # ignored: does not start with /
            "/items/0",  # list final index replacement
            "/items/1/value",  # nested list->dict replacement
            "/items/not_an_int/value",  # invalid list index
            "/items/99/value",  # out of range
        ],
    )

    payload = {
        "text": 123,  # non-string stream text, should not create text_len
        "items": ["first", {"value": "secret"}, {"value": "other"}],
        "nested": {"arr": [{"k": "v"}]},  # exercises _walk list recursion
    }
    result = normaliser.normalise(payload, kind="stream_chunk")

    assert "text_len" not in result
    assert result["items"][0] == "<redacted>"
    assert result["items"][1]["value"] == "<redacted>"
    assert result["nested"]["arr"][0]["k"] == "v"


def test_validate_with_config_normalizes_events_and_filters_selected_violations(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = load_trace_config(
        {
            "contracts": {
                "fail_fast": False,
                "stream_boundaries": {
                    "chunk_kind": "chunk_evt",
                    "chunk_index_key": "idx",
                    "first_chunk_index": 1,
                    "require_end": False,
                    "require_monotonic_chunks": False,
                },
                "tool_results": {
                    "call_id_key": "cid",
                    "require_exactly_one_result": False,
                },
                "tool_payloads": {
                    "tool_key": "tool",
                    "args_key": "args",
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {"query": {"type": "string"}},
                            }
                        }
                    },
                },
                "tool_order": {
                    "must_follow": {"summarise": ["search"]},
                },
            }
        }
    )

    class EventNoToDict:
        seq = 3
        kind = "tool_result"
        payload = "raw-result"  # will be wrapped as {"value": ...}

    class EventWithToDict:
        def to_dict(self):
            return {
                "seq": 2,
                "kind": "tool_call_start",
                "payload": {"tool": "search", "args": {"query": "q"}, "cid": "c1"},
            }

    captured_events: list[list[dict[str, object]]] = []

    def validate_generate(events):
        captured_events.append(events)
        return []

    def validate_stream(events):
        captured_events.append(events)
        return [
            _violation("STREAM_NO_END", 10),
            _violation("STREAM_CHUNK_INDEX_MISMATCH", 11),
        ]

    def validate_tool_results(events):
        captured_events.append(events)
        return [_violation("TOOL_NO_RESULT", 12)]

    def validate_tool_payloads(events, _schemas):
        captured_events.append(events)
        return [_violation("TOOL_INVALID_ARGUMENTS", 13)]

    def validate_tool_order(events, _rules):
        captured_events.append(events)
        return [_violation("TOOL_ORDER_VIOLATION", 14)]

    monkeypatch.setattr(trace_contracts, "validate_generate_boundaries", validate_generate)
    monkeypatch.setattr(trace_contracts, "validate_stream_boundaries", validate_stream)
    monkeypatch.setattr(trace_contracts, "validate_tool_results", validate_tool_results)
    monkeypatch.setattr(trace_contracts, "validate_tool_payloads", validate_tool_payloads)
    monkeypatch.setattr(trace_contracts, "validate_tool_order", validate_tool_order)

    events = [
        {"seq": 1, "kind": "chunk_evt", "payload": {"idx": 2}},
        EventWithToDict(),
        EventNoToDict(),
    ]
    violations = validate_with_config(events, cfg)
    codes = [v.code for v in violations]

    # Filtered by config flags:
    assert "STREAM_NO_END" not in codes
    assert "STREAM_CHUNK_INDEX_MISMATCH" not in codes
    assert "TOOL_NO_RESULT" not in codes
    # Remaining validations still included:
    assert "TOOL_INVALID_ARGUMENTS" in codes
    assert "TOOL_ORDER_VIOLATION" in codes

    # Normalization checks on captured events.
    normalized = captured_events[0]
    assert normalized[0]["kind"] == "stream_chunk"
    assert normalized[0]["payload"]["chunk_index"] == 1
    assert normalized[1]["payload"]["tool_name"] == "search"
    assert normalized[1]["payload"]["tool_call_id"] == "c1"
    assert normalized[2]["payload"] == {"value": "raw-result"}


def test_validate_with_config_fail_fast_after_stream(monkeypatch: pytest.MonkeyPatch):
    cfg = load_trace_config({"contracts": {"fail_fast": True}})
    called = {"tool_results": False}

    monkeypatch.setattr(trace_contracts, "validate_generate_boundaries", lambda _events: [])
    monkeypatch.setattr(
        trace_contracts, "validate_stream_boundaries", lambda _events: [_violation("X", 1)]
    )
    monkeypatch.setattr(
        trace_contracts,
        "validate_tool_results",
        lambda _events: called.__setitem__("tool_results", True) or [],
    )
    monkeypatch.setattr(trace_contracts, "validate_tool_payloads", lambda _events, _schemas: [])
    monkeypatch.setattr(trace_contracts, "validate_tool_order", lambda _events, _rules: [])

    out = validate_with_config([{"seq": 0, "kind": "generate_start", "payload": {}}], cfg)
    assert len(out) == 1
    assert called["tool_results"] is False


def test_validate_with_config_fail_fast_after_tool_results(monkeypatch: pytest.MonkeyPatch):
    cfg = load_trace_config({"contracts": {"fail_fast": True}})
    called = {"payloads": False}

    monkeypatch.setattr(trace_contracts, "validate_generate_boundaries", lambda _events: [])
    monkeypatch.setattr(trace_contracts, "validate_stream_boundaries", lambda _events: [])
    monkeypatch.setattr(
        trace_contracts, "validate_tool_results", lambda _events: [_violation("TOOL", 2)]
    )
    monkeypatch.setattr(
        trace_contracts,
        "validate_tool_payloads",
        lambda _events, _schemas: called.__setitem__("payloads", True) or [],
    )
    monkeypatch.setattr(trace_contracts, "validate_tool_order", lambda _events, _rules: [])

    out = validate_with_config([{"seq": 0, "kind": "generate_start", "payload": {}}], cfg)
    assert len(out) == 1
    assert called["payloads"] is False


def test_validate_with_config_fail_fast_after_tool_payloads(monkeypatch: pytest.MonkeyPatch):
    cfg = load_trace_config(
        {
            "contracts": {
                "fail_fast": True,
                "tool_payloads": {
                    "tools": {
                        "search": {
                            "args_schema": {
                                "type": "object",
                                "required": [],
                                "properties": {},
                            }
                        }
                    }
                },
            }
        }
    )
    called = {"order": False}

    monkeypatch.setattr(trace_contracts, "validate_generate_boundaries", lambda _events: [])
    monkeypatch.setattr(trace_contracts, "validate_stream_boundaries", lambda _events: [])
    monkeypatch.setattr(trace_contracts, "validate_tool_results", lambda _events: [])
    monkeypatch.setattr(
        trace_contracts, "validate_tool_payloads", lambda _events, _schemas: [_violation("P", 3)]
    )
    monkeypatch.setattr(
        trace_contracts,
        "validate_tool_order",
        lambda _events, _rules: called.__setitem__("order", True) or [],
    )

    out = validate_with_config([{"seq": 0, "kind": "generate_start", "payload": {}}], cfg)
    assert len(out) == 1
    assert called["order"] is False
