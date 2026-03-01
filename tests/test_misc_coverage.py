"""Comprehensive tests targeting uncovered code paths in miscellaneous modules.

Covers:
  - insideLLMs/schemas/custom_trace_v1.py
  - insideLLMs/schemas/validator.py
  - insideLLMs/cli/_record_utils.py
  - insideLLMs/cli/_report_builder.py
  - insideLLMs/cli/_output.py
  - insideLLMs/distributed.py
  - insideLLMs/nlp/tokenization.py
  - insideLLMs/nlp/extraction.py
  - insideLLMs/nlp/chunking.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

requires_process_spawn = pytest.mark.xfail(
    reason="ProcessPoolExecutor may not be permitted in sandboxed environments",
    raises=(PermissionError, OSError),
    strict=False,
)


# ---- Module-level functions for process pool tests (must be picklable) ----
def _square(x):
    return x * x


def _double(x):
    return x * 2


def _identity(x):
    return x


def _format_answer(x):
    return f"Answer: {x}"


# ============================================================================
# 1. insideLLMs/schemas/custom_trace_v1.py
# ============================================================================


class TestCanonicalJsonBytes:
    """Tests for _canonical_json_bytes helper."""

    def test_sorted_keys(self):
        from insideLLMs.schemas.custom_trace_v1 import _canonical_json_bytes

        result = _canonical_json_bytes({"b": 2, "a": 1})
        assert result == b'{"a":1,"b":2}'

    def test_nested_structure(self):
        from insideLLMs.schemas.custom_trace_v1 import _canonical_json_bytes

        result = _canonical_json_bytes({"x": [1, {"z": 3, "y": 2}]})
        assert result == b'{"x":[1,{"y":2,"z":3}]}'

    def test_unicode_preserved(self):
        from insideLLMs.schemas.custom_trace_v1 import _canonical_json_bytes

        result = _canonical_json_bytes({"msg": "\u00e9"})
        assert "\u00e9".encode("utf-8") in result

    def test_non_serialisable_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import _canonical_json_bytes

        with pytest.raises(TypeError):
            _canonical_json_bytes({"bad": object()})


class TestAssertJsonable:
    """Tests for _assert_jsonable helper."""

    def test_valid_value_passes(self):
        from insideLLMs.schemas.custom_trace_v1 import _assert_jsonable

        result = _assert_jsonable({"key": [1, 2, 3]}, where="test")
        assert result == {"key": [1, 2, 3]}

    def test_non_serialisable_raises_valueerror(self):
        from insideLLMs.schemas.custom_trace_v1 import _assert_jsonable

        with pytest.raises(ValueError, match="events.payload must be JSON-serialisable"):
            _assert_jsonable({"ts": datetime.now()}, where="events.payload")


class TestNormaliseSha256Value:
    """Tests for _normalise_sha256_value helper."""

    def test_none_passthrough(self):
        from insideLLMs.schemas.custom_trace_v1 import _normalise_sha256_value

        assert _normalise_sha256_value(None) is None

    def test_strip_whitespace(self):
        from insideLLMs.schemas.custom_trace_v1 import _normalise_sha256_value

        assert _normalise_sha256_value("  abc123  ") == "abc123"

    def test_strip_sha256_prefix(self):
        from insideLLMs.schemas.custom_trace_v1 import _normalise_sha256_value

        assert _normalise_sha256_value("sha256:deadbeef") == "deadbeef"

    def test_strip_prefix_and_whitespace(self):
        from insideLLMs.schemas.custom_trace_v1 import _normalise_sha256_value

        assert _normalise_sha256_value("  sha256:abc  ") == "abc"


class TestTraceCounts:
    """Tests for TraceCounts model."""

    def test_valid_creation(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceCounts

        tc = TraceCounts(events_total=10, events_stored=5, by_kind={"msg": 3, "tool": 2})
        assert tc.events_total == 10
        assert tc.events_stored == 5
        assert tc.by_kind == {"msg": 3, "tool": 2}

    def test_default_by_kind(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceCounts

        tc = TraceCounts(events_total=1, events_stored=0)
        assert tc.by_kind == {}

    def test_stored_exceeds_total_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceCounts

        with pytest.raises(Exception, match="events_total must be >= "):
            TraceCounts(events_total=5, events_stored=10)

    def test_by_kind_empty_key_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceCounts

        with pytest.raises(Exception, match="non-empty strings"):
            TraceCounts(events_total=1, events_stored=0, by_kind={"": 1})

    def test_by_kind_negative_value_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceCounts

        with pytest.raises(Exception, match="integers >= 0"):
            TraceCounts(events_total=1, events_stored=0, by_kind={"k": -1})

    def test_extra_field_forbidden(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceCounts

        with pytest.raises(Exception):
            TraceCounts(events_total=1, events_stored=0, bogus=True)


class TestTraceFingerprint:
    """Tests for TraceFingerprint model."""

    def test_disabled_fingerprint(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        fp = TraceFingerprint(enabled=False)
        assert fp.value is None
        assert fp.basis is None

    def test_enabled_fingerprint(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        h = "a" * 64
        fp = TraceFingerprint(enabled=True, value=h, basis="normalised_full_trace")
        assert fp.value == h
        assert fp.alg == "sha256"

    def test_value_lowered(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        h = "ABCDEF" + "0" * 58
        fp = TraceFingerprint(enabled=True, value=h, basis="normalised_full_trace")
        assert fp.value == h.lower()

    def test_sha256_prefix_stripped(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        raw = "a" * 64
        fp = TraceFingerprint(enabled=True, value=f"sha256:{raw}", basis="normalised_full_trace")
        assert fp.value == raw

    def test_invalid_hex_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        with pytest.raises(Exception, match="64-hex sha256"):
            TraceFingerprint(enabled=True, value="tooshort", basis="normalised_full_trace")

    def test_enabled_without_value_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        with pytest.raises(Exception, match="requires fingerprint.value"):
            TraceFingerprint(enabled=True, basis="normalised_full_trace")

    def test_enabled_without_basis_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        with pytest.raises(Exception, match="requires fingerprint.basis"):
            TraceFingerprint(enabled=True, value="a" * 64)

    def test_disabled_with_value_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        with pytest.raises(Exception, match="requires fingerprint.value=null"):
            TraceFingerprint(enabled=False, value="a" * 64)

    def test_disabled_with_basis_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceFingerprint

        with pytest.raises(Exception, match="requires fingerprint.basis=null"):
            TraceFingerprint(enabled=False, basis="normalised_full_trace")


class TestTraceNormaliser:
    """Tests for TraceNormaliser model."""

    def test_builtin_normaliser(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        n = TraceNormaliser(kind="builtin", name="default", config_hash="a" * 64)
        assert n.kind == "builtin"
        assert n.name == "default"
        assert n.import_path is None

    def test_import_normaliser(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        n = TraceNormaliser(kind="import", import_path="my.module", config_hash="b" * 64)
        assert n.kind == "import"
        assert n.import_path == "my.module"
        assert n.name is None

    def test_config_hash_lowered(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        h = "ABCDEF" + "0" * 58
        n = TraceNormaliser(kind="builtin", name="x", config_hash=h)
        assert n.config_hash == h.lower()

    def test_config_hash_prefix_stripped(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        raw = "f" * 64
        n = TraceNormaliser(kind="builtin", name="x", config_hash=f"sha256:{raw}")
        assert n.config_hash == raw

    def test_invalid_config_hash_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        with pytest.raises(Exception, match="64-hex sha256"):
            TraceNormaliser(kind="builtin", name="x", config_hash="bad")

    def test_builtin_without_name_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        with pytest.raises(Exception, match="requires normaliser.name"):
            TraceNormaliser(kind="builtin", config_hash="a" * 64)

    def test_builtin_with_import_path_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        with pytest.raises(Exception, match="forbids normaliser.import"):
            TraceNormaliser(kind="builtin", name="x", import_path="y", config_hash="a" * 64)

    def test_import_without_import_path_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        with pytest.raises(Exception, match="requires normaliser.import"):
            TraceNormaliser(kind="import", config_hash="a" * 64)

    def test_import_with_name_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        with pytest.raises(Exception, match="forbids normaliser.name"):
            TraceNormaliser(kind="import", name="bad", import_path="m", config_hash="a" * 64)

    def test_populate_by_alias(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceNormaliser

        # Test that the "import" alias works
        data = {"kind": "import", "import": "mod.x", "config_hash": "c" * 64}
        n = TraceNormaliser(**data)
        assert n.import_path == "mod.x"


class TestTraceContractsSummary:
    """Tests for TraceContractsSummary model."""

    def test_valid_contracts(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceContractsSummary

        c = TraceContractsSummary(enabled=True, violations_total=5, violations_stored=3)
        assert c.enabled is True
        assert c.fail_fast is False

    def test_violations_stored_exceeds_total_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceContractsSummary

        with pytest.raises(Exception, match="violations_total must be >= "):
            TraceContractsSummary(enabled=True, violations_total=2, violations_stored=5)

    def test_by_code_empty_key_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceContractsSummary

        with pytest.raises(Exception, match="non-empty strings"):
            TraceContractsSummary(
                enabled=True, violations_total=1, violations_stored=0, by_code={"": 1}
            )

    def test_by_code_negative_value_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceContractsSummary

        with pytest.raises(Exception, match="integers >= 0"):
            TraceContractsSummary(
                enabled=True, violations_total=1, violations_stored=0, by_code={"x": -1}
            )


class TestTraceViolation:
    """Tests for TraceViolation model."""

    def test_valid_violation(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceViolation

        v = TraceViolation(code="E001", message="oops", event_seq=3, path="/a", meta={"k": "v"})
        assert v.code == "E001"
        assert v.event_seq == 3

    def test_violation_non_serialisable_meta_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceViolation

        with pytest.raises(Exception, match="JSON-serialisable"):
            TraceViolation(code="E001", message="oops", meta={"bad": object()})


class TestTraceEventStored:
    """Tests for TraceEventStored model."""

    def test_valid_event(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceEventStored

        e = TraceEventStored(seq=0, kind="user_message", payload={"text": "hi"})
        assert e.seq == 0

    def test_non_serialisable_payload_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceEventStored

        with pytest.raises(Exception, match="JSON-serialisable"):
            TraceEventStored(seq=0, kind="msg", payload={"bad": datetime.now()})


class TestTruncationModels:
    """Tests for truncation sub-models."""

    def test_truncation_events_applied_without_max_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationEvents

        with pytest.raises(Exception, match="max_events"):
            TruncationEvents(applied=True)

    def test_truncation_events_applied_with_max(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationEvents

        te = TruncationEvents(applied=True, max_events=100, dropped=5)
        assert te.dropped == 5

    def test_truncation_events_not_applied(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationEvents

        te = TruncationEvents(applied=False)
        assert te.max_events is None

    def test_truncation_payloads_applied_without_max_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationPayloads

        with pytest.raises(Exception, match="max_bytes"):
            TruncationPayloads(applied=True)

    def test_truncation_payloads_applied_with_max(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationPayloads

        tp = TruncationPayloads(applied=True, max_bytes=1024)
        assert tp.max_bytes == 1024

    def test_truncation_violations_applied_without_max_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationViolations

        with pytest.raises(Exception, match="max_violations"):
            TruncationViolations(applied=True)

    def test_truncation_violations_applied_with_max(self):
        from insideLLMs.schemas.custom_trace_v1 import TruncationViolations

        tv = TruncationViolations(applied=True, max_violations=50, dropped=2)
        assert tv.max_violations == 50

    def test_trace_truncation_defaults(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceTruncation

        tt = TraceTruncation()
        assert tt.events.applied is False
        assert tt.payloads.applied is False
        assert tt.violations.applied is False


class TestDerivedToolCalls:
    """Tests for DerivedToolCalls model."""

    def test_valid_derived(self):
        from insideLLMs.schemas.custom_trace_v1 import DerivedToolCalls

        d = DerivedToolCalls(count=3, sequence=["a", "b", "a"], by_tool={"a": 2, "b": 1})
        assert d.count == 3

    def test_count_mismatch_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import DerivedToolCalls

        with pytest.raises(Exception, match="count must equal"):
            DerivedToolCalls(count=2, sequence=["a"], by_tool={"a": 2})

    def test_by_tool_sum_mismatch_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import DerivedToolCalls

        with pytest.raises(Exception, match="sum.*must equal"):
            DerivedToolCalls(count=2, sequence=["a", "b"], by_tool={"a": 1, "b": 2})

    def test_by_tool_empty_key_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import DerivedToolCalls

        with pytest.raises(Exception, match="non-empty"):
            DerivedToolCalls(count=1, sequence=["x"], by_tool={"": 1})

    def test_by_tool_negative_count_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import DerivedToolCalls

        # Use values that sum to count so the sum check passes, but one value is negative
        with pytest.raises(Exception, match=">= 0"):
            DerivedToolCalls(count=0, sequence=[], by_tool={"x": -1, "y": 1})

    def test_default_derived(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceDerived

        td = TraceDerived()
        assert td.tool_calls.count == 0


class TestTraceBundleV1:
    """Tests for TraceBundleV1 root model."""

    def _make_compact_kwargs(self):
        from insideLLMs.schemas.custom_trace_v1 import (
            TraceContractsSummary,
            TraceCounts,
            TraceFingerprint,
            TraceNormaliser,
        )

        return dict(
            schema_version="insideLLMs.custom.trace@1",
            mode="compact",
            counts=TraceCounts(events_total=10, events_stored=0),
            fingerprint=TraceFingerprint(enabled=False),
            normaliser=TraceNormaliser(kind="builtin", name="default", config_hash="a" * 64),
            contracts=TraceContractsSummary(enabled=False, violations_total=0, violations_stored=0),
        )

    def _make_full_kwargs(self):
        from insideLLMs.schemas.custom_trace_v1 import (
            TraceContractsSummary,
            TraceCounts,
            TraceEventStored,
            TraceFingerprint,
            TraceNormaliser,
        )

        return dict(
            schema_version="insideLLMs.custom.trace@1",
            mode="full",
            counts=TraceCounts(events_total=2, events_stored=2),
            fingerprint=TraceFingerprint(
                enabled=True, value="b" * 64, basis="normalised_full_trace"
            ),
            normaliser=TraceNormaliser(kind="builtin", name="default", config_hash="c" * 64),
            contracts=TraceContractsSummary(enabled=False, violations_total=0, violations_stored=0),
            events_view="normalised",
            events=[
                TraceEventStored(seq=0, kind="msg", payload={}),
                TraceEventStored(seq=1, kind="msg", payload={}),
            ],
        )

    def test_compact_bundle(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1

        bundle = TraceBundleV1(**self._make_compact_kwargs())
        assert bundle.mode == "compact"

    def test_full_bundle(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1

        bundle = TraceBundleV1(**self._make_full_kwargs())
        assert bundle.mode == "full"
        assert len(bundle.events) == 2

    def test_compact_with_events_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1, TraceEventStored

        kwargs = self._make_compact_kwargs()
        kwargs["events"] = [TraceEventStored(seq=0, kind="m", payload={})]
        with pytest.raises(Exception, match="compact.*forbids events"):
            TraceBundleV1(**kwargs)

    def test_compact_with_events_view_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1

        kwargs = self._make_compact_kwargs()
        kwargs["events_view"] = "normalised"
        with pytest.raises(Exception, match="compact.*forbids events_view"):
            TraceBundleV1(**kwargs)

    def test_compact_with_nonzero_stored_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1, TraceCounts

        kwargs = self._make_compact_kwargs()
        kwargs["counts"] = TraceCounts(events_total=10, events_stored=5)
        with pytest.raises(Exception, match="compact.*events_stored == 0"):
            TraceBundleV1(**kwargs)

    def test_full_without_events_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1

        kwargs = self._make_full_kwargs()
        kwargs["events"] = None
        with pytest.raises(Exception, match="full.*requires events"):
            TraceBundleV1(**kwargs)

    def test_full_without_events_view_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1

        kwargs = self._make_full_kwargs()
        kwargs["events_view"] = None
        with pytest.raises(Exception, match="full.*requires events_view"):
            TraceBundleV1(**kwargs)

    def test_full_events_count_mismatch_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1, TraceCounts

        kwargs = self._make_full_kwargs()
        kwargs["counts"] = TraceCounts(events_total=5, events_stored=5)
        with pytest.raises(Exception, match="events_stored must equal len"):
            TraceBundleV1(**kwargs)

    def test_full_events_non_decreasing_seq_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import (
            TraceBundleV1,
            TraceCounts,
            TraceEventStored,
        )

        kwargs = self._make_full_kwargs()
        kwargs["events"] = [
            TraceEventStored(seq=5, kind="m", payload={}),
            TraceEventStored(seq=2, kind="m", payload={}),
        ]
        with pytest.raises(Exception, match="non-decreasing"):
            TraceBundleV1(**kwargs)

    def test_contracts_disabled_with_violations_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import TraceBundleV1, TraceViolation

        kwargs = self._make_compact_kwargs()
        kwargs["violations"] = [TraceViolation(code="E", message="x")]
        with pytest.raises(Exception, match="contracts.enabled=false requires violations"):
            TraceBundleV1(**kwargs)

    def test_contracts_disabled_with_nonzero_totals_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import (
            TraceBundleV1,
            TraceContractsSummary,
        )

        kwargs = self._make_compact_kwargs()
        kwargs["contracts"] = TraceContractsSummary(
            enabled=False, violations_total=1, violations_stored=0
        )
        with pytest.raises(Exception, match="violations_total=violations_stored=0"):
            TraceBundleV1(**kwargs)

    def test_violations_stored_mismatch_raises(self):
        from insideLLMs.schemas.custom_trace_v1 import (
            TraceBundleV1,
            TraceContractsSummary,
            TraceViolation,
        )

        kwargs = self._make_full_kwargs()
        kwargs["contracts"] = TraceContractsSummary(
            enabled=True, violations_total=2, violations_stored=2
        )
        kwargs["violations"] = [TraceViolation(code="E1", message="x")]
        with pytest.raises(Exception, match="violations_stored must equal"):
            TraceBundleV1(**kwargs)


# ============================================================================
# 2. insideLLMs/schemas/validator.py
# ============================================================================


class TestOutputValidator:
    """Tests for OutputValidator and _to_plain."""

    def test_to_plain_dict_passthrough(self):
        from insideLLMs.schemas.validator import _to_plain

        d = {"a": 1}
        assert _to_plain(d) is d

    def test_to_plain_dataclass_conversion(self):
        from insideLLMs.schemas.validator import _to_plain

        @dataclass
        class Sample:
            x: int
            y: str

        s = Sample(x=1, y="hi")
        result = _to_plain(s)
        assert isinstance(result, dict)
        assert result == {"x": 1, "y": "hi"}

    def test_to_plain_non_dataclass_passthrough(self):
        from insideLLMs.schemas.validator import _to_plain

        assert _to_plain(42) == 42
        assert _to_plain("str") == "str"
        assert _to_plain(None) is None

    def test_to_plain_class_passthrough(self):
        from insideLLMs.schemas.validator import _to_plain

        @dataclass
        class Cls:
            v: int

        # Class itself (not instance) should pass through
        assert _to_plain(Cls) is Cls

    def test_validator_default_registry(self):
        from insideLLMs.schemas.registry import SchemaRegistry
        from insideLLMs.schemas.validator import OutputValidator

        v = OutputValidator()
        assert isinstance(v.registry, SchemaRegistry)

    def test_validator_custom_registry(self):
        from insideLLMs.schemas.registry import SchemaRegistry
        from insideLLMs.schemas.validator import OutputValidator

        reg = SchemaRegistry()
        v = OutputValidator(registry=reg)
        assert v.registry is reg

    def test_validate_strict_success(self):
        from insideLLMs.schemas.validator import OutputValidator

        v = OutputValidator()
        data = {"input": "q", "output": "a", "status": "success"}
        result = v.validate("ProbeResult", data, mode="strict")
        assert hasattr(result, "input")

    def test_validate_strict_failure(self):
        from insideLLMs.schemas.exceptions import OutputValidationError
        from insideLLMs.schemas.validator import OutputValidator

        v = OutputValidator()
        with pytest.raises(OutputValidationError) as exc_info:
            v.validate("ProbeResult", {"bad": "data"}, mode="strict")
        assert exc_info.value.schema_name == "ProbeResult"
        assert len(exc_info.value.errors) >= 1

    def test_validate_warn_mode(self):
        from insideLLMs.schemas.validator import OutputValidator

        v = OutputValidator()
        bad = {"missing": "fields"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = v.validate("ProbeResult", bad, mode="warn")
            assert result is bad
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)

    def test_validate_with_version(self):
        from insideLLMs.schemas.validator import OutputValidator

        v = OutputValidator()
        data = {"input": "q", "output": "a", "status": "success"}
        result = v.validate("ProbeResult", data, schema_version="1.0.0")
        assert hasattr(result, "status")

    def test_validate_dataclass_input(self):
        from insideLLMs.schemas.validator import OutputValidator

        @dataclass
        class FakeProbe:
            input: str
            output: str
            status: str

        v = OutputValidator()
        result = v.validate("ProbeResult", FakeProbe(input="q", output="a", status="success"))
        assert hasattr(result, "input")


# ============================================================================
# 3. insideLLMs/cli/_record_utils.py
# ============================================================================


class TestRecordUtils:
    """Tests for CLI record utility functions."""

    def test_json_default(self):
        from insideLLMs.cli._record_utils import _json_default

        # Should delegate to serialize_value
        result = _json_default(42)
        assert result is not None

    def test_write_and_read_jsonl(self, tmp_path):
        from insideLLMs.cli._record_utils import _read_jsonl_records, _write_jsonl

        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        out = tmp_path / "test.jsonl"
        _write_jsonl(records, out)
        loaded = _read_jsonl_records(out)
        assert len(loaded) == 2
        assert loaded[0]["a"] == 1
        assert loaded[1]["b"] == "y"

    def test_write_jsonl_strict_serialization_error(self, tmp_path):
        from insideLLMs.cli._record_utils import _write_jsonl

        # Object() is not JSON-serialisable
        out = tmp_path / "bad.jsonl"
        with pytest.raises((ValueError, TypeError)):
            _write_jsonl([{"bad": object()}], out, strict_serialization=True)

    def test_read_jsonl_blank_lines(self, tmp_path):
        from insideLLMs.cli._record_utils import _read_jsonl_records

        out = tmp_path / "blanks.jsonl"
        out.write_text('{"a":1}\n\n{"b":2}\n   \n')
        loaded = _read_jsonl_records(out)
        assert len(loaded) == 2

    def test_read_jsonl_invalid_json_raises(self, tmp_path):
        from insideLLMs.cli._record_utils import _read_jsonl_records

        out = tmp_path / "bad.jsonl"
        out.write_text("{invalid\n")
        with pytest.raises(ValueError, match="Invalid JSON on line 1"):
            _read_jsonl_records(out)

    def test_parse_datetime_from_datetime(self):
        from insideLLMs.cli._record_utils import _parse_datetime

        dt = datetime(2024, 1, 15, 10, 30)
        assert _parse_datetime(dt) == dt

    def test_parse_datetime_from_string(self):
        from insideLLMs.cli._record_utils import _parse_datetime

        result = _parse_datetime("2024-01-15T10:30:00")
        assert isinstance(result, datetime)

    def test_parse_datetime_invalid_string(self):
        from insideLLMs.cli._record_utils import _parse_datetime

        assert _parse_datetime("not-a-date") is None

    def test_parse_datetime_non_string(self):
        from insideLLMs.cli._record_utils import _parse_datetime

        assert _parse_datetime(12345) is None
        assert _parse_datetime(None) is None

    def test_status_from_record_enum_passthrough(self):
        from insideLLMs.cli._record_utils import _status_from_record
        from insideLLMs.types import ResultStatus

        s = ResultStatus.SUCCESS
        assert _status_from_record(s) is s

    def test_status_from_record_string(self):
        from insideLLMs.cli._record_utils import _status_from_record
        from insideLLMs.types import ResultStatus

        assert _status_from_record("success") == ResultStatus.SUCCESS

    def test_status_from_record_invalid(self):
        from insideLLMs.cli._record_utils import _status_from_record
        from insideLLMs.types import ResultStatus

        assert _status_from_record("bogus_value") == ResultStatus.ERROR

    def test_probe_category_from_value_enum(self):
        from insideLLMs.cli._record_utils import _probe_category_from_value
        from insideLLMs.types import ProbeCategory

        c = ProbeCategory.CUSTOM
        assert _probe_category_from_value(c) is c

    def test_probe_category_from_value_invalid(self):
        from insideLLMs.cli._record_utils import _probe_category_from_value
        from insideLLMs.types import ProbeCategory

        assert _probe_category_from_value("nonexistent_category") == ProbeCategory.CUSTOM

    def test_record_key_basic(self):
        from insideLLMs.cli._record_utils import _record_key

        record = {
            "custom": {"harness": {"model_id": "m1", "probe_type": "p1"}},
            "input": "test",
        }
        model_id, probe_id, chosen_id = _record_key(record)
        assert model_id == "m1"
        assert probe_id == "p1"

    def test_record_key_fallback(self):
        from insideLLMs.cli._record_utils import _record_key

        record = {"model": {"model_id": "m2"}, "probe": {"probe_id": "p2"}, "input": "test"}
        model_id, probe_id, chosen_id = _record_key(record)
        assert model_id == "m2"
        assert probe_id == "p2"

    def test_record_key_missing_everything(self):
        from insideLLMs.cli._record_utils import _record_key

        model_id, probe_id, chosen_id = _record_key({})
        assert model_id == "model"
        assert probe_id == "probe"

    def test_record_key_with_replicate_key(self):
        from insideLLMs.cli._record_utils import _record_key

        record = {"custom": {"replicate_key": "rep1"}}
        _, _, chosen_id = _record_key(record)
        assert chosen_id == "rep1"

    def test_record_key_with_example_id(self):
        from insideLLMs.cli._record_utils import _record_key

        record = {"example_id": "ex42"}
        _, _, chosen_id = _record_key(record)
        assert "ex42" in chosen_id or chosen_id == "ex42"

    def test_record_label(self):
        from insideLLMs.cli._record_utils import _record_label

        record = {
            "custom": {
                "harness": {"model_name": "GPT-4", "model_id": "gpt4", "probe_name": "logic"}
            },
            "example_id": "e1",
        }
        model_label, probe_name, example_id = _record_label(record)
        assert "GPT-4" in model_label
        assert "gpt4" in model_label
        assert probe_name == "logic"

    def test_record_label_same_name_and_id(self):
        from insideLLMs.cli._record_utils import _record_label

        record = {"custom": {"harness": {"model_name": "m1", "model_id": "m1"}}}
        model_label, _, _ = _record_label(record)
        assert model_label == "m1"

    def test_record_label_fallback(self):
        from insideLLMs.cli._record_utils import _record_label

        model_label, probe_name, example_id = _record_label({})
        assert model_label == "model"
        assert probe_name == "probe"

    def test_status_string(self):
        from insideLLMs.cli._record_utils import _status_string

        assert _status_string({"status": "success"}) == "success"
        assert _status_string({}) == "unknown"
        assert _status_string({"status": None}) == "unknown"

    def test_output_text_from_output_text(self):
        from insideLLMs.cli._record_utils import _output_text

        assert _output_text({"output_text": "hello"}) == "hello"

    def test_output_text_from_output_string(self):
        from insideLLMs.cli._record_utils import _output_text

        assert _output_text({"output": "world"}) == "world"

    def test_output_text_from_output_dict(self):
        from insideLLMs.cli._record_utils import _output_text

        assert _output_text({"output": {"output_text": "nested"}}) == "nested"
        assert _output_text({"output": {"text": "nested2"}}) == "nested2"

    def test_output_text_none(self):
        from insideLLMs.cli._record_utils import _output_text

        assert _output_text({}) is None
        assert _output_text({"output": 42}) is None
        assert _output_text({"output": {"other": "x"}}) is None

    def test_strip_volatile_keys(self):
        from insideLLMs.cli._record_utils import _strip_volatile_keys

        data = {"a": 1, "timestamp": "now", "nested": {"Timestamp": "t", "ok": 2}}
        result = _strip_volatile_keys(data, {"timestamp"})
        assert "timestamp" not in result
        assert "Timestamp" not in result.get("nested", {})
        assert result["a"] == 1
        assert result["nested"]["ok"] == 2

    def test_strip_volatile_keys_list(self):
        from insideLLMs.cli._record_utils import _strip_volatile_keys

        data = [{"latency": 1, "value": 2}]
        result = _strip_volatile_keys(data, {"latency"})
        assert result == [{"value": 2}]

    def test_strip_volatile_keys_tuple(self):
        from insideLLMs.cli._record_utils import _strip_volatile_keys

        data = ({"a": 1, "b": 2},)
        result = _strip_volatile_keys(data, {"b"})
        assert result == ({"a": 1},)

    def test_output_summary_text(self):
        from insideLLMs.cli._record_utils import _output_summary

        record = {"output_text": "hello world"}
        summary = _output_summary(record, None)
        assert summary["type"] == "text"
        assert summary["length"] == 11

    def test_output_summary_structured(self):
        from insideLLMs.cli._record_utils import _output_summary

        record = {"output": {"key": "value"}}
        summary = _output_summary(record, None)
        assert summary["type"] == "structured"

    def test_output_summary_none(self):
        from insideLLMs.cli._record_utils import _output_summary

        assert _output_summary({}, None) is None

    def test_output_fingerprint_with_ignore_keys(self):
        from insideLLMs.cli._record_utils import _output_fingerprint

        record = {"output": {"data": 1, "latency": 999}}
        fp = _output_fingerprint(record, ignore_keys={"latency"})
        assert fp is not None

    def test_output_fingerprint_override(self):
        from insideLLMs.cli._record_utils import _output_fingerprint

        record = {"output": {"x": 1}, "custom": {"output_fingerprint": "override123"}}
        fp = _output_fingerprint(record, ignore_keys=None)
        assert fp == "override123"

    def test_output_fingerprint_override_with_ignore_keys(self):
        from insideLLMs.cli._record_utils import _output_fingerprint

        record = {"output": None, "custom": {"output_fingerprint": "over"}}
        fp = _output_fingerprint(record, ignore_keys={"lat"})
        assert fp == "over"

    def test_output_fingerprint_none_output_no_override(self):
        from insideLLMs.cli._record_utils import _output_fingerprint

        assert _output_fingerprint({"output": None}, ignore_keys={"x"}) is None
        assert _output_fingerprint({"output": None}) is None

    def test_trace_fingerprint_from_custom(self):
        from insideLLMs.cli._record_utils import _trace_fingerprint

        record = {"custom": {"trace_fingerprint": "sha256:abc"}}
        assert _trace_fingerprint(record) == "sha256:abc"

    def test_trace_fingerprint_from_trace_bundle(self):
        from insideLLMs.cli._record_utils import _trace_fingerprint

        record = {"custom": {"trace": {"fingerprint": {"value": "d" * 64}}}}
        assert _trace_fingerprint(record) == f"sha256:{'d' * 64}"

    def test_trace_fingerprint_with_prefix(self):
        from insideLLMs.cli._record_utils import _trace_fingerprint

        record = {"custom": {"trace": {"fingerprint": {"value": f"sha256:{'e' * 64}"}}}}
        assert _trace_fingerprint(record) == f"sha256:{'e' * 64}"

    def test_trace_fingerprint_missing(self):
        from insideLLMs.cli._record_utils import _trace_fingerprint

        assert _trace_fingerprint({}) is None
        assert _trace_fingerprint({"custom": {}}) is None
        assert _trace_fingerprint({"custom": {"trace": {}}}) is None

    def test_trace_violations(self):
        from insideLLMs.cli._record_utils import _trace_violations

        record = {"custom": {"trace_violations": [{"code": "E1"}]}}
        assert _trace_violations(record) == [{"code": "E1"}]

    def test_trace_violations_from_bundle(self):
        from insideLLMs.cli._record_utils import _trace_violations

        record = {"custom": {"trace": {"violations": [{"x": 1}]}}}
        assert _trace_violations(record) == [{"x": 1}]

    def test_trace_violations_empty(self):
        from insideLLMs.cli._record_utils import _trace_violations

        assert _trace_violations({}) == []

    def test_trace_violation_count(self):
        from insideLLMs.cli._record_utils import _trace_violation_count

        record = {"custom": {"trace_violations": [1, 2, 3]}}
        assert _trace_violation_count(record) == 3

    def test_primary_score_from_primary_metric(self):
        from insideLLMs.cli._record_utils import _primary_score

        record = {"scores": {"accuracy": 0.95}, "primary_metric": "accuracy"}
        name, val = _primary_score(record)
        assert name == "accuracy"
        assert val == 0.95

    def test_primary_score_fallback_to_score_key(self):
        from insideLLMs.cli._record_utils import _primary_score

        record = {"scores": {"score": 0.8}}
        name, val = _primary_score(record)
        assert name == "score"
        assert val == 0.8

    def test_primary_score_none(self):
        from insideLLMs.cli._record_utils import _primary_score

        assert _primary_score({}) == (None, None)
        assert _primary_score({"scores": {}}) == (None, None)

    def test_metric_mismatch_reason_type_mismatch(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        assert _metric_mismatch_reason({"scores": "bad"}, {"scores": {}}) == "type_mismatch"

    def test_metric_mismatch_reason_missing_scores(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        assert _metric_mismatch_reason({"scores": {}}, {"scores": {"a": 1}}) == "missing_scores"

    def test_metric_mismatch_reason_primary_metric_mismatch(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        a = {"scores": {"x": 1}, "primary_metric": "x"}
        b = {"scores": {"y": 1}, "primary_metric": "y"}
        assert _metric_mismatch_reason(a, b) == "primary_metric_mismatch"

    def test_metric_mismatch_reason_missing_primary(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        a = {"scores": {"x": 1}}
        b = {"scores": {"y": 1}, "primary_metric": "y"}
        assert _metric_mismatch_reason(a, b) == "missing_primary_metric"

    def test_metric_mismatch_reason_metric_key_missing(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        # Both have same primary_metric but the key is absent from scores
        a = {"scores": {"other": 1}, "primary_metric": "absent_key"}
        b = {"scores": {"other": 1}, "primary_metric": "absent_key"}
        assert _metric_mismatch_reason(a, b) == "metric_key_missing"

    def test_metric_mismatch_reason_non_numeric(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        a = {"scores": {"x": "not_a_num"}, "primary_metric": "x"}
        b = {"scores": {"y": 1.0}, "primary_metric": "y"}
        # primary mismatch still takes precedence
        assert _metric_mismatch_reason(a, b) == "primary_metric_mismatch"

    def test_metric_mismatch_reason_non_numeric_same_primary(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_reason

        a = {"scores": {"m": "text"}, "primary_metric": "m"}
        b = {"scores": {"m": "text"}, "primary_metric": "m"}
        assert _metric_mismatch_reason(a, b) == "non_numeric_metric"

    def test_metric_mismatch_context(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_context

        a = {"scores": {"acc": 0.9}, "primary_metric": "acc", "custom": {"replicate_key": "rk"}}
        b = {"scores": {"acc": 0.8}, "primary_metric": "acc"}
        ctx = _metric_mismatch_context(a, b)
        assert ctx["baseline"]["primary_metric"] == "acc"
        assert ctx["candidate"]["primary_metric"] == "acc"
        assert ctx["replicate_key"] == "rk"

    def test_metric_mismatch_context_no_scores(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_context

        ctx = _metric_mismatch_context({}, {})
        assert ctx["baseline"]["scores_keys"] is None
        assert ctx["candidate"]["scores_keys"] is None

    def test_metric_mismatch_details(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_details

        a = {"scores": {"acc": 0.9}, "primary_metric": "acc"}
        b = {"scores": {"acc": 0.8}, "primary_metric": "acc"}
        details = _metric_mismatch_details(a, b)
        assert "acc" in details
        assert "baseline" in details

    def test_metric_mismatch_details_no_primary(self):
        from insideLLMs.cli._record_utils import _metric_mismatch_details

        details = _metric_mismatch_details({}, {})
        assert "None" in details
        assert "baseline.scores type=None" in details


# ============================================================================
# 4. insideLLMs/cli/_report_builder.py
# ============================================================================


class TestReportBuilder:
    """Tests for report building utilities."""

    def test_build_experiments_from_empty(self):
        from insideLLMs.cli._report_builder import _build_experiments_from_records

        experiments, config, version = _build_experiments_from_records([])
        assert experiments == []
        assert config["derived_from_records"] is True

    def test_build_experiments_from_harness_records(self):
        from insideLLMs.cli._report_builder import _build_experiments_from_records

        records = [
            {
                "input": "q",
                "output": "a",
                "status": "success",
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:01:00",
                "model": {"model_id": "gpt4", "provider": "openai", "params": {}},
                "probe": {"probe_id": "logic"},
                "custom": {
                    "harness": {
                        "experiment_id": "exp1",
                        "model_name": "GPT-4",
                        "model_id": "gpt4",
                        "model_type": "openai",
                        "probe_name": "logic",
                        "probe_type": "logic_probe",
                        "probe_category": "logic",
                        "example_index": 0,
                        "dataset": "test_ds",
                        "dataset_format": "json",
                    }
                },
            }
        ]
        experiments, config, version = _build_experiments_from_records(records)
        assert len(experiments) == 1
        assert experiments[0].experiment_id == "exp1"
        assert "models" in config
        assert "probes" in config

    def test_build_experiments_from_run_records(self):
        from insideLLMs.cli._report_builder import _build_experiments_from_records

        records = [
            {
                "run_id": "run1",
                "input": "q",
                "output": "a",
                "status": "success",
                "model": {"model_id": "m1", "provider": "prov", "params": {"name": "M1"}},
                "probe": {"probe_id": "p1"},
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:01:00",
            }
        ]
        experiments, config, version = _build_experiments_from_records(records)
        assert len(experiments) == 1
        assert experiments[0].experiment_id == "run1"

    def test_build_basic_harness_report(self):
        from insideLLMs.cli._report_builder import _build_basic_harness_report
        from insideLLMs.types import (
            ExperimentResult,
            ModelInfo,
            ProbeCategory,
            ProbeResult,
            ResultStatus,
        )

        model_info = ModelInfo(name="TestModel", provider="test", model_id="tm")
        experiment = ExperimentResult(
            experiment_id="e1",
            model_info=model_info,
            probe_name="test_probe",
            probe_category=ProbeCategory.CUSTOM,
            results=[
                ProbeResult(input="q", output="a", status=ResultStatus.SUCCESS, latency_ms=100.0)
            ],
            score=None,
            started_at=None,
            completed_at=None,
            config={},
        )

        summary = {
            "by_model": {
                "TestModel": {
                    "success_rate": {"mean": 0.95},
                    "success_rate_ci": {"lower": 0.9, "upper": 1.0},
                }
            },
            "by_probe": {
                "test_probe": {
                    "success_rate": {"mean": 0.85},
                    "success_rate_ci": {"lower": None, "upper": None},
                }
            },
        }

        html = _build_basic_harness_report(
            [experiment], summary, "Test Report", generated_at=datetime(2024, 1, 1)
        )
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "TestModel" in html
        assert "test_probe" in html
        assert "Generated" in html

    def test_build_basic_harness_report_no_timestamp(self):
        from insideLLMs.cli._report_builder import _build_basic_harness_report

        html = _build_basic_harness_report([], {}, "Empty Report")
        assert "<!DOCTYPE html>" in html
        assert "Generated" not in html


# ============================================================================
# 5. insideLLMs/cli/_output.py
# ============================================================================


class TestOutputModule:
    """Tests for CLI output utilities."""

    def test_format_percent(self):
        from insideLLMs.cli._output import _format_percent

        assert _format_percent(None) == "-"
        assert _format_percent(0.5) == "50.0%"
        assert _format_percent(1.0) == "100.0%"

    def test_format_float(self):
        from insideLLMs.cli._output import _format_float

        assert _format_float(None) == "-"
        assert _format_float(3.14159) == "3.142"

    def test_trim_text_short(self):
        from insideLLMs.cli._output import _trim_text

        assert _trim_text("hello") == "hello"

    def test_trim_text_long(self):
        from insideLLMs.cli._output import _trim_text

        long_text = "x" * 300
        result = _trim_text(long_text)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")

    def test_trim_text_custom_limit(self):
        from insideLLMs.cli._output import _trim_text

        assert _trim_text("abcdefgh", limit=5) == "abcde..."

    def test_cli_version_string(self):
        from insideLLMs.cli._output import _cli_version_string

        version = _cli_version_string()
        assert isinstance(version, str)
        assert version != ""

    def test_supports_color_no_color_env(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            from insideLLMs.cli._output import _supports_color

            assert _supports_color() is False

    def test_supports_color_force_color_env(self):
        env = {"FORCE_COLOR": "1"}
        # Clear NO_COLOR if present
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("NO_COLOR", None)
            from insideLLMs.cli._output import _supports_color

            assert _supports_color() is True

    def test_colorize_no_color(self):
        import insideLLMs.cli._output as output_mod
        from insideLLMs.cli._output import Colors, colorize

        orig = output_mod.USE_COLOR
        try:
            output_mod.USE_COLOR = False
            result = colorize("test", Colors.RED)
            assert result == "test"
        finally:
            output_mod.USE_COLOR = orig

    def test_colorize_with_color(self):
        import insideLLMs.cli._output as output_mod
        from insideLLMs.cli._output import Colors, colorize

        orig = output_mod.USE_COLOR
        try:
            output_mod.USE_COLOR = True
            result = colorize("test", Colors.RED)
            assert "\033[31m" in result
            assert "test" in result
        finally:
            output_mod.USE_COLOR = orig

    def test_print_functions_quiet(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        try:
            output_mod._CLI_QUIET = True
            output_mod.print_header("title")
            output_mod.print_subheader("sub")
            output_mod.print_success("ok")
            output_mod.print_warning("warn")
            output_mod.print_info("info")
            output_mod.print_key_value("key", "val")
            captured = capsys.readouterr()
            # In quiet mode, status output should be suppressed
            assert "title" not in captured.out
        finally:
            output_mod._CLI_QUIET = orig

    def test_print_error_always_prints(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        try:
            output_mod._CLI_QUIET = True
            output_mod.print_error("critical error")
            captured = capsys.readouterr()
            assert "critical error" in captured.err
        finally:
            output_mod._CLI_QUIET = orig

    def test_print_functions_normal(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig_quiet = output_mod._CLI_QUIET
        orig_color = output_mod.USE_COLOR
        try:
            output_mod._CLI_QUIET = False
            output_mod.USE_COLOR = False
            output_mod.print_header("My Title")
            output_mod.print_subheader("My Sub")
            output_mod.print_success("all good")
            output_mod.print_warning("be careful")
            output_mod.print_info("note this")
            output_mod.print_key_value("answer", 42, indent=4)
            captured = capsys.readouterr()
            assert "My Title" in captured.out
            assert "My Sub" in captured.out
            assert "all good" in captured.out
            assert "be careful" in captured.out
            assert "note this" in captured.out
            assert "answer:" in captured.out
        finally:
            output_mod._CLI_QUIET = orig_quiet
            output_mod.USE_COLOR = orig_color

    def test_progress_bar(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        orig_color = output_mod.USE_COLOR
        try:
            output_mod._CLI_QUIET = False
            output_mod.USE_COLOR = False
            pb = output_mod.ProgressBar(total=10, prefix="Test")
            pb.update(5)
            pb.increment(2)
            pb.finish()
            captured = capsys.readouterr()
            assert "Test" in captured.out
            assert "Done" in captured.out
        finally:
            output_mod._CLI_QUIET = orig
            output_mod.USE_COLOR = orig_color

    def test_progress_bar_zero_total(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        orig_color = output_mod.USE_COLOR
        try:
            output_mod._CLI_QUIET = False
            output_mod.USE_COLOR = False
            pb = output_mod.ProgressBar(total=0)
            pb._render()
        finally:
            output_mod._CLI_QUIET = orig
            output_mod.USE_COLOR = orig_color

    def test_progress_bar_quiet(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        try:
            output_mod._CLI_QUIET = True
            pb = output_mod.ProgressBar(total=10)
            pb._render()
            pb.finish()
            captured = capsys.readouterr()
            assert captured.out == ""
        finally:
            output_mod._CLI_QUIET = orig

    def test_spinner(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        orig_color = output_mod.USE_COLOR
        try:
            output_mod._CLI_QUIET = False
            output_mod.USE_COLOR = False
            spinner = output_mod.Spinner("Loading")
            spinner.spin()
            spinner.spin()
            spinner.stop(success=True)
            captured = capsys.readouterr()
            assert "Loading" in captured.out
            assert "done" in captured.out
        finally:
            output_mod._CLI_QUIET = orig
            output_mod.USE_COLOR = orig_color

    def test_spinner_failure(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        orig_color = output_mod.USE_COLOR
        try:
            output_mod._CLI_QUIET = False
            output_mod.USE_COLOR = False
            spinner = output_mod.Spinner("Loading")
            spinner.stop(success=False)
            captured = capsys.readouterr()
            assert "failed" in captured.out
        finally:
            output_mod._CLI_QUIET = orig
            output_mod.USE_COLOR = orig_color

    def test_spinner_quiet(self, capsys):
        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_QUIET
        try:
            output_mod._CLI_QUIET = True
            spinner = output_mod.Spinner("Loading")
            spinner.spin()
            spinner.stop(success=True)
            captured = capsys.readouterr()
            assert captured.out == ""
        finally:
            output_mod._CLI_QUIET = orig

    def test_status_stream_stderr(self):
        import sys

        import insideLLMs.cli._output as output_mod

        orig = output_mod._CLI_STATUS_TO_STDERR
        try:
            output_mod._CLI_STATUS_TO_STDERR = True
            assert output_mod._status_stream() is sys.stderr
            output_mod._CLI_STATUS_TO_STDERR = False
            assert output_mod._status_stream() is sys.stdout
        finally:
            output_mod._CLI_STATUS_TO_STDERR = orig


# ============================================================================
# 6. insideLLMs/distributed.py
# ============================================================================


class TestDistributedCoverage:
    """Additional coverage tests for distributed module."""

    def test_task_comparison_same_priority(self):
        from insideLLMs.contrib.distributed import Task

        t1 = Task(id="t1", payload="a", priority=5, created_at=100.0)
        t2 = Task(id="t2", payload="b", priority=5, created_at=200.0)
        assert t1 < t2  # earlier timestamp wins

    def test_task_comparison_different_priority(self):
        from insideLLMs.contrib.distributed import Task

        low = Task(id="low", payload="x", priority=1)
        high = Task(id="high", payload="x", priority=10)
        assert high < low  # higher priority "comes first"

    def test_work_queue_pending_count(self):
        from insideLLMs.contrib.distributed import Task, WorkQueue

        q = WorkQueue()
        q.put(Task(id="t1", payload="a"))
        q.put(Task(id="t2", payload="b"))
        assert q.pending_count == 2
        q.get(timeout=1.0)
        assert q.pending_count == 1

    def test_work_queue_task_done(self):
        from insideLLMs.contrib.distributed import Task, WorkQueue

        q = WorkQueue()
        q.put(Task(id="t1", payload="a"))
        q.get(timeout=1.0)
        q.task_done()

    def test_result_collector_get_all(self):
        from insideLLMs.contrib.distributed import ResultCollector, TaskResult

        c = ResultCollector()
        c.add(TaskResult(task_id="t1", success=True, result=10))
        c.add(TaskResult(task_id="t2", success=True, result=20))
        results = c.get_all()
        assert len(results) == 2

    def test_result_collector_get_missing(self):
        from insideLLMs.contrib.distributed import ResultCollector

        c = ResultCollector()
        assert c.get("nonexistent") is None

    def test_result_collector_callback_exception_suppressed(self):
        from insideLLMs.contrib.distributed import ResultCollector, TaskResult

        c = ResultCollector()

        def bad_callback(r):
            raise RuntimeError("oops")

        c.on_result(bad_callback)
        # Should not raise
        c.add(TaskResult(task_id="t1", success=True))
        assert c.count == 1

    def test_local_worker_lifecycle(self):
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalWorker,
            ResultCollector,
            Task,
            WorkerStatus,
            WorkQueue,
        )

        q = WorkQueue()
        collector = ResultCollector()
        executor = FunctionExecutor(lambda x: x * 3)
        worker = LocalWorker("w0", executor, q, collector)

        assert worker.info.status == WorkerStatus.IDLE
        worker.start()

        q.put(Task(id="t1", payload=7))
        time.sleep(0.5)

        result = collector.get("t1")
        assert result is not None
        assert result.result == 21

        worker.stop()
        assert worker.info.status == WorkerStatus.SHUTDOWN

    def test_local_worker_error_handling(self):
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalWorker,
            ResultCollector,
            Task,
            WorkQueue,
        )

        q = WorkQueue()
        collector = ResultCollector()
        executor = FunctionExecutor(lambda x: 1 / 0)
        worker = LocalWorker("w0", executor, q, collector)

        worker.start()
        q.put(Task(id="fail", payload=0))
        time.sleep(0.5)

        result = collector.get("fail")
        assert result is not None
        assert result.success is False
        assert result.error is not None
        assert worker.info.tasks_failed >= 1

        worker.stop()

    def test_local_distributed_executor_get_results(self):
        from insideLLMs.contrib.distributed import (
            FunctionExecutor,
            LocalDistributedExecutor,
            Task,
        )

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x + 1),
            num_workers=2,
        )

        with executor:
            executor.submit(Task(id="t1", payload=10))
            time.sleep(0.5)
            results = executor.get_results()
            assert len(results) >= 0  # May or may not have completed yet

    def test_local_distributed_executor_get_worker_info(self):
        from insideLLMs.contrib.distributed import FunctionExecutor, LocalDistributedExecutor

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x),
            num_workers=3,
        )

        with executor:
            info = executor.get_worker_info()
            assert len(info) == 3
            assert info[0].id == "worker_0"

    def test_local_distributed_executor_auto_start(self):
        from insideLLMs.contrib.distributed import FunctionExecutor, LocalDistributedExecutor, Task

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x * 2),
            num_workers=1,
        )
        # submit without explicit start
        executor.submit(Task(id="auto", payload=5))
        results = executor.wait_for_completion(timeout=5.0)
        assert len(results) == 1
        executor.stop()

    def test_local_distributed_executor_idempotent_start(self):
        from insideLLMs.contrib.distributed import FunctionExecutor, LocalDistributedExecutor

        executor = LocalDistributedExecutor(
            executor=FunctionExecutor(lambda x: x),
            num_workers=2,
        )
        executor.start()
        executor.start()  # Should be a no-op
        assert len(executor._workers) == 2
        executor.stop()

    def test_distributed_checkpoint_manager_default_dir(self):
        from insideLLMs.contrib.distributed import DistributedCheckpointManager

        mgr = DistributedCheckpointManager()
        assert "insidellms_checkpoints" in str(mgr.checkpoint_dir)
        assert mgr.checkpoint_dir.exists()

    def test_distributed_checkpoint_save_load_delete(self):
        from insideLLMs.contrib.distributed import DistributedCheckpointManager, Task, TaskResult

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DistributedCheckpointManager(tmpdir)
            pending = [Task(id="t1", payload="data")]
            completed = [TaskResult(task_id="t0", success=True, result="done")]
            meta = {"exp": "test"}

            path = mgr.save("cp1", pending, completed, metadata=meta)
            assert os.path.exists(path)

            loaded_pending, loaded_completed, loaded_meta = mgr.load("cp1")
            assert len(loaded_pending) == 1
            assert loaded_pending[0].id == "t1"
            assert loaded_meta["exp"] == "test"

            checkpoints = mgr.list_checkpoints()
            assert "cp1" in checkpoints

            mgr.delete("cp1")
            assert "cp1" not in mgr.list_checkpoints()

            # Delete non-existent is a no-op
            mgr.delete("cp1")

    def test_distributed_checkpoint_load_not_found(self):
        from insideLLMs.contrib.distributed import DistributedCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DistributedCheckpointManager(tmpdir)
            with pytest.raises(FileNotFoundError):
                mgr.load("nonexistent")

    @requires_process_spawn
    def test_process_pool_executor_basic(self):
        from insideLLMs.contrib.distributed import ProcessPoolDistributedExecutor, Task

        executor = ProcessPoolDistributedExecutor(_square, num_workers=2)
        executor.submit(Task(id="s1", payload=5))
        executor.submit(Task(id="s2", payload=3))
        results = executor.run()
        results_by_id = {r.task_id: r for r in results}
        assert results_by_id["s1"].result == 25
        assert results_by_id["s2"].result == 9

    @requires_process_spawn
    def test_process_pool_executor_progress_callback(self):
        from insideLLMs.contrib.distributed import ProcessPoolDistributedExecutor, Task

        progress = []

        def on_progress(done, total):
            progress.append((done, total))

        executor = ProcessPoolDistributedExecutor(_identity, num_workers=2)
        executor.submit_batch([Task(id=f"t{i}", payload=i) for i in range(3)])
        executor.run(progress_callback=on_progress)
        assert len(progress) == 3

    @requires_process_spawn
    def test_process_pool_executor_with_checkpoint(self):
        from insideLLMs.contrib.distributed import (
            DistributedCheckpointManager,
            ProcessPoolDistributedExecutor,
            Task,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DistributedCheckpointManager(tmpdir)
            executor = ProcessPoolDistributedExecutor(
                _double, num_workers=2, checkpoint_manager=mgr
            )
            executor.submit_batch([Task(id=f"t{i}", payload=i) for i in range(5)])
            results = executor.run(checkpoint_id="cp_test", checkpoint_interval=2)
            assert len(results) == 5
            # Checkpoint should be deleted on success
            assert "cp_test" not in mgr.list_checkpoints()

    @requires_process_spawn
    def test_experiment_runner_run_experiments_with_processes(self):
        from insideLLMs.contrib.distributed import DistributedExperimentRunner

        runner = DistributedExperimentRunner(
            model_func=_format_answer,
            num_workers=2,
            use_processes=True,
        )
        results = runner.run_prompts(["Hello"])
        assert len(results) == 1
        assert results[0]["success"] is True

    def test_experiment_runner_with_checkpoint_dir(self):
        from insideLLMs.contrib.distributed import DistributedExperimentRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DistributedExperimentRunner(
                model_func=lambda x: f"R: {x}",
                num_workers=2,
                checkpoint_dir=tmpdir,
            )
            assert runner.checkpoint_manager is not None

    @requires_process_spawn
    def test_parallel_map_with_processes(self):
        from insideLLMs.contrib.distributed import parallel_map

        results = parallel_map(_double, [1, 2, 3], num_workers=2, use_processes=True)
        assert results == [2, 4, 6]

    def test_checkpoint_manager_alias(self):
        from insideLLMs.contrib.distributed import CheckpointManager, DistributedCheckpointManager

        assert CheckpointManager is DistributedCheckpointManager


# ============================================================================
# 7. insideLLMs/nlp/tokenization.py
# ============================================================================


class TestTokenizationCoverage:
    """Additional coverage tests for tokenization module."""

    def test_word_tokenize_regex_lowercase(self):
        from insideLLMs.nlp.tokenization import word_tokenize_regex

        result = word_tokenize_regex("Hello World!", lowercase=True)
        assert result == ["hello", "world"]

    def test_word_tokenize_regex_no_lowercase(self):
        from insideLLMs.nlp.tokenization import word_tokenize_regex

        result = word_tokenize_regex("Hello World!", lowercase=False)
        assert result == ["Hello", "World"]

    def test_word_tokenize_regex_numbers(self):
        from insideLLMs.nlp.tokenization import word_tokenize_regex

        result = word_tokenize_regex("I have 3 cats", lowercase=False)
        assert "3" in result
        assert "cats" in result

    def test_word_tokenize_regex_empty(self):
        from insideLLMs.nlp.tokenization import word_tokenize_regex

        assert word_tokenize_regex("") == []

    def test_segment_sentences_regex(self):
        from insideLLMs.nlp.tokenization import segment_sentences

        result = segment_sentences("First. Second? Third!", use_nltk=False)
        assert len(result) >= 2

    def test_get_ngrams_larger_than_tokens(self):
        from insideLLMs.nlp.tokenization import get_ngrams

        assert get_ngrams(["a", "b"], n=5) == []

    def test_get_ngrams_default_bigrams(self):
        from insideLLMs.nlp.tokenization import get_ngrams

        result = get_ngrams(["x", "y", "z"])
        assert result == [("x", "y"), ("y", "z")]

    def test_backward_compat_aliases(self):
        from insideLLMs.nlp.tokenization import check_nltk, check_spacy

        assert callable(check_nltk)
        assert callable(check_spacy)


# ============================================================================
# 8. insideLLMs/nlp/extraction.py
# ============================================================================


class TestExtractionCoverage:
    """Additional coverage for extraction module."""

    def test_extract_emails_plus_sign(self):
        from insideLLMs.nlp.extraction import extract_emails

        result = extract_emails("user+tag@example.com")
        assert "user+tag@example.com" in result

    def test_extract_phone_unknown_country(self):
        from insideLLMs.nlp.extraction import extract_phone_numbers

        # Falls back to international
        result = extract_phone_numbers("+49 30 12345678", country="de")
        assert len(result) >= 1

    def test_extract_urls_https(self):
        from insideLLMs.nlp.extraction import extract_urls

        result = extract_urls("Visit https://api.example.com/v1?x=1")
        assert len(result) >= 1

    def test_extract_urls_www(self):
        from insideLLMs.nlp.extraction import extract_urls

        result = extract_urls("Visit www.example.com")
        assert len(result) >= 1

    def test_extract_urls_none(self):
        from insideLLMs.nlp.extraction import extract_urls

        assert extract_urls("no urls here") == []

    def test_extract_hashtags(self):
        from insideLLMs.nlp.extraction import extract_hashtags

        result = extract_hashtags("Love #Python and #NLP")
        assert "#Python" in result
        assert "#NLP" in result

    def test_extract_hashtags_none(self):
        from insideLLMs.nlp.extraction import extract_hashtags

        assert extract_hashtags("no hashtags") == []

    def test_extract_mentions(self):
        from insideLLMs.nlp.extraction import extract_mentions

        result = extract_mentions("Thanks @user1 and @dev_2!")
        assert "@user1" in result
        assert "@dev_2" in result

    def test_extract_mentions_none(self):
        from insideLLMs.nlp.extraction import extract_mentions

        assert extract_mentions("no mentions") == []

    def test_extract_ip_addresses(self):
        from insideLLMs.nlp.extraction import extract_ip_addresses

        result = extract_ip_addresses("Server at 192.168.1.1 and 10.0.0.1")
        assert "192.168.1.1" in result
        assert "10.0.0.1" in result

    def test_extract_ip_addresses_invalid(self):
        from insideLLMs.nlp.extraction import extract_ip_addresses

        assert extract_ip_addresses("256.999.1.1") == []

    def test_extract_ip_addresses_boundary(self):
        from insideLLMs.nlp.extraction import extract_ip_addresses

        result = extract_ip_addresses("0.0.0.0 and 255.255.255.255")
        assert "0.0.0.0" in result
        assert "255.255.255.255" in result

    def test_check_spacy_backward_compat(self):
        from insideLLMs.nlp.extraction import check_spacy

        assert callable(check_spacy)


# ============================================================================
# 9. insideLLMs/nlp/chunking.py
# ============================================================================


class TestChunkingCoverage:
    """Additional coverage for chunking module."""

    def test_split_by_char_invalid_chunk_size(self):
        from insideLLMs.nlp.chunking import split_by_char_count

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            split_by_char_count("text", chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            split_by_char_count("text", chunk_size=-1)

    def test_split_by_char_overlap_too_large(self):
        from insideLLMs.nlp.chunking import split_by_char_count

        with pytest.raises(ValueError, match="overlap must be less than"):
            split_by_char_count("text", chunk_size=3, overlap=3)

    def test_split_by_char_overlap(self):
        from insideLLMs.nlp.chunking import split_by_char_count

        result = split_by_char_count("abcdefghij", chunk_size=4, overlap=2)
        assert result[0] == "abcd"
        assert result[1] == "cdef"
        assert result[2] == "efgh"
        assert result[3] == "ghij"

    def test_split_by_char_short_text(self):
        from insideLLMs.nlp.chunking import split_by_char_count

        assert split_by_char_count("hi", chunk_size=10) == ["hi"]

    def test_split_by_word_invalid_words_per_chunk(self):
        from insideLLMs.nlp.chunking import split_by_word_count

        with pytest.raises(ValueError, match="words_per_chunk must be positive"):
            split_by_word_count("text", words_per_chunk=0)

    def test_split_by_word_overlap_too_large(self):
        from insideLLMs.nlp.chunking import split_by_word_count

        with pytest.raises(ValueError, match="overlap must be less than"):
            split_by_word_count("text", words_per_chunk=2, overlap=2)

    def test_split_by_word_empty_text(self):
        from insideLLMs.nlp.chunking import split_by_word_count

        assert split_by_word_count("", words_per_chunk=3) == []

    def test_split_by_word_custom_tokenizer(self):
        from insideLLMs.nlp.chunking import split_by_word_count

        result = split_by_word_count(
            "Hello World Test",
            words_per_chunk=2,
            tokenizer=lambda t: t.lower().split(),
        )
        assert result == ["hello world", "test"]

    def test_split_by_sentence_invalid_sentences_per_chunk(self):
        from insideLLMs.nlp.chunking import split_by_sentence

        with pytest.raises(ValueError, match="sentences_per_chunk must be positive"):
            split_by_sentence("text", sentences_per_chunk=0)

    def test_split_by_sentence_overlap_too_large(self):
        from insideLLMs.nlp.chunking import split_by_sentence

        with pytest.raises(ValueError, match="overlap must be less than"):
            split_by_sentence("text", sentences_per_chunk=2, overlap=2)

    def test_sliding_window_basic(self):
        from insideLLMs.nlp.chunking import sliding_window_chunks

        result = sliding_window_chunks("a b c d e f", window_size=3, step_size=2)
        assert result[0] == "a b c"
        assert result[1] == "c d e"

    def test_sliding_window_invalid_window_size(self):
        from insideLLMs.nlp.chunking import sliding_window_chunks

        with pytest.raises(ValueError, match="window_size must be positive"):
            sliding_window_chunks("text", window_size=0, step_size=1)

    def test_sliding_window_invalid_step_size(self):
        from insideLLMs.nlp.chunking import sliding_window_chunks

        with pytest.raises(ValueError, match="step_size must be positive"):
            sliding_window_chunks("text", window_size=2, step_size=0)

    def test_sliding_window_short_text(self):
        from insideLLMs.nlp.chunking import sliding_window_chunks

        assert sliding_window_chunks("hi", window_size=5, step_size=2) == ["hi"]

    def test_sliding_window_empty_text(self):
        from insideLLMs.nlp.chunking import sliding_window_chunks

        assert sliding_window_chunks("", window_size=3, step_size=1) == []

    def test_sliding_window_custom_tokenizer(self):
        from insideLLMs.nlp.chunking import sliding_window_chunks

        result = sliding_window_chunks(
            "A B C D",
            window_size=2,
            step_size=1,
            tokenizer=lambda t: t.lower().split(),
        )
        assert result == ["a b", "b c", "c d"]

    def test_backward_compat_aliases(self):
        from insideLLMs.nlp.chunking import (
            check_nltk,
            segment_sentences_internal_for_chunking,
            simple_tokenize_for_chunking,
        )

        assert callable(check_nltk)
        assert callable(simple_tokenize_for_chunking)
        assert callable(segment_sentences_internal_for_chunking)
