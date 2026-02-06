"""Comprehensive tests to increase coverage for:
- insideLLMs/runtime/_sync_runner.py
- insideLLMs/runtime/_result_utils.py
- insideLLMs/runtime/_artifact_utils.py
- insideLLMs/results.py

Focuses on uncovered code paths not exercised by existing test files.
"""

import json
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from insideLLMs.types import (
    BenchmarkComparison,
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_info():
    return ModelInfo(name="TestModel", provider="TestProvider", model_id="test-v1")


def _make_experiment(
    *,
    score=None,
    results=None,
    started_at=None,
    completed_at=None,
):
    if results is None:
        results = [
            ProbeResult(
                input="test input",
                output="test output",
                status=ResultStatus.SUCCESS,
                latency_ms=100.0,
            )
        ]
    return ExperimentResult(
        experiment_id="exp-001",
        model_info=_make_model_info(),
        probe_name="test_probe",
        probe_category=ProbeCategory.LOGIC,
        results=results,
        score=score,
        started_at=started_at,
        completed_at=completed_at,
    )


# ===================================================================
# _artifact_utils.py coverage
# ===================================================================


class TestTruncateIncompleteJsonl:
    """Tests for _truncate_incomplete_jsonl."""

    def test_empty_file(self, tmp_path):
        """Empty file is left unchanged."""
        from insideLLMs.runtime._artifact_utils import _truncate_incomplete_jsonl

        p = tmp_path / "empty.jsonl"
        p.write_bytes(b"")
        _truncate_incomplete_jsonl(p)
        assert p.read_bytes() == b""

    def test_file_ending_with_newline(self, tmp_path):
        """File that already ends with newline is left unchanged."""
        from insideLLMs.runtime._artifact_utils import _truncate_incomplete_jsonl

        content = b'{"a":1}\n{"b":2}\n'
        p = tmp_path / "complete.jsonl"
        p.write_bytes(content)
        _truncate_incomplete_jsonl(p)
        assert p.read_bytes() == content

    def test_file_with_incomplete_last_line(self, tmp_path):
        """Incomplete last line is removed."""
        from insideLLMs.runtime._artifact_utils import _truncate_incomplete_jsonl

        p = tmp_path / "incomplete.jsonl"
        p.write_bytes(b'{"a":1}\n{"incomplete":')
        _truncate_incomplete_jsonl(p)
        assert p.read_bytes() == b'{"a":1}\n'

    def test_file_no_newline_at_all(self, tmp_path):
        """File with no newlines at all is truncated to empty."""
        from insideLLMs.runtime._artifact_utils import _truncate_incomplete_jsonl

        p = tmp_path / "noline.jsonl"
        p.write_bytes(b'{"incomplete":')
        _truncate_incomplete_jsonl(p)
        assert p.read_bytes() == b""


class TestReadJsonlRecords:
    """Tests for _read_jsonl_records."""

    def test_nonexistent_file(self, tmp_path):
        """Returns empty list for non-existent file."""
        from insideLLMs.runtime._artifact_utils import _read_jsonl_records

        result = _read_jsonl_records(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_read_with_truncate_incomplete(self, tmp_path):
        """truncate_incomplete=True removes incomplete last line before reading."""
        from insideLLMs.runtime._artifact_utils import _read_jsonl_records

        p = tmp_path / "records.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n{"incomplete":', encoding="utf-8")
        records = _read_jsonl_records(p, truncate_incomplete=True)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["b"] == 2

    def test_skips_empty_lines(self, tmp_path):
        """Empty lines are silently skipped."""
        from insideLLMs.runtime._artifact_utils import _read_jsonl_records

        p = tmp_path / "records.jsonl"
        p.write_text('{"a":1}\n\n{"b":2}\n\n', encoding="utf-8")
        records = _read_jsonl_records(p)
        assert len(records) == 2

    def test_skips_non_dict_json(self, tmp_path):
        """Non-dict JSON values are skipped."""
        from insideLLMs.runtime._artifact_utils import _read_jsonl_records

        p = tmp_path / "records.jsonl"
        p.write_text('{"a":1}\n42\n"string"\n[1,2,3]\n{"b":2}\n', encoding="utf-8")
        records = _read_jsonl_records(p)
        assert len(records) == 2

    def test_invalid_json_raises(self, tmp_path):
        """Invalid JSON raises ValueError."""
        from insideLLMs.runtime._artifact_utils import _read_jsonl_records

        p = tmp_path / "records.jsonl"
        p.write_text('{"a":1}\n{invalid json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSONL record"):
            _read_jsonl_records(p)


class TestValidateResumeRecord:
    """Tests for _validate_resume_record."""

    def test_index_mismatch_raises(self):
        """Non-contiguous index raises ValueError."""
        from insideLLMs.runtime._artifact_utils import _validate_resume_record

        record = {"custom": {"record_index": 5}, "input": "test"}
        with pytest.raises(ValueError, match="not a contiguous prefix"):
            _validate_resume_record(
                record, expected_index=0, expected_item="test", run_id=None
            )

    def test_run_id_mismatch_raises(self):
        """Mismatched run_id raises ValueError."""
        from insideLLMs.runtime._artifact_utils import _validate_resume_record

        record = {
            "custom": {"record_index": 0},
            "run_id": "old-run",
            "input": "test",
        }
        with pytest.raises(ValueError, match="does not match current run_id"):
            _validate_resume_record(
                record,
                expected_index=0,
                expected_item="test",
                run_id="new-run",
            )

    def test_input_fingerprint_mismatch_raises(self):
        """Mismatched input fingerprint raises ValueError."""
        from insideLLMs.runtime._artifact_utils import _validate_resume_record

        record = {
            "custom": {"record_index": 0},
            "input": "original item",
        }
        with pytest.raises(ValueError, match="input mismatch"):
            _validate_resume_record(
                record,
                expected_index=0,
                expected_item="different item",
                run_id=None,
            )

    def test_matching_record_passes(self):
        """Valid record passes validation without error."""
        from insideLLMs.runtime._artifact_utils import _validate_resume_record

        record = {
            "custom": {"record_index": 0},
            "run_id": "test-run",
            "input": "test item",
        }
        # Should not raise
        _validate_resume_record(
            record,
            expected_index=0,
            expected_item="test item",
            run_id="test-run",
        )

    def test_run_id_none_in_record_skips_check(self):
        """If record has no run_id, the check is skipped."""
        from insideLLMs.runtime._artifact_utils import _validate_resume_record

        record = {
            "custom": {"record_index": 0},
            "input": "test item",
        }
        # Should not raise even with a different run_id
        _validate_resume_record(
            record,
            expected_index=0,
            expected_item="test item",
            run_id="any-run",
        )

    def test_strict_serialization_error_in_fingerprinting(self):
        """strict_serialization with non-serializable prompts raises ValueError."""
        from insideLLMs.runtime._artifact_utils import _validate_resume_record

        class NotSerializable:
            pass

        record = {
            "custom": {"record_index": 0},
            "input": "ok",
        }
        with pytest.raises(ValueError, match="strict_serialization"):
            _validate_resume_record(
                record,
                expected_index=0,
                expected_item=NotSerializable(),
                run_id=None,
                strict_serialization=True,
            )


class TestPrepareRunDirForResume:
    """Tests for _prepare_run_dir_for_resume."""

    def test_creates_new_directory(self, tmp_path):
        """Creates directory if it doesn't exist."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir_for_resume

        run_dir = tmp_path / "new_resume_dir"
        _prepare_run_dir_for_resume(run_dir)
        assert run_dir.exists()

    def test_uses_existing_empty_directory(self, tmp_path):
        """Uses existing empty directory without error."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir_for_resume

        run_dir = tmp_path / "empty_dir"
        run_dir.mkdir()
        _prepare_run_dir_for_resume(run_dir)
        assert run_dir.exists()

    def test_allows_directory_with_sentinel(self, tmp_path):
        """Non-empty directory with sentinel is allowed."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir_for_resume

        run_dir = tmp_path / "sentinel_dir"
        run_dir.mkdir()
        (run_dir / ".insidellms_run").write_text("marker")
        (run_dir / "records.jsonl").write_text("{}\n")
        _prepare_run_dir_for_resume(run_dir)

    def test_allows_directory_with_manifest(self, tmp_path):
        """Non-empty directory with manifest.json is allowed."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir_for_resume

        run_dir = tmp_path / "manifest_dir"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{}")
        _prepare_run_dir_for_resume(run_dir)

    def test_rejects_non_empty_without_sentinel(self, tmp_path):
        """Non-empty directory without sentinel is rejected."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir_for_resume

        run_dir = tmp_path / "no_sentinel"
        run_dir.mkdir()
        (run_dir / "random.txt").write_text("content")
        with pytest.raises(ValueError, match="does not look like an insideLLMs run"):
            _prepare_run_dir_for_resume(run_dir)

    def test_rejects_file_not_directory(self, tmp_path):
        """Path that is a file, not directory, is rejected."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir_for_resume

        file_path = tmp_path / "is_a_file"
        file_path.write_text("I am a file")
        with pytest.raises(FileExistsError, match="is not a directory"):
            _prepare_run_dir_for_resume(file_path)


class TestPrepareRunDirSafetyGuards:
    """Additional safety guard tests for _prepare_run_dir."""

    def test_overwrite_with_records_sentinel(self, tmp_path):
        """Overwrite with records.jsonl sentinel."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir

        run_dir = tmp_path / "records_sentinel"
        run_dir.mkdir()
        (run_dir / "records.jsonl").write_text("{}\n")
        (run_dir / "old.txt").write_text("old")

        _prepare_run_dir(run_dir, overwrite=True, run_root=tmp_path)
        assert run_dir.exists()
        assert not (run_dir / "old.txt").exists()

    def test_overwrite_with_config_resolved_sentinel(self, tmp_path):
        """Overwrite with config.resolved.yaml sentinel."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir

        run_dir = tmp_path / "config_sentinel"
        run_dir.mkdir()
        (run_dir / "config.resolved.yaml").write_text("model: test\n")
        (run_dir / "old.txt").write_text("old")

        _prepare_run_dir(run_dir, overwrite=True, run_root=tmp_path)
        assert run_dir.exists()
        assert not (run_dir / "old.txt").exists()

    def test_refuse_overwrite_run_root_itself(self, tmp_path):
        """Refuse to overwrite the run_root directory itself."""
        from insideLLMs.runtime._artifact_utils import _prepare_run_dir

        run_root = tmp_path / "runs"
        run_root.mkdir()
        (run_root / "manifest.json").write_text("{}")

        with pytest.raises(ValueError, match="Refusing to overwrite run_root"):
            _prepare_run_dir(run_root, overwrite=True, run_root=run_root)


class TestAtomicWriteYaml:
    """Tests for _atomic_write_yaml."""

    def test_basic_yaml_write(self, tmp_path):
        """Basic YAML write works correctly."""
        from insideLLMs.runtime._artifact_utils import _atomic_write_yaml

        path = tmp_path / "config.yaml"
        data = {"model": {"type": "openai"}, "probe": {"type": "logic"}}
        _atomic_write_yaml(path, data)

        loaded = yaml.safe_load(path.read_text())
        assert loaded["model"]["type"] == "openai"

    def test_yaml_write_with_strict_serialization(self, tmp_path):
        """Strict serialization mode serializes values before YAML dump."""
        from insideLLMs.runtime._artifact_utils import _atomic_write_yaml

        path = tmp_path / "strict.yaml"
        data = {"key": "value", "num": 42}
        _atomic_write_yaml(path, data, strict_serialization=True)

        loaded = yaml.safe_load(path.read_text())
        assert loaded["key"] == "value"


# ===================================================================
# _result_utils.py coverage
# ===================================================================


class TestNormalizeStatusExtended:
    """Extended tests for _normalize_status."""

    def test_normalize_result_status_success(self):
        from insideLLMs.runtime._result_utils import _normalize_status

        assert _normalize_status(ResultStatus.SUCCESS) == "success"

    def test_normalize_result_status_error(self):
        from insideLLMs.runtime._result_utils import _normalize_status

        assert _normalize_status(ResultStatus.ERROR) == "error"

    def test_normalize_none(self):
        from insideLLMs.runtime._result_utils import _normalize_status

        assert _normalize_status(None) == "error"

    def test_normalize_string(self):
        from insideLLMs.runtime._result_utils import _normalize_status

        assert _normalize_status("custom") == "custom"

    def test_normalize_non_result_status_enum(self):
        """Non-ResultStatus Enum should use its value."""
        from insideLLMs.runtime._result_utils import _normalize_status

        class CustomStatus(Enum):
            PENDING = "pending"
            DONE = "done"

        assert _normalize_status(CustomStatus.PENDING) == "pending"
        assert _normalize_status(CustomStatus.DONE) == "done"

    def test_normalize_integer(self):
        """Integer is stringified."""
        from insideLLMs.runtime._result_utils import _normalize_status

        assert _normalize_status(42) == "42"


class TestRecordIndexFromRecord:
    """Tests for _record_index_from_record."""

    def test_valid_index(self):
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"custom": {"record_index": 5}}
        assert _record_index_from_record(record) == 5

    def test_missing_custom(self):
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"input": "test"}
        assert _record_index_from_record(record) is None

    def test_missing_custom_with_default(self):
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"input": "test"}
        assert _record_index_from_record(record, default=0) == 0

    def test_non_dict_custom(self):
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"custom": "not a dict"}
        assert _record_index_from_record(record) is None

    def test_non_integer_index(self):
        """Non-parseable index returns default."""
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"custom": {"record_index": "not_a_number"}}
        assert _record_index_from_record(record, default=-1) == -1

    def test_string_integer_index(self):
        """String that can be parsed as int works."""
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"custom": {"record_index": "7"}}
        assert _record_index_from_record(record) == 7

    def test_none_index(self):
        """None index returns default."""
        from insideLLMs.runtime._result_utils import _record_index_from_record

        record = {"custom": {"record_index": None}}
        assert _record_index_from_record(record, default=3) == 3


class TestResultDictFromProbeResult:
    """Tests for _result_dict_from_probe_result."""

    def test_success_result(self):
        from insideLLMs.runtime._result_utils import _result_dict_from_probe_result

        result = ProbeResult(
            input="test",
            output="response",
            status=ResultStatus.SUCCESS,
            latency_ms=150.5,
            metadata={"key": "val"},
        )
        d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        assert d["status"] == "success"
        assert d["latency_ms"] == 150.5
        assert d["metadata"] == {"key": "val"}
        assert "error" not in d
        assert "error_type" not in d

    def test_error_result_with_metadata_error_type(self):
        from insideLLMs.runtime._result_utils import _result_dict_from_probe_result

        result = ProbeResult(
            input="test",
            status=ResultStatus.ERROR,
            error="API timeout",
            metadata={"error_type": "TimeoutError"},
        )
        d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        assert d["error"] == "API timeout"
        assert d["error_type"] == "TimeoutError"

    def test_error_result_with_explicit_error_type(self):
        from insideLLMs.runtime._result_utils import _result_dict_from_probe_result

        result = ProbeResult(
            input="test",
            status=ResultStatus.ERROR,
            error="boom",
            metadata={},
        )
        d = _result_dict_from_probe_result(
            result, schema_version="1.0.0", error_type="ValueError"
        )
        assert d["error_type"] == "ValueError"

    def test_non_dict_metadata(self):
        """Non-dict metadata is treated as empty dict."""
        from insideLLMs.runtime._result_utils import _result_dict_from_probe_result

        result = ProbeResult(
            input="test",
            output="ok",
            status=ResultStatus.SUCCESS,
            metadata="not a dict",
        )
        d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        assert d["metadata"] == {}

    def test_no_error_no_error_type(self):
        """Result with no error and no error_type omits those keys."""
        from insideLLMs.runtime._result_utils import _result_dict_from_probe_result

        result = ProbeResult(
            input="test",
            output="ok",
            status=ResultStatus.SUCCESS,
        )
        d = _result_dict_from_probe_result(result, schema_version="1.0.0")
        assert "error" not in d
        assert "error_type" not in d


class TestResultDictFromRecord:
    """Tests for _result_dict_from_record."""

    def test_success_record(self):
        from insideLLMs.runtime._result_utils import _result_dict_from_record

        record = {
            "schema_version": "1.0.0",
            "input": "test",
            "output": "response",
            "latency_ms": 100.0,
            "status": "success",
        }
        d = _result_dict_from_record(record, schema_version="1.0.0")
        assert d["status"] == "success"
        assert d["input"] == "test"
        assert d["metadata"] == {}
        assert "error" not in d

    def test_error_record(self):
        from insideLLMs.runtime._result_utils import _result_dict_from_record

        record = {
            "input": "test",
            "status": "error",
            "error": "Connection failed",
            "error_type": "ConnectionError",
        }
        d = _result_dict_from_record(record, schema_version="1.0.0")
        assert d["error"] == "Connection failed"
        assert d["error_type"] == "ConnectionError"

    def test_fallback_schema_version(self):
        """Uses fallback schema_version if not in record."""
        from insideLLMs.runtime._result_utils import _result_dict_from_record

        record = {"input": "test", "status": "success"}
        d = _result_dict_from_record(record, schema_version="2.0.0")
        assert d["schema_version"] == "2.0.0"

    def test_record_schema_version_takes_priority(self):
        """Record's own schema_version takes priority."""
        from insideLLMs.runtime._result_utils import _result_dict_from_record

        record = {
            "schema_version": "1.5.0",
            "input": "test",
            "status": "success",
        }
        d = _result_dict_from_record(record, schema_version="2.0.0")
        assert d["schema_version"] == "1.5.0"


class TestBuildDatasetSpecExtended:
    """Extended tests for _build_dataset_spec."""

    def test_with_revision_and_version(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec(
            {"name": "ds", "version": "1.0", "revision": "main"}
        )
        assert spec["dataset_version"] == "1.0@main"

    def test_with_revision_only(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec({"name": "ds", "revision": "abc123"})
        assert spec["dataset_version"] == "abc123"

    def test_with_split_and_version(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec(
            {"name": "ds", "version": "1.0", "split": "train"}
        )
        assert spec["dataset_version"] == "1.0::train"

    def test_with_split_only(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec({"name": "ds", "split": "test"})
        assert spec["dataset_version"] == "test"

    def test_with_revision_version_and_split(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec(
            {"name": "ds", "version": "1.0", "revision": "main", "split": "train"}
        )
        assert spec["dataset_version"] == "1.0@main::train"

    def test_with_hash(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec({"name": "ds", "hash": "sha256:abc"})
        assert spec["dataset_hash"] == "sha256:abc"

    def test_with_dataset_hash(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec({"name": "ds", "dataset_hash": "sha256:xyz"})
        assert spec["dataset_hash"] == "sha256:xyz"

    def test_with_provenance_sources(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        assert _build_dataset_spec({"provenance": "hf"})["provenance"] == "hf"
        assert _build_dataset_spec({"source": "local"})["provenance"] == "local"
        assert _build_dataset_spec({"format": "jsonl"})["provenance"] == "jsonl"

    def test_with_dataset_key(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec({"dataset": "my-dataset"})
        assert spec["dataset_id"] == "my-dataset"

    def test_with_path_key(self):
        from insideLLMs.runtime._result_utils import _build_dataset_spec

        spec = _build_dataset_spec({"path": "/data/test.jsonl"})
        assert spec["dataset_id"] == "/data/test.jsonl"


class TestBuildModelSpecExtended:
    """Extended tests for _build_model_spec."""

    def test_model_with_no_info_method(self):
        """Model without info() falls back to class name."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class SimpleModel:
            pass

        spec = _build_model_spec(SimpleModel())
        assert spec["model_id"] == "SimpleModel"
        assert spec["provider"] is None

    def test_model_with_info_returning_none(self):
        """Model whose info() returns None."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class NoneInfoModel:
            def info(self):
                return None

        spec = _build_model_spec(NoneInfoModel())
        assert spec["model_id"] == "NoneInfoModel"

    def test_model_info_raises_exception(self):
        """Model whose info() raises falls back gracefully."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class BrokenModel:
            def info(self):
                raise TypeError("broken")

        spec = _build_model_spec(BrokenModel())
        assert spec["model_id"] == "BrokenModel"

    def test_model_with_various_id_keys(self):
        """Various info dict keys for model_id."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class M1:
            def info(self):
                return {"id": "from-id"}

        assert _build_model_spec(M1())["model_id"] == "from-id"

        class M2:
            def info(self):
                return {"name": "from-name"}

        assert _build_model_spec(M2())["model_id"] == "from-name"

        class M3:
            def info(self):
                return {"model_name": "from-model-name"}

        assert _build_model_spec(M3())["model_id"] == "from-model-name"

    def test_model_with_attribute_fallback(self):
        """Model with model_id attribute used as fallback."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class AttrModel:
            model_id = "attr-model-id"

            def info(self):
                return {}

        spec = _build_model_spec(AttrModel())
        assert spec["model_id"] == "attr-model-id"

    def test_provider_from_type_key(self):
        """Provider from 'type' key in info dict."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class M:
            def info(self):
                return {"model_id": "m", "type": "openai"}

        spec = _build_model_spec(M())
        assert spec["provider"] == "openai"

    def test_provider_from_model_type_key(self):
        """Provider from 'model_type' key in info dict."""
        from insideLLMs.runtime._result_utils import _build_model_spec

        class M:
            def info(self):
                return {"model_id": "m", "model_type": "anthropic"}

        spec = _build_model_spec(M())
        assert spec["provider"] == "anthropic"


class TestBuildProbeSpecExtended:
    """Extended tests for _build_probe_spec."""

    def test_probe_with_name(self):
        from insideLLMs.runtime._result_utils import _build_probe_spec

        class P:
            name = "my_probe"
            version = "2.0.0"

        spec = _build_probe_spec(P())
        assert spec["probe_id"] == "my_probe"
        assert spec["probe_version"] == "2.0.0"

    def test_probe_with_probe_id(self):
        from insideLLMs.runtime._result_utils import _build_probe_spec

        class P:
            probe_id = "probe-abc"

        spec = _build_probe_spec(P())
        assert spec["probe_id"] == "probe-abc"

    def test_probe_with_no_attributes(self):
        from insideLLMs.runtime._result_utils import _build_probe_spec

        class P:
            pass

        spec = _build_probe_spec(P())
        assert spec["probe_id"] == "P"
        assert spec["probe_version"] is None

    def test_probe_with_probe_version(self):
        from insideLLMs.runtime._result_utils import _build_probe_spec

        class P:
            name = "test"
            probe_version = "3.0.0"

        spec = _build_probe_spec(P())
        assert spec["probe_version"] == "3.0.0"


class TestCoerceModelInfoExtended:
    """Extended tests for _coerce_model_info."""

    def test_extra_fields_propagated(self):
        """Extra fields from info dict propagated to extra."""
        from insideLLMs.runtime._result_utils import _coerce_model_info

        class M:
            def info(self):
                return {
                    "name": "test",
                    "provider": "prov",
                    "model_id": "mid",
                    "custom_field": "custom_val",
                    "another": 42,
                }

        info = _coerce_model_info(M())
        assert info.extra["custom_field"] == "custom_val"
        assert info.extra["another"] == 42

    def test_extra_dict_in_info(self):
        """Extra dict from info is merged."""
        from insideLLMs.runtime._result_utils import _coerce_model_info

        class M:
            def info(self):
                return {
                    "name": "test",
                    "extra": {"existing_key": "existing_val"},
                    "new_key": "new_val",
                }

        info = _coerce_model_info(M())
        assert info.extra["existing_key"] == "existing_val"
        assert info.extra["new_key"] == "new_val"

    def test_model_without_info(self):
        """Model with no info() method."""
        from insideLLMs.runtime._result_utils import _coerce_model_info

        class NoInfoModel:
            pass

        info = _coerce_model_info(NoInfoModel())
        assert info.name == "NoInfoModel"
        assert info.provider == "NoInfo"  # strips "Model" from class name

    def test_model_info_returns_none(self):
        """Model whose info() returns None."""
        from insideLLMs.runtime._result_utils import _coerce_model_info

        class NoneModel:
            def info(self):
                return None

        info = _coerce_model_info(NoneModel())
        assert info.name == "NoneModel"

    def test_model_with_supports_flags(self):
        """supports_streaming and supports_chat flags."""
        from insideLLMs.runtime._result_utils import _coerce_model_info

        class M:
            def info(self):
                return {
                    "name": "test",
                    "supports_streaming": True,
                    "supports_chat": False,
                }

        info = _coerce_model_info(M())
        assert info.supports_streaming is True
        assert info.supports_chat is False


class TestBuildResultRecordExtended:
    """Extended tests for _build_result_record."""

    def _common_kwargs(self):
        return dict(
            schema_version="1.0.0",
            run_id="test-run",
            started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            model={"model_id": "test-model"},
            probe={"probe_id": "test-probe"},
            dataset={"dataset_id": "test-dataset"},
        )

    def test_string_error(self):
        """String error is handled correctly."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="test input",
            output=None,
            latency_ms=None,
            store_messages=False,
            index=0,
            status="error",
            error="string error message",
            error_type="CustomError",
        )
        assert record["error"] == "string error message"
        assert record["error_type"] == "CustomError"

    def test_none_error_with_error_type(self):
        """None error with error_type still sets error_type."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="test input",
            output="ok",
            latency_ms=100.0,
            store_messages=False,
            index=0,
            status="success",
            error=None,
            error_type="SomeType",
        )
        assert record["error"] is None
        assert record["error_type"] == "SomeType"

    def test_messages_without_store(self):
        """Messages not stored when store_messages=False."""
        from insideLLMs.runtime._result_utils import _build_result_record

        item = {"messages": [{"role": "user", "content": "hello"}]}
        record = _build_result_record(
            **self._common_kwargs(),
            item=item,
            output="response",
            latency_ms=100.0,
            store_messages=False,
            index=0,
            status="success",
            error=None,
        )
        assert record["messages"] is None
        assert record["messages_hash"] is not None  # hash is still computed

    def test_non_dict_messages(self):
        """Non-dict items in messages list are normalized."""
        from insideLLMs.runtime._result_utils import _build_result_record

        item = {"messages": ["plain text message", {"role": "assistant", "content": "hi"}]}
        record = _build_result_record(
            **self._common_kwargs(),
            item=item,
            output="response",
            latency_ms=100.0,
            store_messages=True,
            index=0,
            status="success",
            error=None,
        )
        assert record["messages"][0]["role"] == "user"
        assert record["messages"][0]["content"] == "plain text message"
        assert record["messages"][1]["role"] == "assistant"

    def test_dict_output_with_score_key(self):
        """Dict output with 'score' key (not 'scores')."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="test",
            output={"score": 0.95, "text": "output text"},
            latency_ms=100.0,
            store_messages=False,
            index=0,
            status="success",
            error=None,
        )
        assert record["scores"] == {"score": 0.95}
        assert record["output_text"] == "output text"

    def test_output_fingerprint_for_non_string_output(self):
        """Non-string output gets a fingerprint."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="test",
            output={"key": "value"},
            latency_ms=100.0,
            store_messages=False,
            index=0,
            status="success",
            error=None,
        )
        assert record["custom"]["output_fingerprint"] is not None

    def test_string_output_no_fingerprint(self):
        """String output does not get a separate fingerprint."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="test",
            output="plain string",
            latency_ms=100.0,
            store_messages=False,
            index=0,
            status="success",
            error=None,
        )
        assert record["custom"]["output_fingerprint"] is None

    def test_item_with_id_key(self):
        """Item dict with 'id' key used as example_id."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item={"id": "item-42"},
            output="ok",
            latency_ms=None,
            store_messages=False,
            index=5,
            status="success",
            error=None,
        )
        assert record["example_id"] == "item-42"

    def test_non_dict_item_uses_index(self):
        """Non-dict item uses index as example_id."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="just a string",
            output="ok",
            latency_ms=None,
            store_messages=False,
            index=7,
            status="success",
            error=None,
        )
        assert record["example_id"] == "7"

    def test_messages_hash_computed_for_message_input(self):
        """Messages hash is computed when messages are present."""
        from insideLLMs.runtime._result_utils import _build_result_record

        item = {"messages": [{"role": "user", "content": "hello"}]}
        record = _build_result_record(
            **self._common_kwargs(),
            item=item,
            output="ok",
            latency_ms=None,
            store_messages=True,
            index=0,
            status="success",
            error=None,
        )
        assert record["messages_hash"] is not None
        assert len(record["messages_hash"]) == 64  # SHA-256 hex

    def test_message_without_role(self):
        """Message without role defaults to 'user'."""
        from insideLLMs.runtime._result_utils import _build_result_record

        item = {"messages": [{"content": "no role here"}]}
        record = _build_result_record(
            **self._common_kwargs(),
            item=item,
            output="ok",
            latency_ms=None,
            store_messages=True,
            index=0,
            status="success",
            error=None,
        )
        assert record["messages"][0]["role"] == "user"

    def test_dict_output_with_usage(self):
        """Dict output with usage field."""
        from insideLLMs.runtime._result_utils import _build_result_record

        record = _build_result_record(
            **self._common_kwargs(),
            item="test",
            output={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            latency_ms=None,
            store_messages=False,
            index=0,
            status="success",
            error=None,
        )
        assert record["usage"]["prompt_tokens"] == 10


class TestNormalizeInfoObjToDictExtended:
    """Extended tests for _normalize_info_obj_to_dict."""

    def test_non_type_class_not_treated_as_dataclass(self):
        """The dataclass type itself (not instance) returns empty dict."""
        from insideLLMs.runtime._result_utils import _normalize_info_obj_to_dict

        from dataclasses import dataclass

        @dataclass
        class Info:
            name: str

        # Passing the class, not an instance
        result = _normalize_info_obj_to_dict(Info)
        assert result == {}

    def test_unknown_type_returns_empty_dict(self):
        """Unknown types without dict/model_dump return empty dict."""
        from insideLLMs.runtime._result_utils import _normalize_info_obj_to_dict

        result = _normalize_info_obj_to_dict(42)
        assert result == {}


# ===================================================================
# _sync_runner.py coverage
# ===================================================================


class TestSyncRunnerBatchMode:
    """Tests for batch mode in ProbeRunner (use_probe_batch=True)."""

    def test_batch_mode_basic(self, tmp_path):
        """Test batch mode execution path."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runtime.runner import ProbeRunner

        class TestModel(Model):
            def __init__(self):
                super().__init__(name="test-model")

            def generate(self, prompt, **kwargs):
                return f"response to {prompt}"

        class BatchProbe(Probe):
            def __init__(self):
                super().__init__(name="batch-probe")

            def run(self, model, item, **kwargs):
                return model.generate(item)

            def run_batch(self, model, dataset, **kwargs):
                results = []
                for item in dataset:
                    results.append(
                        ProbeResult(
                            input=item,
                            output=f"batch: {item}",
                            status=ResultStatus.SUCCESS,
                            metadata={},
                        )
                    )
                return results

        runner = ProbeRunner(TestModel(), BatchProbe())
        results = runner.run(
            ["q1", "q2", "q3"],
            use_probe_batch=True,
            emit_run_artifacts=False,
        )
        assert len(results) == 3
        assert results[0]["status"] == "success"

    def test_batch_mode_with_artifacts(self, tmp_path):
        """Test batch mode with artifact emission."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runtime.runner import ProbeRunner

        class TestModel(Model):
            def __init__(self):
                super().__init__(name="test-model")

            def generate(self, prompt, **kwargs):
                return "ok"

        class BatchProbe(Probe):
            def __init__(self):
                super().__init__(name="batch-probe")

            def run(self, model, item, **kwargs):
                return "ok"

            def run_batch(self, model, dataset, **kwargs):
                return [
                    ProbeResult(
                        input=item,
                        output="batch ok",
                        status=ResultStatus.SUCCESS,
                        metadata={},
                    )
                    for item in dataset
                ]

        run_dir = tmp_path / "batch_run"
        runner = ProbeRunner(TestModel(), BatchProbe())
        results = runner.run(
            ["q1", "q2"],
            use_probe_batch=True,
            emit_run_artifacts=True,
            run_dir=run_dir,
        )
        assert len(results) == 2
        assert (run_dir / "records.jsonl").exists()
        assert (run_dir / "manifest.json").exists()

    def test_batch_mode_stop_on_error(self, tmp_path):
        """Test batch mode with stop_on_error raising RunnerExecutionError."""
        from insideLLMs.exceptions import RunnerExecutionError
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runtime.runner import ProbeRunner

        class TestModel(Model):
            def __init__(self):
                super().__init__(name="test-model")

            def generate(self, prompt, **kwargs):
                return "ok"

        class FailingBatchProbe(Probe):
            def __init__(self):
                super().__init__(name="failing-batch")

            def run(self, model, item, **kwargs):
                return "ok"

            def run_batch(self, model, dataset, **kwargs):
                results = []
                for i, item in enumerate(dataset):
                    if i == 1:
                        results.append(
                            ProbeResult(
                                input=item,
                                status=ResultStatus.ERROR,
                                error="batch error",
                                metadata={"error_type": "RuntimeError"},
                            )
                        )
                    else:
                        results.append(
                            ProbeResult(
                                input=item,
                                output="ok",
                                status=ResultStatus.SUCCESS,
                                metadata={},
                            )
                        )
                return results

        runner = ProbeRunner(TestModel(), FailingBatchProbe())
        with pytest.raises(RunnerExecutionError):
            runner.run(
                ["a", "b", "c"],
                use_probe_batch=True,
                stop_on_error=True,
                emit_run_artifacts=False,
            )

    def test_batch_mode_with_progress_callback(self, tmp_path):
        """Test batch mode progress callback is invoked."""
        from insideLLMs.models.base import Model
        from insideLLMs.probes.base import Probe
        from insideLLMs.runtime.runner import ProbeRunner

        class TestModel(Model):
            def __init__(self):
                super().__init__(name="test-model")

            def generate(self, prompt, **kwargs):
                return "ok"

        class BatchProbe(Probe):
            def __init__(self):
                super().__init__(name="batch-probe")

            def run(self, model, item, **kwargs):
                return "ok"

            def run_batch(self, model, dataset, progress_callback=None, **kwargs):
                results = []
                for i, item in enumerate(dataset):
                    if progress_callback:
                        progress_callback(i + 1, len(dataset))
                    results.append(
                        ProbeResult(
                            input=item,
                            output="ok",
                            status=ResultStatus.SUCCESS,
                            metadata={},
                        )
                    )
                return results

        progress_calls = []

        def cb(current, total):
            progress_calls.append((current, total))

        runner = ProbeRunner(TestModel(), BatchProbe())
        runner.run(
            ["q1", "q2"],
            use_probe_batch=True,
            progress_callback=cb,
            emit_run_artifacts=False,
        )
        # Final progress call for "complete" status
        assert progress_calls[-1] == (2, 2)


class TestSyncRunnerConfigSnapshotPath:
    """Tests for config_snapshot path in _sync_runner."""

    def test_config_snapshot_written_to_yaml(self, tmp_path):
        """config_snapshot is written as config.resolved.yaml when artifacts emitted."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        run_dir = tmp_path / "snapshot_run"
        snapshot = {"model": {"type": "dummy"}, "probe": {"type": "logic"}}

        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(
            ["test"],
            emit_run_artifacts=True,
            run_dir=run_dir,
            config_snapshot=snapshot,
        )

        config_yaml = run_dir / "config.resolved.yaml"
        assert config_yaml.exists()
        loaded = yaml.safe_load(config_yaml.read_text())
        assert loaded["model"]["type"] == "dummy"

    def test_config_snapshot_generates_deterministic_run_id(self, tmp_path):
        """config_snapshot drives deterministic run_id derivation."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        snapshot = {"model": {"type": "dummy"}, "probe": {"type": "logic"}}

        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(
            ["test"],
            emit_run_artifacts=False,
            config_snapshot=snapshot,
        )
        run_id_1 = runner.last_run_id

        runner2 = ProbeRunner(DummyModel(), LogicProbe())
        runner2.run(
            ["test"],
            emit_run_artifacts=False,
            config_snapshot=snapshot,
        )
        run_id_2 = runner2.last_run_id

        assert run_id_1 == run_id_2
        assert len(run_id_1) > 0


class TestSyncRunnerDeterministicArtifacts:
    """Tests for deterministic_artifacts flag."""

    def test_deterministic_artifacts_from_strict_serialization(self, tmp_path):
        """deterministic_artifacts defaults to strict_serialization value."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        run_dir = tmp_path / "det_run"
        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(
            ["test"],
            emit_run_artifacts=True,
            run_dir=run_dir,
            strict_serialization=True,
            # deterministic_artifacts not specified -> defaults to strict_serialization
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["python_version"] is None
        assert manifest["platform"] is None


class TestSyncRunnerManifestVersionComparison:
    """Tests for schema version comparison in manifest."""

    def test_manifest_run_completed_field_with_v1_0_1(self, tmp_path):
        """manifest includes run_completed for schema >= 1.0.1."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        run_dir = tmp_path / "version_run"
        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(
            ["test"],
            emit_run_artifacts=True,
            run_dir=run_dir,
            schema_version="1.0.1",
        )

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest.get("run_completed") is True


class TestSyncRunnerReturnExperiment:
    """Tests for return_experiment flag."""

    def test_return_experiment_true(self):
        """return_experiment=True returns ExperimentResult."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        runner = ProbeRunner(DummyModel(), LogicProbe())
        result = runner.run(
            ["test"],
            emit_run_artifacts=False,
            return_experiment=True,
        )
        assert isinstance(result, ExperimentResult)

    def test_return_experiment_false(self):
        """return_experiment=False returns list."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        runner = ProbeRunner(DummyModel(), LogicProbe())
        result = runner.run(
            ["test"],
            emit_run_artifacts=False,
            return_experiment=False,
        )
        assert isinstance(result, list)


class TestSyncRunnerResumeWithMoreRecordsThanPrompts:
    """Test resume with too many existing records."""

    def test_resume_too_many_records_raises(self, tmp_path):
        """Raises ValueError if existing records exceed prompt_set length."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        run_dir = tmp_path / "resume_too_many"
        run_dir.mkdir()
        (run_dir / ".insidellms_run").write_text("marker")

        # Write 3 records but we'll only provide 2 prompts
        records_path = run_dir / "records.jsonl"
        for i in range(3):
            record = {
                "input": f"q{i}",
                "custom": {"record_index": i},
                "run_id": "test-run",
                "status": "success",
            }
            with open(records_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        runner = ProbeRunner(DummyModel(), LogicProbe())
        with pytest.raises(ValueError, match="more entries"):
            runner.run(
                ["q0", "q1"],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="test-run",
                resume=True,
            )


class TestSyncRunnerExplicitRunId:
    """Test explicit run_id."""

    def test_explicit_run_id_used(self, tmp_path):
        """Explicit run_id bypasses deterministic derivation."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(
            ["test"],
            emit_run_artifacts=False,
            run_id="my-explicit-run-id",
        )
        assert runner.last_run_id == "my-explicit-run-id"


class TestSyncRunnerValidateOutput:
    """Test validation paths in sync runner."""

    def test_validate_output_in_non_batch_mode(self, tmp_path):
        """validate_output=True invokes validation for each record."""
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        run_dir = tmp_path / "validate_run"
        runner = ProbeRunner(DummyModel(), LogicProbe())
        # Should complete without error
        results = runner.run(
            ["test"],
            emit_run_artifacts=True,
            run_dir=run_dir,
            validate_output=True,
            schema_version="1.0.0",
            validation_mode="lenient",
        )
        assert len(results) == 1


class TestSyncRunnerNoArtifacts:
    """Test that last_run_dir is None when artifacts are not emitted."""

    def test_no_artifacts_run_dir_none(self):
        from insideLLMs.models import DummyModel
        from insideLLMs.probes import LogicProbe
        from insideLLMs.runtime.runner import ProbeRunner

        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(["test"], emit_run_artifacts=False)
        assert runner.last_run_dir is None


# ===================================================================
# results.py coverage
# ===================================================================


class TestSerializeForJsonExtended:
    """Extended tests for _serialize_for_json."""

    def test_serialize_object_with_dict(self):
        """Objects with __dict__ are serialized as dicts."""
        from insideLLMs.results import _serialize_for_json

        class Obj:
            def __init__(self):
                self.x = 1
                self.y = "hello"

        result = _serialize_for_json(Obj())
        assert result == {"x": 1, "y": "hello"}

    def test_serialize_tuple(self):
        """Tuples are serialized as lists."""
        from insideLLMs.results import _serialize_for_json

        result = _serialize_for_json((1, 2, 3))
        assert result == [1, 2, 3]

    def test_serialize_nested_dataclass_with_enum(self):
        """Nested structures with enums and dataclasses."""
        from insideLLMs.results import _serialize_for_json

        result = _serialize_for_json(ResultStatus.SUCCESS)
        assert result == "success"

    def test_serialize_primitive_passthrough(self):
        """Primitives pass through unchanged."""
        from insideLLMs.results import _serialize_for_json

        assert _serialize_for_json(42) == 42
        assert _serialize_for_json("hello") == "hello"
        assert _serialize_for_json(None) is None
        assert _serialize_for_json(True) is True


class TestSaveResultsJsonExtended:
    """Extended tests for save_results_json."""

    def test_save_experiment_result(self, tmp_path):
        """ExperimentResult is serializable."""
        from insideLLMs.results import load_results_json, save_results_json

        exp = _make_experiment()
        path = str(tmp_path / "exp.json")
        save_results_json(exp, path)

        loaded = load_results_json(path)
        assert loaded["experiment_id"] == "exp-001"

    def test_save_parent_not_exists_raises(self, tmp_path):
        """FileNotFoundError if parent dir does not exist."""
        from insideLLMs.results import save_results_json

        with pytest.raises(FileNotFoundError, match="Parent directory"):
            save_results_json([{"a": 1}], str(tmp_path / "nonexistent" / "data.json"))

    def test_save_with_validation_explicit_schema(self, tmp_path):
        """validate_output=True with explicit schema_name."""
        from insideLLMs.results import save_results_json

        results = [
            {"input": "q", "output": "a", "status": "success", "schema_version": "1.0.0"}
        ]
        path = str(tmp_path / "validated.json")
        # Use correct schema name: "ProbeResult" (SchemaRegistry.RUNNER_ITEM)
        save_results_json(
            results,
            path,
            validate_output=True,
            schema_name="ProbeResult",
            schema_version="1.0.0",
            validation_mode="lenient",
        )
        assert Path(path).exists()

    def test_save_with_validation_heuristic(self, tmp_path):
        """validate_output=True without schema_name uses heuristic."""
        from insideLLMs.results import save_results_json

        results = [
            {"input": "q", "output": "a", "status": "success", "schema_version": "1.0.0"}
        ]
        path = str(tmp_path / "heuristic.json")
        save_results_json(
            results,
            path,
            validate_output=True,
            schema_version="1.0.0",
            validation_mode="lenient",
        )
        assert Path(path).exists()

    def test_save_non_list_with_validation(self, tmp_path):
        """validate_output with non-list (e.g., dict) and explicit schema_name."""
        from insideLLMs.results import save_results_json

        data = {"input": "q", "output": "a", "status": "success"}
        path = str(tmp_path / "dict_validated.json")
        # Use correct schema name: "ProbeResult" (SchemaRegistry.RUNNER_ITEM)
        save_results_json(
            data,
            path,
            validate_output=True,
            schema_name="ProbeResult",
            schema_version="1.0.0",
            validation_mode="lenient",
        )
        assert Path(path).exists()


class TestSaveResultsMarkdownExtended:
    """Extended tests for save_results_markdown."""

    def test_save_experiment_result(self, tmp_path):
        """ExperimentResult triggers full markdown report."""
        from insideLLMs.results import save_results_markdown

        exp = _make_experiment()
        path = str(tmp_path / "report.md")
        save_results_markdown(exp, path)

        content = Path(path).read_text()
        assert "# Experiment Report" in content

    def test_save_parent_not_exists_raises(self, tmp_path):
        """FileNotFoundError if parent dir does not exist."""
        from insideLLMs.results import save_results_markdown

        with pytest.raises(FileNotFoundError, match="Parent directory"):
            save_results_markdown(
                [{"input": "q"}], str(tmp_path / "nonexistent" / "r.md")
            )


class TestSaveResultsCsvExtended:
    """Extended tests for save_results_csv."""

    def test_parent_not_exists_raises(self, tmp_path):
        """FileNotFoundError if parent dir does not exist."""
        from insideLLMs.results import save_results_csv

        with pytest.raises(FileNotFoundError, match="Parent directory"):
            save_results_csv(
                [{"a": 1}], str(tmp_path / "nonexistent" / "r.csv")
            )


class TestSaveResultsHtmlExtended:
    """Extended tests for save_results_html."""

    def test_basic_save(self, tmp_path):
        """Basic HTML save."""
        from insideLLMs.results import save_results_html

        exp = _make_experiment()
        path = str(tmp_path / "report.html")
        save_results_html(exp, path)

        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content

    def test_parent_not_exists_raises(self, tmp_path):
        """FileNotFoundError if parent dir does not exist."""
        from insideLLMs.results import save_results_html

        exp = _make_experiment()
        with pytest.raises(FileNotFoundError, match="Parent directory"):
            save_results_html(exp, str(tmp_path / "nonexistent" / "r.html"))


class TestSaveComparisonMarkdownExtended:
    """Extended tests for save_comparison_markdown."""

    def test_basic_save(self, tmp_path):
        """Basic comparison markdown save."""
        from insideLLMs.results import save_comparison_markdown

        comp = BenchmarkComparison(
            name="Test",
            experiments=[_make_experiment()],
        )
        path = str(tmp_path / "comparison.md")
        save_comparison_markdown(comp, path)

        content = Path(path).read_text()
        assert "# Benchmark Comparison" in content

    def test_parent_not_exists_raises(self, tmp_path):
        """FileNotFoundError if parent dir does not exist."""
        from insideLLMs.results import save_comparison_markdown

        comp = BenchmarkComparison(
            name="Test",
            experiments=[_make_experiment()],
        )
        with pytest.raises(FileNotFoundError, match="Parent directory"):
            save_comparison_markdown(comp, str(tmp_path / "nonexistent" / "c.md"))


class TestExperimentToMarkdownExtended:
    """Extended tests for experiment_to_markdown."""

    def test_no_score(self):
        """Experiment without score omits Scores section."""
        from insideLLMs.results import experiment_to_markdown

        exp = _make_experiment(score=None)
        md = experiment_to_markdown(exp)
        assert "## Scores" not in md

    def test_with_duration_seconds(self):
        """Experiment with duration_seconds includes Timing section."""
        from insideLLMs.results import experiment_to_markdown

        # duration_seconds is a computed property from started_at/completed_at
        started = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2024, 1, 1, 0, 0, 10, 500000, tzinfo=timezone.utc)
        exp = _make_experiment(started_at=started, completed_at=completed)
        md = experiment_to_markdown(exp)
        assert "## Timing" in md
        assert "10.50 seconds" in md

    def test_explicit_generated_at(self):
        """Explicit generated_at overrides experiment timestamps."""
        from insideLLMs.results import experiment_to_markdown

        gen_at = datetime(2025, 6, 15, 12, 0, 0)
        exp = _make_experiment()
        md = experiment_to_markdown(exp, generated_at=gen_at)
        assert "2025-06-15T12:00:00" in md

    def test_no_generated_at_no_timestamps(self):
        """No generated_at and no experiment timestamps omits footer."""
        from insideLLMs.results import experiment_to_markdown

        exp = _make_experiment(started_at=None, completed_at=None)
        md = experiment_to_markdown(exp)
        assert "_Generated at" not in md

    def test_generated_at_from_completed_at(self):
        """Uses completed_at when generated_at not provided."""
        from insideLLMs.results import experiment_to_markdown

        completed = datetime(2025, 3, 1, 9, 0, 0)
        exp = _make_experiment(completed_at=completed)
        md = experiment_to_markdown(exp)
        assert "2025-03-01T09:00:00" in md

    def test_generated_at_from_started_at(self):
        """Uses started_at when completed_at is None."""
        from insideLLMs.results import experiment_to_markdown

        started = datetime(2025, 2, 1, 8, 0, 0)
        exp = _make_experiment(started_at=started, completed_at=None)
        md = experiment_to_markdown(exp)
        assert "2025-02-01T08:00:00" in md

    def test_output_none_in_results(self):
        """Result with None output shows empty in table."""
        from insideLLMs.results import experiment_to_markdown

        results = [
            ProbeResult(
                input="test",
                output=None,
                status=ResultStatus.ERROR,
                error="Failed",
            )
        ]
        exp = _make_experiment(results=results)
        md = experiment_to_markdown(exp)
        assert "error" in md


class TestScoreToMarkdownLines:
    """Tests for _score_to_markdown_lines."""

    def test_full_score(self):
        from insideLLMs.results import _score_to_markdown_lines

        score = ProbeScore(
            accuracy=0.95,
            precision=0.92,
            recall=0.89,
            f1_score=0.905,
            mean_latency_ms=150.0,
            error_rate=0.05,
        )
        lines = _score_to_markdown_lines(score)
        text = "\n".join(lines)
        assert "Accuracy" in text
        assert "Precision" in text
        assert "Recall" in text
        assert "F1 Score" in text
        assert "Mean Latency" in text
        assert "Error Rate" in text

    def test_partial_score(self):
        """Score with some None fields."""
        from insideLLMs.results import _score_to_markdown_lines

        score = ProbeScore(accuracy=0.9, error_rate=0.1)
        lines = _score_to_markdown_lines(score)
        text = "\n".join(lines)
        assert "Accuracy" in text
        assert "Precision" not in text
        assert "Recall" not in text

    def test_score_with_custom_metrics(self):
        from insideLLMs.results import _score_to_markdown_lines

        score = ProbeScore(
            accuracy=0.9,
            error_rate=0.1,
            custom_metrics={"perplexity": 12.5, "bleu": 0.82},
        )
        lines = _score_to_markdown_lines(score)
        text = "\n".join(lines)
        assert "Custom Metrics" in text
        assert "perplexity" in text
        assert "bleu" in text


class TestExperimentToHtmlExtended:
    """Extended tests for experiment_to_html."""

    def test_no_score_section(self):
        """HTML without score omits Scores section."""
        from insideLLMs.results import experiment_to_html

        exp = _make_experiment(score=None)
        html = experiment_to_html(exp)
        assert "Accuracy" not in html

    def test_with_all_score_fields(self):
        """HTML with all score fields."""
        from insideLLMs.results import experiment_to_html

        score = ProbeScore(
            accuracy=0.95,
            precision=0.92,
            recall=0.89,
            f1_score=0.905,
            error_rate=0.05,
        )
        exp = _make_experiment(score=score)
        html = experiment_to_html(exp)
        assert "95.0%" in html
        assert "Precision" in html
        assert "Recall" in html
        assert "F1 Score" in html

    def test_escape_html_false(self):
        """escape_html=False does not escape."""
        from insideLLMs.results import experiment_to_html

        exp = _make_experiment()
        exp.results[0].input = "<b>bold</b>"
        html = experiment_to_html(exp, escape_html=False)
        assert "<b>bold</b>" in html

    def test_generated_at_string(self):
        """generated_at as string."""
        from insideLLMs.results import experiment_to_html

        exp = _make_experiment()
        html = experiment_to_html(exp, generated_at="2025-01-01")
        assert "2025-01-01" in html

    def test_generated_at_datetime(self):
        """generated_at as datetime."""
        from insideLLMs.results import experiment_to_html

        exp = _make_experiment()
        html = experiment_to_html(exp, generated_at=datetime(2025, 6, 1, 12, 0, 0))
        assert "2025-06-01T12:00:00" in html

    def test_no_generated_at(self):
        """No generated_at and no timestamps."""
        from insideLLMs.results import experiment_to_html

        exp = _make_experiment(started_at=None, completed_at=None)
        html = experiment_to_html(exp)
        assert "Generated at" not in html

    def test_error_result_in_table(self):
        """Error results get error class."""
        from insideLLMs.results import experiment_to_html

        results = [
            ProbeResult(
                input="test",
                output=None,
                status=ResultStatus.ERROR,
                error="Failed",
            )
        ]
        exp = _make_experiment(results=results)
        html = experiment_to_html(exp)
        assert "class='error'" in html


class TestComparisonToMarkdownExtended:
    """Extended tests for comparison_to_markdown."""

    def test_no_rankings_or_summary(self):
        """Comparison without rankings or summary."""
        from insideLLMs.results import comparison_to_markdown

        comp = BenchmarkComparison(
            name="Simple",
            experiments=[_make_experiment()],
        )
        md = comparison_to_markdown(comp)
        assert "## Rankings" not in md
        assert "## Summary" not in md

    def test_with_no_score(self):
        """Experiment without score shows N/A."""
        from insideLLMs.results import comparison_to_markdown

        comp = BenchmarkComparison(
            name="NoScore",
            experiments=[_make_experiment(score=None)],
        )
        md = comparison_to_markdown(comp)
        assert "N/A" in md

    def test_with_score_but_no_latency(self):
        """Score without latency shows N/A for latency."""
        from insideLLMs.results import comparison_to_markdown

        score = ProbeScore(accuracy=0.9, error_rate=0.1)
        comp = BenchmarkComparison(
            name="NoLatency",
            experiments=[_make_experiment(score=score)],
        )
        md = comparison_to_markdown(comp)
        assert "90.0%" in md


class TestStatisticalReportExtended:
    """Extended tests for generate_statistical_report."""

    def _create_experiments(self, n=3):
        experiments = []
        for i in range(n):
            experiments.append(
                _make_experiment(
                    score=ProbeScore(accuracy=0.8 + i * 0.05, error_rate=0.2 - i * 0.05),
                    started_at=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
                    completed_at=datetime(2024, 1, 1 + i, 0, 30, tzinfo=timezone.utc),
                )
            )
        return experiments

    def test_with_custom_confidence_level(self):
        from insideLLMs.results import generate_statistical_report

        experiments = self._create_experiments()
        report = generate_statistical_report(
            experiments, confidence_level=0.99, format="markdown"
        )
        assert "99%" in report

    def test_save_to_file(self, tmp_path):
        from insideLLMs.results import generate_statistical_report

        experiments = self._create_experiments()
        path = str(tmp_path / "report.md")
        generate_statistical_report(experiments, output_path=path)
        assert Path(path).exists()

    def test_save_to_nonexistent_parent_raises(self, tmp_path):
        from insideLLMs.results import generate_statistical_report

        experiments = self._create_experiments()
        with pytest.raises(FileNotFoundError, match="Parent directory"):
            generate_statistical_report(
                experiments,
                output_path=str(tmp_path / "nonexistent" / "report.md"),
            )

    def test_generated_at_from_experiments(self):
        """generated_at derived from latest experiment completion."""
        from insideLLMs.results import generate_statistical_report

        experiments = self._create_experiments()
        report = generate_statistical_report(experiments, format="markdown")
        assert "Report generated at" in report

    def test_html_format(self):
        from insideLLMs.results import generate_statistical_report

        experiments = self._create_experiments()
        report = generate_statistical_report(experiments, format="html")
        assert "<!DOCTYPE html>" in report
        assert "Statistical Analysis Report" in report

    def test_json_format(self):
        from insideLLMs.results import generate_statistical_report

        experiments = self._create_experiments()
        report = generate_statistical_report(experiments, format="json")
        data = json.loads(report)
        assert "total_experiments" in data


class TestStatisticalReportToMarkdown:
    """Direct tests for _statistical_report_to_markdown."""

    def test_basic(self):
        from insideLLMs.results import _statistical_report_to_markdown

        summary = {
            "total_experiments": 10,
            "unique_models": ["GPT-4", "Claude"],
            "unique_probes": ["factual"],
            "overall": {
                "success_rate": {"mean": 0.95, "std": 0.02},
                "latency_ms": {"mean": 150.0, "std": 25.0},
            },
            "by_model": {
                "GPT-4": {
                    "n_experiments": 5,
                    "success_rate": {"mean": 0.96},
                    "success_rate_ci": {"lower": 0.92, "upper": 0.99},
                    "latency_ms": {"mean": 140.0},
                },
            },
            "by_probe": {
                "factual": {
                    "n_experiments": 10,
                    "success_rate": {"mean": 0.95},
                    "accuracy": {"mean": 0.90},
                    "accuracy_ci": {"lower": 0.85, "upper": 0.95},
                },
            },
        }
        md = _statistical_report_to_markdown(summary, 0.95, None)
        assert "# Statistical Analysis Report" in md
        assert "GPT-4" in md
        assert "factual" in md
        assert "Overall Performance" in md
        assert "Performance by Model" in md
        assert "Performance by Probe" in md

    def test_with_generated_at(self):
        from insideLLMs.results import _statistical_report_to_markdown

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
        }
        gen_at = datetime(2025, 1, 1, 12, 0, 0)
        md = _statistical_report_to_markdown(summary, 0.95, gen_at)
        assert "2025-01-01T12:00:00" in md

    def test_no_overall(self):
        """Summary without overall section."""
        from insideLLMs.results import _statistical_report_to_markdown

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
        }
        md = _statistical_report_to_markdown(summary, 0.95, None)
        assert "Overall Performance" not in md

    def test_model_without_latency(self):
        """Model data without latency_ms."""
        from insideLLMs.results import _statistical_report_to_markdown

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
            "by_model": {
                "M": {
                    "n_experiments": 1,
                    "success_rate": {"mean": 0.9},
                },
            },
        }
        md = _statistical_report_to_markdown(summary, 0.95, None)
        assert "N/A" in md

    def test_probe_without_accuracy(self):
        """Probe data without accuracy."""
        from insideLLMs.results import _statistical_report_to_markdown

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
            "by_probe": {
                "P": {
                    "n_experiments": 1,
                    "success_rate": {"mean": 0.9},
                },
            },
        }
        md = _statistical_report_to_markdown(summary, 0.95, None)
        assert "N/A" in md


class TestStatisticalReportToHtml:
    """Direct tests for _statistical_report_to_html."""

    def test_basic(self):
        from insideLLMs.results import _statistical_report_to_html

        summary = {
            "total_experiments": 5,
            "unique_models": ["GPT-4"],
            "unique_probes": ["logic"],
            "by_model": {
                "GPT-4": {
                    "n_experiments": 5,
                    "success_rate": {"mean": 0.9},
                    "success_rate_ci": {"lower": 0.85, "upper": 0.95},
                },
            },
        }
        html = _statistical_report_to_html(summary, 0.95, None)
        assert "<!DOCTYPE html>" in html
        assert "GPT-4" in html
        assert "summary-stat" in html

    def test_with_generated_at(self):
        from insideLLMs.results import _statistical_report_to_html

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
        }
        gen_at = datetime(2025, 6, 1, 10, 0, 0)
        html = _statistical_report_to_html(summary, 0.95, gen_at)
        assert "2025-06-01T10:00:00" in html

    def test_without_by_model(self):
        """Summary without by_model."""
        from insideLLMs.results import _statistical_report_to_html

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
        }
        html = _statistical_report_to_html(summary, 0.95, None)
        assert "<table>" not in html

    def test_model_without_ci(self):
        """Model without confidence interval."""
        from insideLLMs.results import _statistical_report_to_html

        summary = {
            "total_experiments": 1,
            "unique_models": ["M"],
            "unique_probes": ["P"],
            "by_model": {
                "M": {
                    "n_experiments": 1,
                    "success_rate": {"mean": 0.9},
                },
            },
        }
        html = _statistical_report_to_html(summary, 0.95, None)
        assert "N/A" in html


class TestFormatNumberExtended:
    """Extended tests for _format_number."""

    def test_zero(self):
        from insideLLMs.results import _format_number

        assert _format_number(0.0) == "0.0000"

    def test_large_number(self):
        from insideLLMs.results import _format_number

        assert _format_number(12345.6789, 2) == "12345.68"

    def test_default_precision(self):
        from insideLLMs.results import _format_number

        assert _format_number(0.123456789) == "0.1235"


class TestEscapeMarkdownCellExtended:
    """Extended tests for _escape_markdown_cell."""

    def test_integer_input(self):
        from insideLLMs.results import _escape_markdown_cell

        assert _escape_markdown_cell(42) == "42"

    def test_none_input(self):
        from insideLLMs.results import _escape_markdown_cell

        assert _escape_markdown_cell(None) == "None"

    def test_no_special_chars(self):
        from insideLLMs.results import _escape_markdown_cell

        assert _escape_markdown_cell("hello world") == "hello world"
