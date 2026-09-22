"""R1 — aborted-run manifests must match persisted records.jsonl."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner


class _EchoProbe:
    name = "echo-probe"

    def run(self, _model: Any, item: Any, **_kwargs: Any) -> Any:
        if isinstance(item, dict) and "messages" in item:
            return item["messages"][-1]["content"]
        return str(item)

    def run_batch(
        self,
        model: Any,
        items: list[Any],
        max_workers: int = 1,
        progress_callback: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        results = []
        for i, item in enumerate(items):
            if progress_callback:
                progress_callback(i + 1, len(items))
            results.append(self.run(model, item, **kwargs))
        return results


def _prompt(text: str = "hi") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def _fail_result_record_on_attempt(fail_on: int):
    """Patch OutputValidator.validate to fail the Nth ResultRecord check."""
    from insideLLMs.schemas import OutputValidationError, OutputValidator, SchemaRegistry

    real_validate = OutputValidator.validate
    attempts = {"n": 0}

    def flaky_validate(self, schema_name, data, *args, **kwargs):  # type: ignore[no-untyped-def]
        if schema_name == SchemaRegistry.RESULT_RECORD:
            attempts["n"] += 1
            if attempts["n"] == fail_on:
                raise OutputValidationError(
                    schema_name="ResultRecord",
                    schema_version="1.0.2",
                    errors=["forced record validation failure"],
                )
        return real_validate(self, schema_name, data, *args, **kwargs)

    return OutputValidator, flaky_validate


class _FaultyRecordsFile:
    """Wrap a real file object to fault write or close after bytes land on disk."""

    def __init__(
        self,
        real_fp: Any,
        *,
        mode: Literal["complete", "partial", "close"],
    ) -> None:
        self._fp = real_fp
        self._mode = mode
        self._write_faulted = False
        self._close_faulted = False

    def write(self, data: str) -> int:
        if self._mode == "close" or self._write_faulted:
            return self._fp.write(data)
        self._write_faulted = True
        if self._mode == "complete":
            self._fp.write(data)
            self._fp.flush()
            raise OSError("simulated write failure after complete line")
        # Persist a non-JSON incomplete tail so truncate must drop it.
        if not data:
            raise OSError("simulated empty partial write")
        half = max(1, len(data) // 2)
        # Ensure the partial is not valid JSON even if the full line was.
        partial = data[:half].rstrip("\n")
        if partial.endswith("}"):
            partial = partial[:-1]
        self._fp.write(partial)
        self._fp.flush()
        raise OSError("simulated write failure after partial line")

    def flush(self) -> None:
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            if self._mode == "close" and not self._close_faulted:
                self._close_faulted = True
                raise OSError("simulated close failure after records written")

    def __enter__(self) -> _FaultyRecordsFile:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fp, name)


def _patch_records_open(mode: Literal["complete", "partial", "close"]):
    """Return a patcher that faults write or close on records.jsonl."""
    real_open = builtins.open

    def open_wrapper(path: Any, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(path, *args, **kwargs)
        path_str = str(path)
        if path_str.endswith("records.jsonl"):
            # Only wrap writable handles used for run emission.
            open_mode = args[0] if args else kwargs.get("mode", "r")
            if any(flag in str(open_mode) for flag in ("w", "x", "a")):
                return _FaultyRecordsFile(handle, mode=mode)
        return handle

    return patch("builtins.open", open_wrapper)


def _assert_manifest_matches_records(run_dir: Path) -> dict[str, Any]:
    records_path = run_dir / "records.jsonl"
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()
    lines = [line for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    success = sum(1 for r in records if r.get("status") == "success")
    error = sum(1 for r in records if r.get("status") == "error")
    timeout = sum(1 for r in records if r.get("status") == "timeout")

    assert manifest["record_count"] == len(records)
    assert manifest["success_count"] == success
    assert manifest["error_count"] == error
    status_counts = (manifest.get("custom") or {}).get("status_counts") or {}
    assert status_counts.get("success", 0) == success
    assert status_counts.get("error", 0) == error
    assert status_counts.get("timeout", 0) == timeout
    health = (manifest.get("custom") or {}).get("health") or {}
    assert health.get("record_count") == len(records)
    assert manifest.get("run_completed") is False
    return manifest


def test_sync_manifest_counts_match_empty_records_on_first_validation_fail(
    tmp_path: Path,
) -> None:
    """First-record validation abort → empty jsonl and zero manifest counts."""
    OutputValidator, flaky_validate = _fail_result_record_on_attempt(1)
    runner = ProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-sync-empty"

    with patch.object(OutputValidator, "validate", flaky_validate):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-sync-empty",
                overwrite=True,
                validate_output=True,
                return_experiment=False,
            )

    _assert_manifest_matches_records(run_dir)
    lines = (run_dir / "records.jsonl").read_text(encoding="utf-8").strip()
    assert lines == ""


def test_sync_manifest_counts_match_persisted_prefix_on_mid_run_validation_fail(
    tmp_path: Path,
) -> None:
    """Second-record validation abort → one persisted line; manifest matches."""
    OutputValidator, flaky_validate = _fail_result_record_on_attempt(2)
    runner = ProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-sync-prefix"

    with patch.object(OutputValidator, "validate", flaky_validate):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            runner.run(
                [_prompt("one"), _prompt("two"), _prompt("three")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-sync-prefix",
                overwrite=True,
                validate_output=True,
                return_experiment=False,
            )

    _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1


@pytest.mark.asyncio
async def test_async_manifest_counts_match_empty_records_on_first_validation_fail(
    tmp_path: Path,
) -> None:
    OutputValidator, flaky_validate = _fail_result_record_on_attempt(1)
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-async-empty"

    with patch.object(OutputValidator, "validate", flaky_validate):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            await runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-async-empty",
                overwrite=True,
                validate_output=True,
                return_experiment=False,
                concurrency=1,
            )

    _assert_manifest_matches_records(run_dir)
    lines = (run_dir / "records.jsonl").read_text(encoding="utf-8").strip()
    assert lines == ""


@pytest.mark.asyncio
async def test_async_manifest_counts_match_persisted_prefix_on_mid_run_validation_fail(
    tmp_path: Path,
) -> None:
    OutputValidator, flaky_validate = _fail_result_record_on_attempt(2)
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-async-prefix"

    with patch.object(OutputValidator, "validate", flaky_validate):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            await runner.run(
                [_prompt("one"), _prompt("two"), _prompt("three")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-async-prefix",
                overwrite=True,
                validate_output=True,
                return_experiment=False,
                concurrency=1,
            )

    _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# Write/flush is not transactional — reconcile actual records.jsonl
# ---------------------------------------------------------------------------


def test_sync_manifest_reconciles_complete_line_after_write_raises(tmp_path: Path) -> None:
    """write() persists a full JSONL line then raises → manifest count is 1."""
    runner = ProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-sync-write-complete"

    with _patch_records_open("complete"):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-sync-write-complete",
                overwrite=True,
                validate_output=False,
                return_experiment=False,
            )

    _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    assert json.loads(lines[0]).get("status") == "success"


def test_sync_manifest_reconciles_partial_line_after_write_raises(tmp_path: Path) -> None:
    """write() persists an incomplete tail then raises → truncate → count 0."""
    runner = ProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-sync-write-partial"

    with _patch_records_open("partial"):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-sync-write-partial",
                overwrite=True,
                validate_output=False,
                return_experiment=False,
            )

    _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 0


@pytest.mark.asyncio
async def test_async_manifest_reconciles_complete_line_after_write_raises(
    tmp_path: Path,
) -> None:
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-async-write-complete"

    with _patch_records_open("complete"):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            await runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-async-write-complete",
                overwrite=True,
                validate_output=False,
                return_experiment=False,
                concurrency=1,
            )

    _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    assert json.loads(lines[0]).get("status") == "success"


@pytest.mark.asyncio
async def test_async_manifest_reconciles_partial_line_after_write_raises(
    tmp_path: Path,
) -> None:
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-async-write-partial"

    with _patch_records_open("partial"):
        with pytest.raises(RunnerExecutionError, match="validation or serialization"):
            await runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-async-write-partial",
                overwrite=True,
                validate_output=False,
                return_experiment=False,
                concurrency=1,
            )

    _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 0


# ---------------------------------------------------------------------------
# close() failure must not skip reconcile + manifest finalization
# ---------------------------------------------------------------------------


def test_sync_manifest_finalizes_after_records_close_raises(tmp_path: Path) -> None:
    """close() raises after successful writes → still emit matching manifest."""
    runner = ProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-sync-close-fail"

    with _patch_records_open("close"):
        with pytest.raises(RunnerExecutionError) as exc_info:
            runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-sync-close-fail",
                overwrite=True,
                validate_output=False,
                return_experiment=False,
            )

    err = exc_info.value
    assert isinstance(err.original_error, OSError)
    assert "close failure" in str(err.original_error)

    manifest = _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    abort = (manifest.get("custom") or {}).get("abort") or {}
    assert abort.get("error_type") == "OSError" or "close" in str(abort.get("message", "")).lower()


@pytest.mark.asyncio
async def test_async_manifest_finalizes_after_records_close_raises(tmp_path: Path) -> None:
    runner = AsyncProbeRunner(DummyModel(), _EchoProbe())
    run_dir = tmp_path / "r1-async-close-fail"

    with _patch_records_open("close"):
        with pytest.raises(RunnerExecutionError) as exc_info:
            await runner.run(
                [_prompt("one"), _prompt("two")],
                emit_run_artifacts=True,
                run_dir=run_dir,
                run_id="r1-async-close-fail",
                overwrite=True,
                validate_output=False,
                return_experiment=False,
                concurrency=1,
            )

    err = exc_info.value
    assert isinstance(err.original_error, OSError)
    assert "close failure" in str(err.original_error)

    manifest = _assert_manifest_matches_records(run_dir)
    lines = [
        line
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    abort = (manifest.get("custom") or {}).get("abort") or {}
    assert abort.get("error_type") == "OSError" or "close" in str(abort.get("message", "")).lower()
