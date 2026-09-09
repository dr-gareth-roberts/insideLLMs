"""Resume must never repeat attempted work or trust ambiguous skipped records."""

import asyncio
import builtins
import json
import os

import pytest

from insideLLMs.exceptions import RunnerExecutionError
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime._async_resume import SCHEDULER_UNATTEMPTED_EXECUTION
from insideLLMs.runtime.runner import AsyncProbeRunner


class EchoProbe(LogicProbe):
    def run(self, model, data, **kwargs):
        return model.generate(data)


async def failed_run(tmp_path, monkeypatch, *, batch=False, schema_version="1.0.2"):
    calls = []

    def generate(self, prompt, **kwargs):
        calls.append(prompt)
        if prompt == "b":
            raise RuntimeError("attempted failure")
        return prompt

    monkeypatch.setattr(DummyModel, "generate", generate)
    runner = AsyncProbeRunner(DummyModel(), EchoProbe())
    directory = tmp_path / "run"
    with pytest.raises(RunnerExecutionError):
        await runner.run(
            ["a", "b", "c"],
            run_dir=directory,
            stop_on_error=True,
            concurrency=1,
            use_probe_batch=batch,
            schema_version=schema_version,
            validate_output=True,
        )
    return runner, directory, calls


def append_scheduler_attested_tail(directory, *, item, index):
    """Append the placeholder older async runners persisted for an undispatched item.

    The runner now writes nothing past a fail-fast failure (W7-0007), but run
    directories written before that change can still end in scheduler-attested
    ``skipped`` records. Resume must keep handling them fail-closed, so the tests
    that exercise that path synthesize one from the last attempted record.
    """
    path = directory / "records.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    tail = dict(json.loads(lines[-1]))
    tail["input"] = item
    tail["status"] = "skipped"
    for field in ("output", "output_text", "error", "error_type", "latency_ms", "primary_metric"):
        if field in tail:
            tail[field] = None
    for field in ("usage", "scores", "metadata"):
        if field in tail:
            tail[field] = None
    if "example_id" in tail:
        tail["example_id"] = str(index)
    tail["custom"] = {
        "record_index": index,
        "execution": dict(SCHEDULER_UNATTEMPTED_EXECUTION),
    }
    path.write_bytes(b"".join(lines) + json.dumps(tail).encode("utf-8") + b"\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1.0.0", "1.0.1", "1.0.2"])
async def test_resume_attempts_only_verified_tail_and_preserves_error(
    tmp_path, monkeypatch, schema_version
):
    runner, directory, calls = await failed_run(
        tmp_path, monkeypatch, schema_version=schema_version
    )
    assert calls == ["a", "b"]
    path = directory / "records.jsonl"
    prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:2])
    await runner.run(
        ["a", "b", "c"],
        run_dir=directory,
        resume=True,
        schema_version=schema_version,
        validate_output=True,
    )
    assert calls == ["a", "b", "c"]
    assert path.read_bytes().startswith(prefix)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["status"] for record in records] == ["success", "error", "success"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["custom"]["health"]["healthy"] is False
    await runner.run(["a", "b", "c"], run_dir=directory, resume=True, schema_version=schema_version)
    assert calls == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_scheduler_attested_legacy_tail_is_truncated_and_reattempted(tmp_path, monkeypatch):
    """Positive control for the synthetic tail: undamaged, it must be admitted and retried."""
    runner, directory, calls = await failed_run(tmp_path, monkeypatch)
    append_scheduler_attested_tail(directory, item="c", index=2)
    path = directory / "records.jsonl"
    prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:2])
    assert [json.loads(line)["status"] for line in path.read_text().splitlines()] == [
        "success",
        "error",
        "skipped",
    ]
    await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert calls == ["a", "b", "c"]
    assert path.read_bytes().startswith(prefix)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["status"] for record in records] == ["success", "error", "success"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "generic",
        "output",
        "error",
        "usage",
        "attempt",
        "interleaved",
        "interleaved_attempted",
        "no_error",
        "malformed",
        "numeric_attempted",
        "non_object",
        "unknown_attempt_evidence",
        "output_fingerprint",
    ],
)
async def test_ambiguous_history_rejects_without_artifact_mutation(tmp_path, monkeypatch, damage):
    runner, directory, calls = await failed_run(tmp_path, monkeypatch)
    append_scheduler_attested_tail(directory, item="c", index=2)
    path = directory / "records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    tail = records[2]
    if damage == "generic":
        tail["custom"].pop("execution", None)
    elif damage in {"output", "error", "usage"}:
        tail[damage] = "attempt evidence"
    elif damage == "attempt":
        tail["custom"]["attempts"] = 1
    elif damage == "interleaved":
        records[1]["status"] = "skipped"
    elif damage == "interleaved_attempted":
        records[0]["status"] = "error"
        records[0]["error"] = "earlier attempted error"
        records[1] = {**tail, "input": "b", "custom": {**tail["custom"], "record_index": 1}}
        records[2]["status"] = "success"
    elif damage == "no_error":
        records[1]["status"] = "success"
    elif damage == "numeric_attempted":
        tail["custom"]["execution"]["attempted"] = 0
    elif damage == "unknown_attempt_evidence":
        tail["provider_call_id"] = "attempt-123"
    elif damage == "output_fingerprint":
        tail["custom"]["output_fingerprint"] = "a" * 64
    text = "".join(json.dumps(record) + "\n" for record in records)
    if damage == "malformed":
        text += "not-json\n"
    elif damage == "non_object":
        text += "42\n"
    path.write_text(text)
    # An invalid final fragment must not be truncated before history validation.
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    snapshot = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        await runner.run(
            ["a", "b", "c"],
            run_dir=directory,
            resume=True,
            run_id=runner.last_run_id,
            config_snapshot={"changed": True},
        )
    assert calls == ["a", "b"]
    assert snapshot == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.mark.asyncio
async def test_batch_attempted_outputs_are_never_retried(tmp_path, monkeypatch):
    runner, directory, calls = await failed_run(tmp_path, monkeypatch, batch=True)
    assert sorted(calls) == ["a", "b", "c"]
    original = (directory / "records.jsonl").read_bytes()
    await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert sorted(calls) == ["a", "b", "c"]
    assert (directory / "records.jsonl").read_bytes() == original


@pytest.mark.asyncio
async def test_sealed_tail_rejects_without_mutation(tmp_path, monkeypatch):
    runner, directory, calls = await failed_run(tmp_path, monkeypatch)
    (directory / "integrity").mkdir()
    (directory / "integrity" / "bundle_identity.json").write_text("sealed")
    snapshot = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert calls == ["a", "b"]
    assert snapshot == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.mark.asyncio
@pytest.mark.parametrize("after_replace", [False, True])
async def test_recovery_interruption_preserves_attempts_and_allows_later_resume(
    tmp_path, monkeypatch, after_replace
):
    runner, directory, calls = await failed_run(tmp_path, monkeypatch)
    append_scheduler_attested_tail(directory, item="c", index=2)
    records_path = directory / "records.jsonl"
    original = records_path.read_bytes()
    prefix = b"".join(original.splitlines(keepends=True)[:2])
    replace = os.replace

    def interrupt_replace(source, destination):
        if destination == records_path:
            if after_replace:
                replace(source, destination)
            raise asyncio.CancelledError()
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt_replace)
        with pytest.raises(asyncio.CancelledError):
            await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert calls == ["a", "b"]
    assert records_path.read_bytes() == (prefix if after_replace else original)
    await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert calls == ["a", "b", "c"]
    assert records_path.read_bytes().startswith(prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [b"", b'\n{"partial":', b"\n\n"])
async def test_ordinary_resume_preserves_valid_prefix_and_recovers_only_partial_tail(
    tmp_path, monkeypatch, ending
):
    runner, directory, calls = await failed_run(tmp_path, monkeypatch)
    path = directory / "records.jsonl"
    prefix = b"\r\n".join(path.read_bytes().splitlines()[:2])
    path.write_bytes(prefix + ending)
    await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert calls == ["a", "b", "c"]
    assert path.read_bytes().startswith(prefix + (ending if ending == b"\n\n" else b"\n"))


@pytest.mark.asyncio
async def test_recovery_preserves_bytes_when_text_writes_translate_newlines(tmp_path, monkeypatch):
    runner, directory, calls = await failed_run(tmp_path, monkeypatch)
    path = directory / "records.jsonl"
    original = path.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(original)
    prefix = b"".join(original.splitlines(keepends=True)[:2])
    original_open = builtins.open

    def windows_text_open(file, mode="r", *args, **kwargs):
        if mode == "w":
            kwargs["newline"] = "\r\n"
        return original_open(file, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", windows_text_open)
        await runner.run(["a", "b", "c"], run_dir=directory, resume=True)
    assert calls == ["a", "b", "c"]
    assert path.read_bytes().startswith(prefix)
