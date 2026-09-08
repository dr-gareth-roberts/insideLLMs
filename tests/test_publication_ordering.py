"""Publication must commit the finalized execution bytes before the remote write."""

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from insideLLMs.attestations import parse_dsse_envelope
from insideLLMs.config_types import RunConfig
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.publish.oras import PushResult
from insideLLMs.runtime._ultimate import run_ultimate_post_artifact
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner


def execution_directory(path: Path) -> Path:
    path.mkdir()
    (path / "manifest.json").write_text('{"run_id":"fixture","run_completed":true}\n')
    (path / "records.jsonl").write_text('{"input":"q","output":"a"}\n')
    (path / "summary.json").write_text('{"summary":{"accuracy":0.5}}\n')
    return path


def tree_bytes(path: Path) -> dict[str, bytes]:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def receipt(path: Path) -> dict:
    envelope = json.loads((path / "attestations/09.publish.dsse.json").read_bytes())
    return parse_dsse_envelope(envelope)[0]["predicate"]


def assert_identity(payload: dict[str, bytes]) -> str:
    raw = payload["integrity/bundle_identity.json"]
    descriptor = json.loads(raw)
    assert descriptor["version"] == 2
    entries = descriptor["files"]
    excluded = {"integrity/bundle_identity.json", "integrity/bundle_id.txt"}
    assert [entry["path"] for entry in entries] == sorted(set(payload) - excluded)
    for entry in entries:
        assert entry["sha256"] == hashlib.sha256(payload[entry["path"]]).hexdigest()
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    bundle_id = hashlib.sha256(canonical.encode()).hexdigest()
    assert payload["integrity/bundle_id.txt"] == (bundle_id + "\n").encode()
    return bundle_id


def test_push_observes_final_policy_identity_and_exact_local_bytes(tmp_path, monkeypatch):
    run_dir = execution_directory(tmp_path / "run")
    captured = []

    def push(payload_dir, ref):
        payload = tree_bytes(payload_dir)
        assert "attestations/08.policy.dsse.json" in payload
        assert "integrity/bundle_identity.json" in payload
        assert "integrity/bundle_id.txt" in payload
        assert "attestations/09.publish.dsse.json" not in payload
        assert payload_dir != run_dir
        assert all((run_dir / name).read_bytes() == data for name, data in payload.items())
        captured.append(payload)
        return PushResult(ref=ref, digest="sha256:confirmed")

    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", push)
    run_ultimate_post_artifact(run_dir, publish_oci_ref="registry.example/run:v2")
    assert len(captured) == 1
    assert set(captured[0]) == {
        "manifest.json",
        "records.jsonl",
        "summary.json",
        "policy/verdict.json",
        "integrity/records.merkle.json",
        "integrity/bundle_identity.json",
        "integrity/bundle_id.txt",
        "attestations/00.source.dsse.json",
        "attestations/01.env.dsse.json",
        "attestations/02.dataset.dsse.json",
        "attestations/03.promptset.dsse.json",
        "attestations/04.execution.dsse.json",
        "attestations/05.scoring.dsse.json",
        "attestations/06.report.dsse.json",
        "attestations/07.claims.dsse.json",
        "attestations/08.policy.dsse.json",
    }
    bundle_id = assert_identity(captured[0])
    assert receipt(run_dir)["outcome"] == "published"
    assert receipt(run_dir)["payload_id"] == bundle_id
    assert all((run_dir / name).read_bytes() == data for name, data in captured[0].items())


@pytest.mark.parametrize(
    ("failure", "outcome"),
    [
        (RuntimeError("registry rejected"), "failed"),
        (TimeoutError("timed out"), "unknown"),
        (subprocess.TimeoutExpired("oras", 1), "unknown"),
        (None, "unknown"),
    ],
)
def test_failed_or_uncertain_push_preserves_payload_and_reports_outcome(
    tmp_path, monkeypatch, failure, outcome
):
    run_dir = execution_directory(tmp_path / "run")
    before = tree_bytes(run_dir)
    captured = []

    def push(path, ref):
        captured.append(tree_bytes(path))
        if failure is not None:
            raise failure
        return PushResult(ref=ref, digest=None)

    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", push)
    with pytest.raises(RuntimeError) as error:
        run_ultimate_post_artifact(run_dir, publish_oci_ref="registry.example/run:v2")
    assert error.value.outcome == outcome
    assert len(captured) == 1
    assert receipt(run_dir)["outcome"] == outcome
    assert receipt(run_dir)["oci_digest"] is None
    assert receipt(run_dir)["oci_ref"] == "registry.example/run:v2"
    assert receipt(run_dir)["payload_id"] == assert_identity(captured[0])
    assert all((run_dir / name).read_bytes() == data for name, data in captured[0].items())
    assert all((run_dir / name).read_bytes() == data for name, data in before.items())


def test_no_publish_still_finalizes_deterministic_v2_identity(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("no publication was requested")

    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", forbidden)
    ids = []
    for name in ("one", "two"):
        run_dir = execution_directory(tmp_path / name)
        run_ultimate_post_artifact(run_dir)
        payload = tree_bytes(run_dir)
        del payload["attestations/09.publish.dsse.json"]
        ids.append(assert_identity(payload))
        assert receipt(run_dir)["outcome"] == "not_requested"
    assert ids[0] == ids[1]


def test_failed_policy_never_pushes_but_preserves_verdict(tmp_path, monkeypatch):
    run_dir = execution_directory(tmp_path / "run")
    (run_dir / "records.jsonl").unlink()
    calls = []
    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", lambda *a: calls.append(a))
    with pytest.raises(RuntimeError, match="policy"):
        run_ultimate_post_artifact(run_dir, publish_oci_ref="registry.example/run:v2")
    assert calls == []
    assert json.loads((run_dir / "policy/verdict.json").read_bytes())["passed"] is False
    assert (run_dir / "attestations/08.policy.dsse.json").is_file()


def test_old_receipt_and_unselected_leftovers_are_preserved_but_never_pushed(tmp_path, monkeypatch):
    run_dir = execution_directory(tmp_path / "run")
    leftovers = {
        "attestations/09.publish.dsse.json": b"old publication receipt",
        "attestations/99.leftover.dsse.json": b"old generated attestation",
        "receipts/scitt/04.execution.receipt.json": b"old transparency receipt",
        "integrity/unused.merkle.json": b"old unrelated root",
        "signing/04.execution.sigstore.bundle.json": b"old detached signature",
        "report.html": b"old report",
        "temp.txt": b"temporary output",
    }
    for name, data in leftovers.items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_bytes(data)
    captured = []

    def push(path, ref):
        captured.append(tree_bytes(path))
        return PushResult(ref=ref, digest="sha256:new")

    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", push)
    run_ultimate_post_artifact(run_dir, publish_oci_ref="registry.example/run:v2")
    assert len(captured) == 1
    assert not (set(leftovers) & set(captured[0]))
    assert all((run_dir / name).read_bytes() == data for name, data in leftovers.items())
    attempts = list((run_dir / "publication/attempts").glob("*.09.publish.dsse.json"))
    assert len(attempts) == 1
    new_receipt = parse_dsse_envelope(json.loads(attempts[0].read_bytes()))[0]["predicate"]
    assert new_receipt["outcome"] == "published"
    assert new_receipt["payload_id"] == assert_identity(captured[0])


@pytest.mark.parametrize(
    "conflict",
    [
        "integrity/bundle_id.txt",
        "integrity/bundle_identity.json",
        "integrity/records.merkle.json",
        "attestations/04.execution.dsse.json",
        "attestations/08.policy.dsse.json",
        "policy/verdict.json",
    ],
)
def test_historical_conflicts_reject_before_any_source_write_or_push(
    tmp_path, monkeypatch, conflict
):
    run_dir = execution_directory(tmp_path / "run")
    path = run_dir / conflict
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"historical evidence")
    before = tree_bytes(run_dir)
    calls = []
    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="fresh export"):
        run_ultimate_post_artifact(run_dir, publish_oci_ref="registry.example/run:v2")
    assert calls == []
    assert tree_bytes(run_dir) == before


def test_matching_v2_reuse_never_rewrites_existing_evidence(tmp_path):
    run_dir = execution_directory(tmp_path / "run")
    run_ultimate_post_artifact(run_dir)
    before = tree_bytes(run_dir)
    metadata = {name: (run_dir / name).stat().st_mtime_ns for name in before}
    run_ultimate_post_artifact(run_dir)
    assert tree_bytes(run_dir) == before
    assert {name: (run_dir / name).stat().st_mtime_ns for name in before} == metadata


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_nonregular_input_fails_before_mutation_or_push(tmp_path, monkeypatch, kind):
    import os

    run_dir = execution_directory(tmp_path / "run")
    path = run_dir / "summary.json"
    path.unlink()
    if kind == "symlink":
        path.symlink_to(run_dir / "manifest.json")
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    calls = []
    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", lambda *a: calls.append(a))
    with pytest.raises((ValueError, OSError)):
        run_ultimate_post_artifact(run_dir, publish_oci_ref="registry.example/run:v2")
    assert calls == []
    assert not (run_dir / "attestations").exists()


def test_legacy_bundle_id_retains_original_concatenation_semantics():
    from insideLLMs.crypto import run_bundle_id

    assert run_bundle_id("manifest", {"z": "last", "a": "first"}, ["two", "one"]) == (
        hashlib.sha256(b"manifestafirstzlastonetwo").hexdigest()
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_runner_publishes_summary_after_scoring_once(tmp_path, monkeypatch, asynchronous):
    class CountingProbe(LogicProbe):
        calls = 0

        def score(self, results):
            self.calls += 1
            return super().score(results)

    probe = CountingProbe()
    runner_type = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_type(DummyModel(canned_response="4"), probe)
    captured = []

    def push(path, ref):
        payload = tree_bytes(path)
        assert "summary.json" in payload
        assert probe.calls == 1
        captured.append(payload)
        return PushResult(ref=ref, digest="sha256:confirmed")

    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", push)
    config = RunConfig(
        run_mode="ultimate",
        run_dir=str(tmp_path / "run"),
        publish_oci_ref="registry.example/run:v2",
        validate_output=True,
    )
    result = runner.run([{"question": "2+2?", "reference_answer": "4"}], config=config)
    if asynchronous:
        asyncio.run(result)
    assert probe.calls == 1
    assert len(captured) == 1
    assert_identity(captured[0])
    summary = json.loads(captured[0]["summary.json"])
    assert summary["summary"]["total_experiments"] == 1
    assert summary["summary"]["by_probe"][probe.name]["accuracy"]["mean"] == 1.0


@pytest.mark.parametrize("asynchronous", [False, True])
def test_aggregate_rejection_never_publishes(tmp_path, monkeypatch, asynchronous):
    class RejectingProbe(LogicProbe):
        def score(self, results):
            raise ValueError("score rejected")

    runner_type = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_type(DummyModel(canned_response="4"), RejectingProbe())
    calls = []
    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", lambda *a: calls.append(a))
    config = RunConfig(
        run_mode="ultimate",
        run_dir=str(tmp_path / "run"),
        publish_oci_ref="registry.example/run:v2",
        validate_output=True,
    )
    with pytest.raises(Exception, match="Aggregate scoring failed"):
        result = runner.run([{"question": "2+2?", "reference_answer": "4"}], config=config)
        if asynchronous:
            asyncio.run(result)
    assert calls == []
    assert not (tmp_path / "run/integrity/bundle_identity.json").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_runner_publication_failure_keeps_execution_health_and_scored_result(
    tmp_path, monkeypatch, asynchronous
):
    runner_type = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_type(DummyModel(canned_response="4"), LogicProbe())
    captured = []

    def push(path, ref):
        captured.append(tree_bytes(path))
        raise TimeoutError("remote acknowledgement lost")

    monkeypatch.setattr("insideLLMs.runtime._ultimate.push_run_oci", push)
    run_dir = tmp_path / "run"
    config = RunConfig(
        run_mode="ultimate",
        run_dir=str(run_dir),
        publish_oci_ref="registry.example/run:v2",
        validate_output=True,
    )
    with pytest.raises(RuntimeError, match="Publication unknown"):
        result = runner.run([{"question": "2+2?", "reference_answer": "4"}], config=config)
        if asynchronous:
            asyncio.run(result)
    assert runner.last_experiment.score.accuracy == 1.0
    assert len(captured) == 1
    assert all((run_dir / name).read_bytes() == data for name, data in captured[0].items())
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    assert manifest["custom"]["health"]["healthy"] is True
    assert "abort" not in manifest["custom"]


def test_cli_attest_unattested_execution_without_summary_remains_supported(tmp_path):
    from insideLLMs.cli import main

    run_dir = execution_directory(tmp_path / "run")
    (run_dir / "summary.json").unlink()
    assert main(["attest", str(run_dir)]) == 0
    assert (run_dir / "integrity/bundle_identity.json").is_file()
    assert not (run_dir / "summary.json").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["resume", "overwrite"])
@pytest.mark.parametrize(
    "marker",
    [
        "integrity/bundle_id.txt",
        "integrity/bundle_identity.json",
        "attestations",
        "signing",
    ],
)
def test_runner_refuses_sealed_directory_before_dispatch_or_writes(
    tmp_path, asynchronous, mode, marker
):
    class CountingProbe(LogicProbe):
        calls = 0

        def run(self, model, data, **kwargs):
            self.calls += 1
            return super().run(model, data, **kwargs)

    probe = CountingProbe()
    runner_type = AsyncProbeRunner if asynchronous else ProbeRunner
    runner = runner_type(DummyModel(canned_response="4"), probe)
    run_dir = tmp_path / "run"
    dataset = [{"question": "2+2?", "reference_answer": "4"}]
    initial = runner.run(dataset, run_dir=run_dir)
    if asynchronous:
        asyncio.run(initial)
    path = run_dir / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    if marker in {"attestations", "signing"}:
        path.mkdir()
        (path / "historical.json").write_bytes(b"immutable signed input")
    else:
        path.write_bytes(b"historical bundle ID")
    before = tree_bytes(run_dir)
    calls_before = probe.calls
    with pytest.raises(ValueError, match="fresh run directory"):
        result = runner.run(dataset, run_dir=run_dir, **{mode: True})
        if asynchronous:
            asyncio.run(result)
    assert probe.calls == calls_before
    assert tree_bytes(run_dir) == before


@pytest.mark.parametrize("marker", ["integrity/bundle_id.txt", "attestations", "signing"])
@pytest.mark.parametrize("kind", ["dangling_link", "fifo"])
def test_nonregular_seal_marker_cannot_evade_runner_guard(tmp_path, marker, kind):
    import os

    run_dir = execution_directory(tmp_path / "run")
    path = run_dir / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "dangling_link":
        path.symlink_to(tmp_path / "absent")
    else:
        os.mkfifo(path)
    runner = ProbeRunner(DummyModel(), LogicProbe())
    before = (run_dir / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="fresh run directory"):
        runner.run(["q"], run_dir=run_dir, overwrite=True)
    assert (run_dir / "manifest.json").read_bytes() == before
    assert path.is_symlink() if kind == "dangling_link" else path.stat().st_mode & 0o10000
