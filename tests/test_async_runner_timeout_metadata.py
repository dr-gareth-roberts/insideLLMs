"""Tests for timeout metadata persistence in AsyncProbeRunner artifacts."""

import json
import time
from pathlib import Path

import pytest

from insideLLMs.models import DummyModel
from insideLLMs.runtime.runner import AsyncProbeRunner


class _SlowProbe:
    name = "slow-probe"

    def run(self, _model, _item, **_kwargs):
        time.sleep(0.05)
        return "late"


@pytest.mark.asyncio
async def test_async_runner_persists_timeout_metadata_in_artifacts(tmp_path: Path) -> None:
    runner = AsyncProbeRunner(DummyModel(), _SlowProbe())
    run_dir = tmp_path / "run-timeout"

    results = await runner.run(
        [{"messages": [{"role": "user", "content": "Hello"}]}],
        timeout=0.001,
        emit_run_artifacts=True,
        run_dir=run_dir,
        run_id="timeout-run",
        overwrite=True,
        return_experiment=False,
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["status"] == "timeout"
    assert results[0]["metadata"]["timeout_seconds"] == pytest.approx(0.001)

    records_path = run_dir / "records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "timeout"
    assert record["custom"]["timeout"] is True
    assert record["custom"]["timeout_seconds"] == pytest.approx(0.001)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["custom"]["timeout_count"] == 1
    assert manifest["custom"]["status_counts"]["timeout"] == 1
