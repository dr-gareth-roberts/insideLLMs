"""CLI budget stops retain partial evidence without making paid requests."""

import json
from types import SimpleNamespace

import pytest
import yaml

from insideLLMs.cli import main
from insideLLMs.models import DummyModel
from insideLLMs.probes import LogicProbe
from insideLLMs.runtime.budget import BudgetLedger, BudgetPolicy, BudgetUnsupportedError
from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner


@pytest.fixture
def paid_config(tmp_path, monkeypatch):
    pytest.importorskip("openai")
    import insideLLMs.models.openai as adapter

    requests = []

    class Client:
        def __init__(self, **kwargs):
            self.max_retries = kwargs["max_retries"]
            self.base_url = "https://api.openai.com/v1"
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                model="budget-test",
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    monkeypatch.setattr(adapter, "OpenAI", Client)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    return {
        "model": {"type": "openai", "args": {"model_name": "budget-test"}},
        "probe": {"type": "logic"},
        "dataset": {"format": "inline", "data": [{"question": "First?"}, {"question": "Second?"}]},
        "budget": {
            "currency": "USD",
            "allowance": "0.12",
            "prices": [
                {
                    "provider": "openai",
                    "model": "budget-test",
                    "endpoint": "https://api.openai.com/v1",
                    "pricing_id": "offline-test-policy-not-real-pricing",
                    "input_cost_per_million": "100",
                    "output_cost_per_million": "200",
                    "maximum_input_tokens": 1000,
                    "maximum_output_tokens": 100,
                    "output_token_parameter": "max_tokens",
                }
            ],
        },
    }, requests


@pytest.mark.parametrize(
    "command,asynchronous", [("run", False), ("run", True), ("harness", False)]
)
def test_budget_stop_is_nonzero_with_partial_artifacts(
    tmp_path, paid_config, command, asynchronous
):
    config, requests = paid_config
    if command == "harness":
        config["models"] = [config.pop("model")]
        config["probes"] = [config.pop("probe")]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    args = [command, str(path), "--run-dir", str(run_dir), "--quiet", "--validate-output"]
    if command == "harness":
        args.append("--skip-report")
    if asynchronous:
        args += ["--async", "--concurrency", "1"]

    assert main(args) == 1
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == 100
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_completed"] is False
    assert manifest["custom"]["abort"]["code"] == "budget_exceeded"
    assert manifest["custom"]["budget"]["aborted"] is True
    assert manifest["custom"]["budget"]["settled"] == "0.002"
    assert manifest["custom"]["health"]["healthy"] is False
    assert main(["validate", str(run_dir), "--quiet"]) == 0


def test_harness_models_share_the_same_total_allowance(tmp_path, paid_config):
    config, requests = paid_config
    model = config.pop("model")
    config["models"] = [model, model]
    config["probes"] = [config.pop("probe")]
    config["dataset"]["data"] = [{"question": "One per model"}]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    assert main(["harness", str(path), "--run-dir", str(run_dir), "--quiet", "--skip-report"]) == 1
    assert len(requests) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["record_count"] == 2
    assert manifest["custom"]["budget"]["aborted"] is True


def test_direct_runner_rejects_a_ledger_not_bound_to_its_model(tmp_path):
    ledger = BudgetLedger(BudgetPolicy(currency="USD", allowance="0", prices=[]))
    runner = ProbeRunner(DummyModel(), LogicProbe())
    with pytest.raises(BudgetUnsupportedError):
        runner.run([{"question": "Do not call"}], budget_ledger=ledger, run_dir=tmp_path / "run")
    assert not (tmp_path / "run" / "records.jsonl").exists()


@pytest.mark.asyncio
async def test_direct_async_runner_rejects_a_ledger_not_bound_to_its_model(tmp_path):
    ledger = BudgetLedger(BudgetPolicy(currency="USD", allowance="0", prices=[]))
    runner = AsyncProbeRunner(DummyModel(), LogicProbe())
    with pytest.raises(BudgetUnsupportedError):
        await runner.run(
            [{"question": "Do not call"}], budget_ledger=ledger, run_dir=tmp_path / "run"
        )
    assert not (tmp_path / "run" / "records.jsonl").exists()
