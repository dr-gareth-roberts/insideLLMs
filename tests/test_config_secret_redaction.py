"""Credentials must not become reproducibility or published provenance data."""

import base64
import json
from pathlib import Path

import pytest

from insideLLMs._secrets import REDACTED, redact_config_secrets
from insideLLMs.attestations.steps.builders import build_attestation_04_execution
from insideLLMs.runtime._config_loader import _build_resolved_config_snapshot
from insideLLMs.runtime._determinism import _deterministic_run_id_from_config_snapshot
from insideLLMs.runtime._ultimate import run_ultimate_post_artifact

SECRET = "sentinel-credential-do-not-persist"


def test_snapshot_redacts_nested_credentials_without_mutating_live_configuration(tmp_path):
    original = {
        "models": [{"type": "dummy", "args": {"api_key": SECRET}}],
        "headers": {"AUTHORIZATION": f"Bearer {SECRET}", "X-API-Key": SECRET},
        "nested": [{"clientSecret": SECRET, "password": SECRET}],
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens": 25,
    }

    snapshot = _build_resolved_config_snapshot(original, tmp_path)

    assert SECRET not in json.dumps(snapshot)
    assert original["models"][0]["args"]["api_key"] == SECRET
    assert snapshot["api_key_env"] == "OPENAI_API_KEY"
    assert snapshot["max_tokens"] == 25


def test_run_identity_does_not_fingerprint_literal_credentials():
    first = {"model": {"type": "dummy", "args": {"api_key": "first-secret"}}}
    second = {"model": {"type": "dummy", "args": {"api_key": "second-secret"}}}

    assert _deterministic_run_id_from_config_snapshot(
        first, schema_version="1.0.1"
    ) == _deterministic_run_id_from_config_snapshot(second, schema_version="1.0.1")


def test_execution_attestation_redacts_direct_builder_input():
    statement = build_attestation_04_execution(
        [],
        runner_config_snapshot={"api_key": SECRET},
        model_identity_snapshot={"extra": {"access_token": SECRET}},
    )

    assert SECRET not in json.dumps(statement)


def test_ultimate_submission_never_contains_configuration_credentials(tmp_path, monkeypatch):
    (tmp_path / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"run_id":"test"}\n', encoding="utf-8")
    submissions = []

    def submit(envelope, service_url):
        submissions.append(json.loads(base64.b64decode(envelope["payload"])))
        return {"receipt": {}, "statement_digest": "test", "service_url": service_url}

    monkeypatch.setattr("insideLLMs.runtime._ultimate.submit_statement", submit)
    run_ultimate_post_artifact(
        tmp_path,
        config_snapshot={"api_key": SECRET},
        scitt_service_url="https://transparency.example.test",
    )

    assert submissions
    assert SECRET not in json.dumps(submissions)
    for path in (tmp_path / "attestations").glob("*.dsse.json"):
        envelope = json.loads(path.read_text())
        assert SECRET not in base64.b64decode(envelope["payload"]).decode()


@pytest.mark.parametrize("use_async", [False, True])
def test_direct_runner_sanitizes_snapshot_before_writing(tmp_path: Path, use_async: bool):
    import asyncio

    from insideLLMs.models import DummyModel
    from insideLLMs.probes import LogicProbe
    from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner

    kwargs = {
        "run_dir": tmp_path / "run",
        "config_snapshot": {"model": {"args": {"api_key": SECRET}}},
        "validate_output": False,
    }
    if use_async:
        runner = AsyncProbeRunner(DummyModel(), LogicProbe())
        asyncio.run(runner.run(["hello"], **kwargs))
    else:
        runner = ProbeRunner(DummyModel(), LogicProbe())
        runner.run(["hello"], **kwargs)

    assert SECRET not in (tmp_path / "run" / "config.resolved.yaml").read_text()


def test_harness_preserves_live_credentials_but_sanitizes_snapshot(tmp_path, monkeypatch):
    import yaml

    from insideLLMs.models import DummyModel
    from insideLLMs.runtime._high_level import run_harness_from_config

    received = []

    def create_model(config, **kwargs):
        received.append(config["args"]["api_key"])
        return DummyModel()

    monkeypatch.setattr("insideLLMs.runtime._high_level._create_model_from_config", create_model)
    (tmp_path / "questions.jsonl").write_text('{"question":"hello"}\n')
    config = {
        "models": [{"type": "dummy", "args": {"api_key": SECRET}}],
        "probes": [{"type": "logic"}],
        "dataset": {"format": "jsonl", "path": "questions.jsonl"},
    }
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = run_harness_from_config(config_path)

    assert received == [SECRET]
    assert SECRET not in json.dumps(result["config_snapshot"])


@pytest.mark.parametrize(
    "name",
    [
        "api_key",
        "X-API-Key",
        "Authorization",
        "clientSecret",
        "password",
        "access_token",
        "api_token",
        "aws_session_token",
        "X-Auth-Token",
        "HF_TOKEN",
    ],
)
def test_secrets_are_recursively_redacted_in_lists_and_tuples(name):
    value = ({"nested": [{name: SECRET}]},)
    assert SECRET not in json.dumps(redact_config_secrets(value))
    assert value[0]["nested"][0][name] == SECRET


def test_url_auth_query_aliases_are_scrubbed_without_masking_tokenizer_settings():
    value = {
        "url": f"https://example.test?key={SECRET}&sig={SECRET}&signature={SECRET}",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
    }
    scrubbed = redact_config_secrets(value)
    assert SECRET not in json.dumps(scrubbed)
    assert scrubbed["bos_token"] == "<bos>"
    assert scrubbed["eos_token"] == "<eos>"


def test_direct_run_identity_does_not_fingerprint_generation_credentials():
    from insideLLMs.runtime._determinism import _deterministic_run_id_from_inputs

    def identity(secret):
        return _deterministic_run_id_from_inputs(
            schema_version="1.0.2",
            model_spec={"model_id": "dummy"},
            probe_spec={"probe_id": "logic"},
            dataset_spec={},
            prompt_set=["question"],
            probe_kwargs={"api_key": secret},
        )

    assert identity("first-secret") == identity("rotated-secret")


def test_urls_and_secret_value_objects_cannot_evade_redaction():
    from pydantic import SecretStr

    data = {
        "base_url": f"https://user:{SECRET}@example.test/api?api_key={SECRET}&version=1",
        "opaque": SecretStr(SECRET),
        "api_key_env": f"Bearer {SECRET}",
        "token_budget": 100,
        "max_tokens": 20,
    }

    scrubbed = redact_config_secrets(data)

    assert SECRET not in json.dumps(scrubbed)
    assert "version=1" in scrubbed["base_url"]
    assert scrubbed["opaque"] == REDACTED
    assert scrubbed["api_key_env"] == REDACTED
    assert scrubbed["token_budget"] == 100
    assert redact_config_secrets(scrubbed) == scrubbed


def test_local_tracker_persists_only_sanitized_parameters(tmp_path):
    from insideLLMs.experiment_tracking import LocalFileTracker

    tracker = LocalFileTracker(output_dir=str(tmp_path / "tracking"))
    tracker.start_run(run_name="redaction")
    tracker.log_params({"model": {"args": {"api_key": SECRET}}, "max_tokens": 10})
    tracker.end_run()

    files = list((tmp_path / "tracking").rglob("*.json"))
    assert files
    assert all(SECRET not in path.read_text() for path in files)


def test_model_metadata_cannot_reintroduce_configuration_secrets():
    from insideLLMs.runtime._result_utils import _build_model_spec, _coerce_model_info
    from insideLLMs.types import ModelInfo

    class CustomModel:
        def info(self):
            return ModelInfo("custom", "custom", "custom", extra={"api_key": SECRET})

    model = CustomModel()
    assert SECRET not in json.dumps(_build_model_spec(model))
    assert SECRET not in json.dumps(_coerce_model_info(model).extra)


def test_dataclass_and_pydantic_configuration_values_are_recursively_scrubbed():
    from dataclasses import dataclass

    from pydantic import BaseModel

    from insideLLMs._serialization import stable_json_dumps

    @dataclass
    class Connection:
        api_key: str
        endpoint: str

    class ProviderSettings(BaseModel):
        access_token: str

    original = {
        "connection": Connection(SECRET, "https://example.test"),
        "provider": ProviderSettings(access_token=SECRET),
    }

    scrubbed = redact_config_secrets(original)

    assert SECRET not in stable_json_dumps(scrubbed)
    assert scrubbed["connection"]["endpoint"] == "https://example.test"
    assert original["connection"].api_key == SECRET
    assert original["provider"].access_token == SECRET


@pytest.mark.parametrize("command", ["run", "harness"])
def test_cli_artifact_bundle_contains_no_live_configuration_credentials(
    tmp_path, monkeypatch, command
):
    from dataclasses import replace

    import yaml

    from insideLLMs.cli import main
    from insideLLMs.models import DummyModel

    received = []

    def create_model(config, **kwargs):
        received.append(config["args"]["api_key"])
        model = DummyModel(canned_response="Paris")
        info = replace(model.info(), extra={"api_key": SECRET})
        monkeypatch.setattr(model, "info", lambda: info)
        return model

    monkeypatch.setattr("insideLLMs.runtime._high_level._create_model_from_config", create_model)
    model = {"type": "dummy", "args": {"api_key": SECRET}}
    probe = {"type": "logic"}
    config = {
        "dataset": {
            "format": "inline",
            "data": [{"question": "Capital of France?", "reference_answer": "Paris"}],
        }
    }
    config.update(
        {"models": [model], "probes": [probe]}
        if command == "harness"
        else {"model": model, "probe": probe}
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "run"

    assert main([command, str(config_path), "--run-dir", str(run_dir)]) == 0
    assert received == [SECRET]
    artifacts = [path for path in run_dir.rglob("*") if path.is_file()]
    assert {"manifest.json", "records.jsonl", "config.resolved.yaml"} <= {
        path.name for path in artifacts
    }
    assert all(SECRET not in path.read_text() for path in artifacts)
