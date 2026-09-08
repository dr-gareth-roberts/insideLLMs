"""Provider discovery reports prerequisites without claiming live verification."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from insideLLMs import registry
from insideLLMs.cli.commands.doctor import cmd_doctor
from insideLLMs.models.catalogue import PROVIDER_CATALOGUE


def _doctor_models(capsys: pytest.CaptureFixture[str]) -> dict[str, dict]:
    status = cmd_doctor(argparse.Namespace(format="json", fail_on_warn=False, capabilities=True))
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    return {item["name"]: item for item in payload["capabilities"]["models"]}


def test_openrouter_is_a_builtin_with_explicit_unchecked_live_verification(capsys):
    model = _doctor_models(capsys)["openrouter"]

    assert model["source"] == "builtin"
    assert model["metadata_status"] == "declared"
    assert model["live_verification"] == "not_checked"
    assert model["credential_alternatives"] == [["OPENROUTER_API_KEY"]]
    assert model["declared_capabilities"]["stream"] == "native"
    assert model["budget_support"] == "unknown"


def test_catalogue_and_builtin_registry_are_immutable_metadata_not_constructor_kwargs():
    assert set(PROVIDER_CATALOGUE) == {
        "dummy",
        "openai",
        "openrouter",
        "anthropic",
        "gemini",
        "cohere",
        "huggingface",
        "llamacpp",
        "ollama",
        "vllm",
    }
    for name in PROVIDER_CATALOGUE:
        assert registry.model_registry.get_factory(name) is registry.get_builtin_model_factory(name)
        assert registry.model_registry.info(name)["default_kwargs"] == {}
    with pytest.raises(TypeError):
        PROVIDER_CATALOGUE["another"] = PROVIDER_CATALOGUE["dummy"]
    with pytest.raises(FrozenInstanceError):
        PROVIDER_CATALOGUE["dummy"].name = "changed"
    with pytest.raises(FrozenInstanceError):
        PROVIDER_CATALOGUE["dummy"].capabilities.stream = "unsupported"
    assert registry.model_registry.get("dummy", canned_response="test").generate("q") == "test"


@pytest.mark.parametrize("credential", ["CO_API_KEY", "COHERE_API_KEY"])
def test_cohere_accepts_either_environment_credential_without_exposing_it(
    credential, monkeypatch, capsys
):
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv(credential, "test-secret-not-for-output")

    model = _doctor_models(capsys)["cohere"]

    assert model["credential_ready"] is True
    assert model["missing_credentials"] == []
    assert "test-secret-not-for-output" not in json.dumps(model)


@pytest.mark.parametrize("provider,module", [("ollama", "ollama"), ("vllm", "openai")])
def test_local_service_adapters_report_missing_client_sdk(provider, module, monkeypatch, capsys):
    # Module discovery is an import-system boundary, without importing an SDK.
    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == module else original(name, *args),
    )

    model = _doctor_models(capsys)[provider]

    assert model["dependency_ready"] is False
    assert model["missing_dependencies"] == [module]
    assert model["dependencies"] == [{"module": module, "distribution": module, "available": False}]
    assert model["external_requirements"]
    assert model["live_verification"] == "not_checked"


@pytest.mark.parametrize("name", ["uncatalogued_plugin", "openai"])
def test_unknown_or_replaced_plugin_is_not_reported_as_ready(name, capsys):
    original = (
        registry.model_registry.get_factory(name) if name in registry.model_registry else None
    )

    def forbidden_factory(**kwargs):
        pytest.fail("Doctor must not construct a model")

    registry.model_registry.register(name, forbidden_factory, overwrite=True)
    try:
        model = _doctor_models(capsys)[name]
    finally:
        registry.model_registry.unregister(name)
        if original is not None:
            registry.model_registry.register(name, original)

    assert model["source"] == "plugin"
    assert model["metadata_status"] == "unknown"
    assert model["status"] == "unknown"
    assert model["dependency_ready"] is None
    assert model["credential_ready"] is None
    assert model["declared_capabilities"] is None
    assert model["live_verification"] == "not_checked"


def test_doctor_discovers_entrypoints_without_loading_plugin_code(monkeypatch, capsys):
    loaded = []

    class EntryPoint:
        name = "untrusted-plugin"
        value = "plugin:register"

        def load(self):
            loaded.append(self.name)
            return lambda: None

    class EntryPoints:
        def select(self, *, group):
            return [EntryPoint()] if group == registry.PLUGIN_ENTRYPOINT_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", EntryPoints)
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    monkeypatch.delenv("INSIDELLMS_DISABLE_PLUGINS", raising=False)

    _doctor_models(capsys)

    assert loaded == []


def test_declared_simulation_does_not_change_runtime_dispatch_contract(capsys):
    model = _doctor_models(capsys)["huggingface"]

    assert model["declared_capabilities"] == {
        "generate": "native",
        "chat": "simulated",
        "stream": "simulated",
        "batch_generate": "simulated",
        "agenerate": "unsupported",
        "achat": "unsupported",
        "astream": "unsupported",
    }


def test_doctor_is_offline_and_does_not_import_provider_adapters_or_sdks():
    script = """
import argparse
import contextlib
import io
import json
import socket
import sys

def deny_network(*args, **kwargs):
    raise AssertionError("Doctor attempted network access")

socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network

from insideLLMs.cli.commands.doctor import cmd_doctor

with contextlib.redirect_stdout(io.StringIO()) as output:
    status = cmd_doctor(argparse.Namespace(format="json", capabilities=True, fail_on_warn=False))
assert status == 0
payload = json.loads(output.getvalue())
assert "test-credential-do-not-print" not in output.getvalue()
assert all(item["live_verification"] == "not_checked" for item in payload["capabilities"]["models"])
for module in (
    "insideLLMs.models.openai", "insideLLMs.models.openrouter", "insideLLMs.models.anthropic",
    "insideLLMs.models.gemini", "insideLLMs.models.cohere", "insideLLMs.models.huggingface",
    "insideLLMs.models.local", "openai", "anthropic", "cohere", "google.generativeai", "ollama",
):
    assert module not in sys.modules, f"Doctor imported {module}"
print("offline provider discovery verified")
"""
    result = subprocess.run(  # noqa: S603 - fixed local test script, no provider calls
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "INSIDELLMS_DISABLE_PLUGINS": "1",
            "OPENAI_API_KEY": "test-credential-do-not-print",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline provider discovery verified"


def test_dummy_simulates_streaming_while_runtime_stream_dispatch_remains_callable(capsys):
    from insideLLMs.models.base import can_stream

    model = _doctor_models(capsys)["dummy"]

    assert model["declared_capabilities"]["stream"] == "simulated"
    assert can_stream(registry.model_registry.get("dummy")) is True
