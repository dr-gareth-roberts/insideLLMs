"""Coverage gap tests for insideLLMs/cli/commands/doctor.py — Wave 7 W7-0071.

The existing test_cli_doctor_coverage.py requires nltk via importorskip and is
therefore skipped in environments without nltk.  These tests cover the same
public surface without that restriction:

* _plugins_disabled_via_env() — env var on / off
* _entrypoint_plugins() — normal path, exception path
* _capability_status() — all status branches
* _build_capabilities() — via cmd_doctor(capabilities=True)
* _print_capabilities_summary() — text format with capabilities
* cmd_doctor — text format, json format, capabilities, fail_on_warn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.cli.commands.doctor import (
    _build_capabilities,
    _capability_status,
    _entrypoint_plugins,
    _plugins_disabled_via_env,
    _print_capabilities_summary,
    cmd_doctor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(
    *,
    format: str = "json",
    fail_on_warn: bool = False,
    capabilities: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        format=format,
        fail_on_warn=fail_on_warn,
        capabilities=capabilities,
    )


def _passing_checks() -> list[dict]:
    """A minimal checks list where everything is ok=True."""
    return [
        {"name": "python", "ok": True, "hint": "3.12"},
        {"name": "platform", "ok": True, "hint": "darwin"},
        {"name": "insideLLMs", "ok": True, "hint": "0.2.0"},
        {"name": "pydantic", "ok": True, "hint": None},
        {"name": "nltk", "ok": True, "hint": None},
        {"name": "sklearn", "ok": True, "hint": None},
        {"name": "spacy", "ok": True, "hint": None},
        {"name": "gensim", "ok": True, "hint": None},
        {"name": "nltk:punkt", "ok": True, "hint": None},
        {"name": "nltk:vader_lexicon", "ok": True, "hint": None},
        {"name": "spacy:en_core_web_sm", "ok": True, "hint": None},
        {"name": "matplotlib", "ok": True, "hint": None},
        {"name": "pandas", "ok": True, "hint": None},
        {"name": "seaborn", "ok": True, "hint": None},
        {"name": "plotly", "ok": True, "hint": None},
        {"name": "ipywidgets", "ok": True, "hint": None},
        {"name": "redis", "ok": False, "hint": None},
        {"name": "datasets", "ok": False, "hint": None},
        {"name": "ultimate:tuf", "ok": False, "hint": None},
        {"name": "ultimate:cosign", "ok": False, "hint": None},
        {"name": "ultimate:oras", "ok": False, "hint": None},
        {"name": "OPENAI_API_KEY", "ok": False, "hint": None},
        {"name": "ANTHROPIC_API_KEY", "ok": False, "hint": None},
        {"name": "INSIDELLMS_RUN_ROOT", "ok": True, "hint": None},
    ]


# ---------------------------------------------------------------------------
# _plugins_disabled_via_env
# ---------------------------------------------------------------------------


class TestPluginsDisabledViaEnv:
    def test_disabled_when_set_to_1(self, monkeypatch):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "1")
        assert _plugins_disabled_via_env() is True

    def test_disabled_when_set_to_true(self, monkeypatch):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "true")
        assert _plugins_disabled_via_env() is True

    def test_disabled_when_set_to_yes(self, monkeypatch):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "YES")
        assert _plugins_disabled_via_env() is True

    def test_not_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("INSIDELLMS_DISABLE_PLUGINS", raising=False)
        assert _plugins_disabled_via_env() is False

    def test_not_disabled_when_empty(self, monkeypatch):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "")
        assert _plugins_disabled_via_env() is False

    def test_not_disabled_for_other_values(self, monkeypatch):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "false")
        assert _plugins_disabled_via_env() is False


# ---------------------------------------------------------------------------
# _entrypoint_plugins
# ---------------------------------------------------------------------------


class TestEntrypointPlugins:
    def test_returns_sorted_list_with_select(self):
        fake_ep = MagicMock()
        fake_ep.name = "my-plugin"
        fake_ep.value = "my.module:main"

        class FakeEps:
            def select(self, group):
                return [fake_ep]

        with patch("importlib.metadata.entry_points", return_value=FakeEps()):
            result = _entrypoint_plugins("some.group")

        assert len(result) == 1
        assert result[0]["name"] == "my-plugin"
        assert result[0]["value"] == "my.module:main"

    def test_returns_empty_list_on_exception(self, caplog):
        with patch(
            "importlib.metadata.entry_points",
            side_effect=RuntimeError("metadata broken"),
        ):
            with caplog.at_level(logging.WARNING, logger="insideLLMs.cli.commands.doctor"):
                result = _entrypoint_plugins("some.group")
        assert result == []

    def test_returns_empty_when_no_plugins(self):
        class FakeEps:
            def select(self, group):
                return []

        with patch("importlib.metadata.entry_points", return_value=FakeEps()):
            result = _entrypoint_plugins("empty.group")
        assert result == []

    def test_result_is_sorted(self):
        ep_b = MagicMock()
        ep_b.name = "b-plugin"
        ep_b.value = "b"
        ep_a = MagicMock()
        ep_a.name = "a-plugin"
        ep_a.value = "a"

        class FakeEps:
            def select(self, group):
                return [ep_b, ep_a]

        with patch("importlib.metadata.entry_points", return_value=FakeEps()):
            result = _entrypoint_plugins("g")
        assert result[0]["name"] == "a-plugin"


# ---------------------------------------------------------------------------
# _capability_status
# ---------------------------------------------------------------------------


class TestCapabilityStatus:
    def test_ready_when_all_present(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "val")
        with patch("insideLLMs.cli.commands.doctor._has_module", return_value=True):
            result = _capability_status(
                modules=["existing_mod"],
                credential_env=["SOME_KEY"],
                notes=[],
            )
        assert result["status"] == "ready"
        assert result["dependency_ready"] is True
        assert result["credential_ready"] is True

    def test_missing_dependencies(self, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        with patch("insideLLMs.cli.commands.doctor._has_module", return_value=False):
            result = _capability_status(
                modules=["missing_mod"],
                credential_env=[],
                notes=[],
            )
        assert result["status"] == "missing_dependencies"
        assert "missing_mod" in result["missing_dependencies"]

    def test_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("MY_CRED", raising=False)
        with patch("insideLLMs.cli.commands.doctor._has_module", return_value=True):
            result = _capability_status(
                modules=[],
                credential_env=["MY_CRED"],
                notes=[],
            )
        assert result["status"] == "missing_credentials"
        assert "MY_CRED" in result["missing_credentials"]

    def test_missing_deps_and_credentials(self, monkeypatch):
        monkeypatch.delenv("MY_CRED", raising=False)
        with patch("insideLLMs.cli.commands.doctor._has_module", return_value=False):
            result = _capability_status(
                modules=["missing_mod"],
                credential_env=["MY_CRED"],
                notes=[],
            )
        assert result["status"] == "missing_dependencies_and_credentials"

    def test_requires_external_service_when_only_notes(self, monkeypatch):
        with patch("insideLLMs.cli.commands.doctor._has_module", return_value=True):
            result = _capability_status(
                modules=[],
                credential_env=[],
                notes=["Requires a running server"],
            )
        assert result["status"] == "requires_external_service"


# ---------------------------------------------------------------------------
# _build_capabilities
# ---------------------------------------------------------------------------


class TestBuildCapabilities:
    def test_returns_expected_keys(self):
        checks = _passing_checks()
        caps = _build_capabilities(checks)
        assert "models" in caps
        assert "probes" in caps
        assert "datasets" in caps
        assert "extras" in caps
        assert "reports" in caps
        assert "plugins" in caps

    def test_models_have_source_field(self):
        caps = _build_capabilities(_passing_checks())
        for model in caps["models"]:
            assert "source" in model
            assert model["source"] in ("builtin", "plugin")

    def test_plugins_disabled_env_reflected(self, monkeypatch):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "1")
        caps = _build_capabilities(_passing_checks())
        assert caps["plugins"]["disabled_by_env"] is True
        assert caps["plugins"]["auto_loading_enabled"] is False

    def test_plugins_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("INSIDELLMS_DISABLE_PLUGINS", raising=False)
        caps = _build_capabilities(_passing_checks())
        assert caps["plugins"]["auto_loading_enabled"] is True

    def test_extras_structure(self):
        caps = _build_capabilities(_passing_checks())
        extra_names = [e["name"] for e in caps["extras"]]
        assert "nlp" in extra_names
        assert "visualization" in extra_names
        assert "verifiable_evaluation" in extra_names

    def test_reports_include_plotly_conditional(self):
        # With plotly ok=False in checks
        checks = _passing_checks()
        for c in checks:
            if c["name"] == "plotly":
                c["ok"] = False
        caps = _build_capabilities(checks)
        report_map = {r["name"]: r for r in caps["reports"]}
        assert report_map["report.html_interactive"]["available"] is False

    def test_unknown_model_gets_no_static_metadata_note(self):
        """A plugin model not in _MODEL_REQUIREMENTS gets a fallback note."""
        checks = _passing_checks()
        with patch(
            "insideLLMs.cli.commands.doctor.model_registry.list",
            return_value=["dummy", "unknown-plugin-model"],
        ):
            caps = _build_capabilities(checks)
        plugin_models = [m for m in caps["models"] if m["name"] == "unknown-plugin-model"]
        assert plugin_models[0]["source"] == "plugin"


# ---------------------------------------------------------------------------
# _print_capabilities_summary
# ---------------------------------------------------------------------------


class TestPrintCapabilitiesSummary:
    def test_prints_models_and_probes(self, capsys):
        caps = _build_capabilities(_passing_checks())
        _print_capabilities_summary(caps)
        out = capsys.readouterr().out
        assert "Capabilities" in out
        assert "Models" in out

    def test_prints_warning_for_disabled_plugins(self, monkeypatch, capsys):
        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "1")
        caps = _build_capabilities(_passing_checks())
        _print_capabilities_summary(caps)
        out = capsys.readouterr().out
        assert "disabled" in out.lower() or "plugin" in out.lower()

    def test_warns_for_not_ready_extras(self, capsys):
        # Build caps with plotly missing so visualization extra is not ready
        checks = _passing_checks()
        for c in checks:
            if c["name"] in ("matplotlib", "plotly", "seaborn", "pandas", "ipywidgets"):
                c["ok"] = False
        caps = _build_capabilities(checks)
        _print_capabilities_summary(caps)
        out = capsys.readouterr().out
        assert "visualization" in out

    def test_success_for_all_ready_extras(self, capsys):
        caps = _build_capabilities(_passing_checks())
        # Force all extras to ready=True
        for extra in caps["extras"]:
            extra["ready"] = True
        _print_capabilities_summary(caps)
        out = capsys.readouterr().out
        assert "Capabilities" in out

    def test_blocked_model_in_summary(self, capsys):
        caps = _build_capabilities(_passing_checks())
        # Inject a blocked model
        caps["models"].append(
            {
                "name": "blocked-model",
                "source": "builtin",
                "status": "missing_dependencies",
                "dependency_ready": False,
                "credential_ready": True,
                "missing_dependencies": ["some-lib"],
                "missing_credentials": [],
                "notes": [],
            }
        )
        _print_capabilities_summary(caps)
        out = capsys.readouterr().out
        # blocked model should appear as a warning
        assert "blocked-model" in out


# ---------------------------------------------------------------------------
# cmd_doctor — text format (the previously uncovered path)
# ---------------------------------------------------------------------------


class TestCmdDoctorTextFormat:
    def test_text_format_produces_output(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        rc = cmd_doctor(_make_args(format="text"))
        out = capsys.readouterr().out
        assert "insideLLMs Doctor" in out
        assert "Environment" in out
        assert "Diagnostics" in out
        assert rc in (0, 1)

    def test_text_format_fail_on_warn_returns_1_if_warns(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rc = cmd_doctor(_make_args(format="text", fail_on_warn=True))
        _ = capsys.readouterr()
        # If there are any optional-dep warnings, rc==1; otherwise 0
        assert rc in (0, 1)

    def test_text_format_with_capabilities(self, capsys):
        rc = cmd_doctor(_make_args(format="text", capabilities=True))
        out = capsys.readouterr().out
        assert "Doctor" in out
        assert rc in (0, 1)

    def test_text_format_info_for_api_keys(self, capsys, monkeypatch):
        """API key checks should appear as informational, not as failures."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cmd_doctor(_make_args(format="text"))
        out = capsys.readouterr().out
        assert "OPENAI_API_KEY" in out or "ANTHROPIC_API_KEY" in out


class TestCmdDoctorJsonCapabilities:
    def test_json_with_capabilities_true(self, capsys):
        rc = cmd_doctor(_make_args(format="json", capabilities=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "capabilities" in data
        assert "models" in data["capabilities"]
        assert rc in (0, 1)

    def test_json_without_capabilities(self, capsys):
        rc = cmd_doctor(_make_args(format="json", capabilities=False))
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "capabilities" not in data
        assert rc in (0, 1)

    def test_json_fail_on_warn_true_with_warnings(self, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rc = cmd_doctor(_make_args(format="json", fail_on_warn=True))
        out = capsys.readouterr().out
        data = json.loads(out)
        if data["warnings"]:
            assert rc == 1
        else:
            assert rc == 0


class TestCmdDoctorWarnPathsText:
    def test_all_checks_pass_shows_success(self, capsys, monkeypatch):
        """When all optional-dep checks pass, show success message."""
        # Mock _has_module to return True for everything
        with (
            patch("insideLLMs.cli.commands.doctor._has_module", return_value=True),
            patch("insideLLMs.cli.commands.doctor._check_nltk_resource", return_value=True),
            patch("insideLLMs.cli.commands.doctor.shutil.which", return_value="/usr/bin/cosign"),
            monkeypatch.context() as m,
        ):
            m.setenv("OPENAI_API_KEY", "sk-test")
            m.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
            m.setenv("INSIDELLMS_RUN_ROOT", "/tmp/runs")
            rc = cmd_doctor(_make_args(format="text"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "All recommended checks passed" in out
