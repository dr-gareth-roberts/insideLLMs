"""Regression tests for Cluster E fixes (serialization, secrets, config, registry)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from insideLLMs._secrets import REDACTED, redact_config_secrets
from insideLLMs._serialization import (
    StrictSerializationError,
    fingerprint_value,
    serialize_value,
)
from insideLLMs.config_schema import RuntimeDatasetConfig
from insideLLMs.config_types import RunConfigBuilder
from insideLLMs.exceptions import (
    AlreadyRegisteredError,
    NotRegisteredError,
    RegistryError,
)
from insideLLMs.registry import NotFoundError, RegistrationError
from insideLLMs.schemas.custom_trace_v1 import _assert_jsonable
from insideLLMs.schemas.registry import SchemaRegistry, semver_tuple
from insideLLMs.schemas.validator import OutputValidator, _to_plain
from insideLLMs.types import ProbeResult, ResultStatus

# ---------------------------------------------------------------------------
# E1 — serialization
# ---------------------------------------------------------------------------


class TestSerializationStrictNonFinite:
    def test_strict_raises_on_nan(self) -> None:
        with pytest.raises(StrictSerializationError, match="Non-finite"):
            serialize_value(float("nan"), strict=True)

    def test_strict_raises_on_inf(self) -> None:
        with pytest.raises(StrictSerializationError, match="Non-finite"):
            serialize_value(float("inf"), strict=True)

    def test_lenient_maps_nan_to_none(self) -> None:
        assert serialize_value(float("nan"), strict=False) is None

    def test_strict_nan_fingerprint_differs_from_none(self) -> None:
        none_fp = fingerprint_value({"x": None}, strict=True)
        with pytest.raises(StrictSerializationError):
            fingerprint_value({"x": float("nan")}, strict=True)
        assert none_fp is not None

    def test_dict_key_collision_rejected_in_lenient_mode(self) -> None:
        with pytest.raises(StrictSerializationError, match="collision"):
            serialize_value({1: "a", "1": "b"}, strict=False)

    def test_dict_key_collision_rejected_in_strict_mode(self) -> None:
        with pytest.raises(StrictSerializationError, match="collision"):
            serialize_value({1: "a", "1": "b"}, strict=True)


# ---------------------------------------------------------------------------
# E2 — secrets
# ---------------------------------------------------------------------------


class TestSecretRedactionAwsAndOAuth:
    def test_aws_presigned_url_query_params(self) -> None:
        url = (
            "https://bucket.s3.amazonaws.com/obj"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=AKIATEST%2F20240101%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Security-Token=FwoGZXIvYXdzEJr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDsecret"
            "&X-Amz-Signature=abcdef0123456789"
            "&X-Amz-Date=20240101T000000Z"
        )
        redacted = redact_config_secrets(url)
        assert "FwoGZXIvYXdz" not in redacted
        assert "abcdef0123456789" not in redacted
        assert "AKIATEST" not in redacted
        # urlencode percent-encodes brackets in the redaction marker.
        assert "REDACTED" in redacted
        assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in redacted

    def test_oauth_fragment_access_token(self) -> None:
        url = "https://app.example/callback#access_token=ya29.secret-token&token_type=Bearer"
        redacted = redact_config_secrets(url)
        assert "ya29.secret-token" not in redacted
        assert "REDACTED" in redacted
        assert "token_type=Bearer" in redacted

    def test_webhook_secret_key_redacted(self) -> None:
        out = redact_config_secrets({"webhook_secret": "whsec_abc", "name": "ok"})
        assert out["webhook_secret"] == REDACTED
        assert out["name"] == "ok"

    def test_signature_dish_not_over_redacted(self) -> None:
        out = redact_config_secrets({"signature_dish": "pasta carbonara"})
        assert out["signature_dish"] == "pasta carbonara"


# ---------------------------------------------------------------------------
# E3 — custom_trace jsonability
# ---------------------------------------------------------------------------


class TestAssertJsonableOriginalValue:
    def test_rejects_set(self) -> None:
        with pytest.raises(ValueError, match="JSON-serialisable"):
            _assert_jsonable({"s": {"a", "b"}}, where="payload")

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="JSON-serialisable"):
            _assert_jsonable({"n": float("nan")}, where="payload")

    def test_accepts_plain_json(self) -> None:
        value = {"a": 1, "b": [True, None]}
        assert _assert_jsonable(value, where="payload") is value


# ---------------------------------------------------------------------------
# E4 — validator / types
# ---------------------------------------------------------------------------


class TestValidatorToPlainAndEnums:
    def test_result_status_is_str_enum(self) -> None:
        assert isinstance(ResultStatus.SUCCESS, str)
        assert ResultStatus.SUCCESS == "success"
        assert ResultStatus.SUCCESS.value == "success"

    def test_probe_result_with_enum_validates(self) -> None:
        result = ProbeResult(input="q", status=ResultStatus.SUCCESS, output="a")
        validator = OutputValidator()
        validated = validator.validate("ProbeResult", result)
        assert validated.status == "success"

    def test_to_plain_enum_and_nested(self) -> None:
        class Color(Enum):
            RED = "red"

        class Nested(BaseModel):
            color: Color

        plain = _to_plain({"nested": Nested(color=Color.RED), "flag": Color.RED})
        assert plain == {"nested": {"color": "red"}, "flag": "red"}

    def test_to_plain_dataclass_with_enum(self) -> None:
        @dataclass
        class Row:
            status: ResultStatus

        plain = _to_plain(Row(status=ResultStatus.ERROR))
        assert plain == {"status": "error"}

    def test_warn_mode_catches_pydantic_validation_error(self) -> None:
        validator = OutputValidator()
        with pytest.warns(RuntimeWarning):
            out = validator.validate("ProbeResult", {"incomplete": True}, mode="warn")
        assert out == {"incomplete": True}


# ---------------------------------------------------------------------------
# E5 — dataset config incompatible fields + case-insensitive suffix
# ---------------------------------------------------------------------------


class TestDatasetConfigIncompatibleFields:
    def test_inline_rejects_path(self) -> None:
        with pytest.raises(ValidationError, match="incompatible"):
            RuntimeDatasetConfig(format="inline", data=[], path="ignored.jsonl")

    def test_jsonl_rejects_data(self) -> None:
        with pytest.raises(ValidationError, match="incompatible"):
            RuntimeDatasetConfig(format="jsonl", path="x.jsonl", data=[1])

    def test_hf_rejects_path(self) -> None:
        with pytest.raises(ValidationError, match="incompatible"):
            RuntimeDatasetConfig(format="hf", name="x", path="ignored")

    def test_load_config_accepts_uppercase_suffix(self, tmp_path: Path) -> None:
        from insideLLMs.runtime._config_loader import load_config

        cfg = {
            "model": {"type": "dummy"},
            "probe": {"type": "logic"},
            "dataset": {"format": "inline", "data": ["q"]},
        }
        path = tmp_path / "cfg.YAML"
        path.write_text(json.dumps(cfg) if False else __import__("yaml").safe_dump(cfg))
        loaded = load_config(path)
        assert loaded["dataset"]["format"] == "inline"


# ---------------------------------------------------------------------------
# E6 — registry exception hierarchy + plugin rollback + disable flag
# ---------------------------------------------------------------------------


class TestRegistryExceptionHierarchy:
    def test_registration_error_is_registry_error(self) -> None:
        err = RegistrationError("dup")
        assert isinstance(err, AlreadyRegisteredError)
        assert isinstance(err, RegistryError)

    def test_not_found_error_is_registry_error(self) -> None:
        err = NotFoundError("missing")
        assert isinstance(err, NotRegisteredError)
        assert isinstance(err, RegistryError)

    def test_plugin_partial_registration_rolls_back(self) -> None:
        from insideLLMs import registry as reg_mod

        model_registry = reg_mod.model_registry
        name = "_cluster_e_partial_plugin_model"
        if name in model_registry:
            model_registry.unregister(name)

        def bad_plugin(**_kwargs: Any) -> None:
            model_registry.register(name, lambda: "x")
            raise RuntimeError("plugin boom")

        ep = MagicMock()
        ep.name = "bad"
        ep.value = "bad:register"
        ep.load.return_value = bad_plugin

        class _Eps:
            def select(self, group: str) -> list[Any]:
                return [ep]

        with patch.object(reg_mod.metadata, "entry_points", return_value=_Eps()):
            with pytest.warns(RuntimeWarning, match="Failed to load plugin"):
                reg_mod.load_entrypoint_plugins(enabled=True)

        assert name not in model_registry

    def test_disable_plugins_true_uppercase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from insideLLMs import registry as reg_mod

        monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "TRUE")
        assert reg_mod.load_entrypoint_plugins() == {}


# ---------------------------------------------------------------------------
# E7 — semver_tuple + migrate model input
# ---------------------------------------------------------------------------


class TestSemverAndMigrate:
    def test_four_component_falls_back(self) -> None:
        assert semver_tuple("1.2.3.4") == (0, 0, 0)

    def test_negative_component_falls_back(self) -> None:
        assert semver_tuple("-1.2.3") == (0, 0, 0)

    def test_valid_three_component(self) -> None:
        assert semver_tuple("1.2.3") == (1, 2, 3)

    def test_migrate_model_invokes_custom_migration(self) -> None:
        registry = SchemaRegistry()
        model_cls = registry.get_model("ProbeResult", "1.0.0")
        instance = model_cls(input="t", status="success")
        seen: list[Any] = []

        def custom(d: Any) -> Any:
            seen.append(d)
            out = dict(d)
            out["custom_flag"] = True
            return out

        migrated = registry.migrate(
            "ProbeResult",
            instance,
            "1.0.0",
            "1.0.0",
            custom_migration=custom,
        )
        assert seen, "custom_migration was not called"
        assert isinstance(migrated, dict)
        assert migrated.get("custom_flag") is True


# ---------------------------------------------------------------------------
# E8 — RunConfigBuilder.with_timeout
# ---------------------------------------------------------------------------


class TestRunConfigBuilderTimeout:
    def test_with_timeout_propagates(self) -> None:
        config = RunConfigBuilder().with_timeout(12.5).build()
        assert config.timeout == 12.5

    def test_default_timeout_none(self) -> None:
        assert RunConfigBuilder().build().timeout is None


# ---------------------------------------------------------------------------
# E9 — trend missing metric fails closed
# ---------------------------------------------------------------------------


class TestTrendMissingMetric:
    def test_missing_metric_nonzero_exit(self, tmp_path: Path) -> None:
        import argparse

        from insideLLMs.cli.commands.trend import cmd_trend

        idx = tmp_path / "index.jsonl"
        idx.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "timestamp": "2025-01-01T00:00:00",
                    "metrics": {"accuracy": 0.9},
                }
            )
            + "\n"
        )
        args = argparse.Namespace(
            index=str(idx),
            add=None,
            label="",
            last=0,
            format="json",
            metric="not_a_real_metric",
            threshold=0.5,
            fail_on_threshold=True,
        )
        assert cmd_trend(args) == 1


# ---------------------------------------------------------------------------
# E10 — shadow writer failure is best-effort
# ---------------------------------------------------------------------------


class TestShadowWriterBestEffort:
    def test_writer_failure_still_returns_response(self, tmp_path: Path) -> None:
        import asyncio

        from insideLLMs.shadow import fastapi

        middleware = fastapi(
            output_path=tmp_path / "records.jsonl",
            sample_rate=1.0,
            run_id="shadow-e10",
        )
        response = object()

        async def call_next(_req: Any) -> Any:
            return response

        class Req:
            method = "GET"
            url = type("U", (), {"path": "/", "query": ""})()
            headers = {}

            async def body(self) -> bytes:
                return b""

        with patch("insideLLMs.shadow.ShadowWriter.append", side_effect=OSError("disk full")):
            returned = asyncio.run(middleware(Req(), call_next))
        assert returned is response

    def test_writer_failure_preserves_app_exception(self, tmp_path: Path) -> None:
        import asyncio

        from insideLLMs.shadow import fastapi

        middleware = fastapi(
            output_path=tmp_path / "records.jsonl",
            sample_rate=1.0,
            run_id="shadow-e10-err",
        )

        async def call_next(_req: Any) -> Any:
            raise ValueError("app boom")

        class Req:
            method = "GET"
            url = type("U", (), {"path": "/", "query": ""})()
            headers = {}

            async def body(self) -> bytes:
                return b""

        with patch("insideLLMs.shadow.ShadowWriter.append", side_effect=OSError("disk full")):
            with pytest.raises(ValueError, match="app boom"):
                asyncio.run(middleware(Req(), call_next))


# ---------------------------------------------------------------------------
# E11 — tracker end_run helper
# ---------------------------------------------------------------------------


class TestEndTracker:
    def test_end_tracker_calls_end_run_once(self) -> None:
        from insideLLMs.cli.commands._run_common import end_tracker

        tracker = MagicMock()
        end_tracker(tracker, status="finished")
        tracker.end_run.assert_called_once_with(status="finished")

    def test_end_tracker_swallows_errors(self) -> None:
        from insideLLMs.cli.commands._run_common import end_tracker

        tracker = MagicMock()
        tracker.end_run.side_effect = RuntimeError("already ended")
        end_tracker(tracker, status="failed")  # must not raise

    def test_end_tracker_none_is_noop(self) -> None:
        from insideLLMs.cli.commands._run_common import end_tracker

        end_tracker(None, status="failed")
