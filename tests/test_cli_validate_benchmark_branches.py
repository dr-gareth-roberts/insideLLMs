import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from insideLLMs.cli.commands.benchmark import cmd_benchmark
from insideLLMs.cli.commands.validate import cmd_validate
from insideLLMs.exceptions import ProbeExecutionError


def _validate_args(config: Path, **overrides):
    defaults = {
        "config": str(config),
        "mode": "strict",
        "schema_version": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _benchmark_args(**overrides):
    defaults = {
        "models": "dummy",
        "probes": "dummy",
        "datasets": None,
        "max_examples": 3,
        "output": None,
        "html_report": False,
        "verbose": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cmd_validate_run_dir_manifest_missing_and_parse_error_modes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rc_missing_manifest = cmd_validate(_validate_args(run_dir, mode="strict"))
    assert rc_missing_manifest == 1

    manifest = run_dir / "manifest.json"
    manifest.write_text("not-json")

    rc_parse_warn = cmd_validate(_validate_args(manifest, mode="warn"))
    rc_parse_strict = cmd_validate(_validate_args(manifest, mode="strict"))
    assert rc_parse_warn == 0
    assert rc_parse_strict == 1


def test_cmd_validate_run_dir_schema_and_record_errors(tmp_path):
    class _FakeOutputValidationError(Exception):
        pass

    class _FakeRegistry:
        RUN_MANIFEST = "RunManifest"
        RESULT_RECORD = "ResultRecord"

    class _FakeValidator:
        def __init__(self, _registry):
            pass

        def validate(self, schema_name, obj, **_kwargs):
            if schema_name == _FakeRegistry.RUN_MANIFEST and obj.get("manifest_fail"):
                raise _FakeOutputValidationError("manifest invalid")
            if schema_name == _FakeRegistry.RESULT_RECORD and obj.get("record_fail"):
                raise _FakeOutputValidationError("record invalid")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"

    with patch("insideLLMs.schemas.SchemaRegistry", _FakeRegistry), patch(
        "insideLLMs.schemas.OutputValidator", _FakeValidator
    ), patch("insideLLMs.schemas.OutputValidationError", _FakeOutputValidationError):
        # Manifest schema failure strict mode -> failure.
        manifest_path.write_text(json.dumps({"manifest_fail": True, "records_file": "records.jsonl"}))
        records_path.write_text(json.dumps({"ok": True}) + "\n")
        rc_manifest_fail = cmd_validate(_validate_args(run_dir, mode="strict"))
        assert rc_manifest_fail == 1

        # Missing records file warn mode -> success with warnings.
        manifest_path.write_text(json.dumps({"records_file": "missing.jsonl"}))
        rc_missing_records_warn = cmd_validate(_validate_args(run_dir, mode="warn"))
        assert rc_missing_records_warn == 0

        # Bad JSON record warn mode continues and returns 0 with warnings.
        manifest_path.write_text(json.dumps({"records_file": "records.jsonl"}))
        records_path.write_text("bad-json\n" + json.dumps({"ok": True}) + "\n")
        rc_bad_json_warn = cmd_validate(_validate_args(run_dir, mode="warn"))
        assert rc_bad_json_warn == 0

        # Record schema mismatch strict mode returns failure.
        records_path.write_text(json.dumps({"record_fail": True}) + "\n")
        rc_record_fail_strict = cmd_validate(_validate_args(run_dir, mode="strict"))
        assert rc_record_fail_strict == 1

        # Error while opening records file is handled by mode.
        with patch("insideLLMs.cli.commands.validate.open", side_effect=OSError("cannot open")):
            rc_open_error_warn = cmd_validate(_validate_args(run_dir, mode="warn"))
            rc_open_error_strict = cmd_validate(_validate_args(run_dir, mode="strict"))
            assert rc_open_error_warn == 0
            assert rc_open_error_strict == 1


def test_cmd_validate_config_validation_errors_and_success_paths(tmp_path):
    # Missing model/probe and no dataset -> strict config errors.
    missing_fields = tmp_path / "missing.yaml"
    missing_fields.write_text(yaml.safe_dump({}))
    rc_missing = cmd_validate(_validate_args(missing_fields, mode="strict"))
    assert rc_missing == 1

    # Missing model.type and probe.type errors.
    missing_types = tmp_path / "missing_types.yaml"
    missing_types.write_text(yaml.safe_dump({"model": {}, "probe": {}, "dataset": {}}))
    rc_missing_types = cmd_validate(_validate_args(missing_types, mode="strict"))
    assert rc_missing_types == 1

    # Unknown model/probe types.
    unknown_types = tmp_path / "unknown.yaml"
    unknown_types.write_text(
        yaml.safe_dump(
            {
                "model": {"type": "unknown-model"},
                "probe": {"type": "unknown-probe"},
                "dataset": {"path": str(tmp_path / "nope.jsonl")},
            }
        )
    )

    with patch("insideLLMs.cli.commands.validate.model_registry.list", return_value=[]), patch(
        "insideLLMs.cli.commands.validate.probe_registry.list", return_value=[]
    ):
        rc_unknown = cmd_validate(_validate_args(unknown_types, mode="strict"))
    assert rc_unknown == 1

    # Valid config with only warning (missing dataset path) still returns 0.
    warning_only = tmp_path / "warning_only.yaml"
    warning_only.write_text(
        yaml.safe_dump(
            {
                "model": {"type": "dummy"},
                "probe": {"type": "dummy"},
                "dataset": {"path": str(tmp_path / "missing_data.jsonl")},
            }
        )
    )

    with patch("insideLLMs.cli.commands.validate.model_registry.list", return_value=["dummy"]), patch(
        "insideLLMs.cli.commands.validate.probe_registry.list", return_value=["dummy"]
    ):
        rc_warning_only = cmd_validate(_validate_args(warning_only, mode="strict"))
    assert rc_warning_only == 0


def test_cmd_validate_config_parse_exception_returns_1(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("model: [")
    rc = cmd_validate(_validate_args(broken, mode="strict"))
    assert rc == 1


def test_cmd_benchmark_explicit_datasets_and_html_report(tmp_path):
    examples = [
        SimpleNamespace(input_text="q1"),
        SimpleNamespace(input_text="q2"),
        SimpleNamespace(input_text="q3"),
    ]

    class _Model:
        pass

    class _Probe:
        pass

    class _Runner:
        def __init__(self, _model, _probe):
            self.calls = 0

        def run_single(self, _inp):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(status="success")
            if self.calls == 2:
                return {"status": "success"}
            return {"status": "error"}

    fake_dataset = SimpleNamespace(sample=lambda _n, seed=42: list(examples))

    with patch("insideLLMs.benchmark_datasets.load_builtin_dataset", return_value=fake_dataset), patch(
        "insideLLMs.cli.commands.benchmark.model_registry.get", return_value=_Model
    ), patch(
        "insideLLMs.cli.commands.benchmark.probe_registry.get", return_value=_Probe
    ), patch("insideLLMs.runtime.runner.ProbeRunner", _Runner):
        rc = cmd_benchmark(
            _benchmark_args(
                datasets="custom",
                output=str(tmp_path / "out"),
                html_report=True,
                models="dummy",
                probes="dummy",
            )
        )

    assert rc == 0
    results_file = tmp_path / "out" / "benchmark_results.json"
    assert results_file.exists()
    payload = json.loads(results_file.read_text())
    assert payload[0]["success"] == 2
    assert payload[0]["total"] == 3


def test_cmd_benchmark_probe_load_failure_and_runner_errors(tmp_path):
    examples = [SimpleNamespace(input_text="q1"), SimpleNamespace(input_text="q2")]
    fake_suite = SimpleNamespace(sample=lambda _n, seed=42: list(examples))

    class _Runner:
        def __init__(self, _model, _probe):
            self.calls = 0

        def run_single(self, _inp):
            self.calls += 1
            if self.calls == 1:
                raise ProbeExecutionError("goodprobe", "probe failed")
            if self.calls == 2:
                raise RuntimeError("runtime failed")
            return {"status": "success"}

    def _probe_get(name):
        if name == "badprobe":
            raise RuntimeError("cannot load probe")
        return SimpleNamespace()

    with patch(
        "insideLLMs.benchmark_datasets.create_comprehensive_benchmark_suite",
        return_value=fake_suite,
    ), patch("insideLLMs.cli.commands.benchmark.model_registry.get", return_value=SimpleNamespace()), patch(
        "insideLLMs.cli.commands.benchmark.probe_registry.get",
        side_effect=_probe_get,
    ), patch("insideLLMs.runtime.runner.ProbeRunner", _Runner):
        rc = cmd_benchmark(
            _benchmark_args(models="dummy", probes="badprobe,goodprobe", max_examples=2)
        )

    assert rc == 0


def test_cmd_benchmark_outer_exception_verbose_returns_1(capsys):
    with patch(
        "insideLLMs.benchmark_datasets.create_comprehensive_benchmark_suite",
        side_effect=RuntimeError("suite boom"),
    ), patch("traceback.print_exc") as print_exc:
        rc = cmd_benchmark(_benchmark_args(verbose=True))

    captured = capsys.readouterr()
    assert rc == 1
    assert "Benchmark error" in captured.err
    assert print_exc.called
