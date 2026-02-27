"""Schema validation tests for versioned serialized outputs.

These tests are intentionally lightweight and skip when the optional Pydantic
dependency is not available.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

pytestmark = pytest.mark.contract


def _has_pydantic() -> bool:
    return importlib.util.find_spec("pydantic") is not None


def test_benchmark_outputs_validate_against_schemas(tmp_path):
    if not _has_pydantic():
        import pytest

        pytest.skip("pydantic not installed")

    from insideLLMs.benchmark import ModelBenchmark, ProbeBenchmark
    from insideLLMs.models import DummyModel
    from insideLLMs.probes import BiasProbe, LogicProbe
    from insideLLMs.schemas import OutputValidator, SchemaRegistry

    registry = SchemaRegistry()
    validator = OutputValidator(registry)

    # ModelBenchmark -> BenchmarkSummary
    mb = ModelBenchmark([DummyModel(name="BenchDummy")], LogicProbe(), name="Bench")
    mb_out = mb.run(["Test?"])
    validator.validate(registry.BENCHMARK_SUMMARY, mb_out, schema_version="1.0.0")

    # compare_models -> ComparisonReport
    cmp_out = mb.compare_models()
    validator.validate(registry.COMPARISON_REPORT, cmp_out, schema_version="1.0.0")

    # ProbeBenchmark -> BenchmarkSummary
    pb = ProbeBenchmark(
        DummyModel(name="ProbeDummy"),
        [LogicProbe(), BiasProbe()],
        name="ProbeBench",
    )
    pb_out = pb.run(["Test?"])
    validator.validate(registry.BENCHMARK_SUMMARY, pb_out, schema_version="1.0.0")

    # Round-trip via JSON to ensure schema validation accepts serialized form
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(mb_out, default=str))
    loaded = json.loads(path.read_text())
    validator.validate(registry.BENCHMARK_SUMMARY, loaded, schema_version="1.0.0")


def test_cli_schema_validate_strict_and_warn(tmp_path):
    if not _has_pydantic():
        import pytest

        pytest.skip("pydantic not installed")

    from insideLLMs.cli import main

    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "input": "hi",
                "output": "ok",
                "status": "success",
                "metadata": {},
            }
        )
    )

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0.0", "input": "hi"}))

    assert main(["schema", "validate", "--name", "ProbeResult", "--input", str(good)]) == 0
    assert main(["schema", "validate", "--name", "ProbeResult", "--input", str(bad)]) == 1
    # warn mode: still exits 0
    assert (
        main(
            [
                "schema",
                "validate",
                "--name",
                "ProbeResult",
                "--input",
                str(bad),
                "--mode",
                "warn",
            ]
        )
        == 0
    )


def test_harness_explain_schema_validation():
    if not _has_pydantic():
        import pytest

        pytest.skip("pydantic not installed")

    from insideLLMs.schemas import OutputValidator, SchemaRegistry

    registry = SchemaRegistry()
    validator = OutputValidator(registry)

    payload = {
        "schema_version": "1.0.1",
        "kind": "HarnessExplain",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "run_id": "run-123",
        "config_resolution": {
            "source_chain": [{"source": "config", "value": "harness.yaml"}],
            "profile_applied": True,
        },
        "effective_config": {"model_types": ["dummy"], "probe_types": ["logic"]},
        "execution": {"validate_output": True, "validation_mode": "strict"},
        "determinism": {"strict_serialization_effective": True},
        "summary": {"record_count": 1, "success_count": 1},
    }

    validator.validate(registry.HARNESS_EXPLAIN, payload, schema_version="1.0.1")
