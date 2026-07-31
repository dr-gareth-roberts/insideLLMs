"""W7-0008 slice 11: safety/high_level/workflows measured gaps."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pytest

from insideLLMs.runtime import _high_level as hl
from insideLLMs.runtime import workflows as wf
from insideLLMs.safety import (
    BiasDetector,
    ContentSafetyAnalyzer,
    RiskLevel,
    SafetyCategory,
    SafetyFlag,
    SafetyHallucinationIndicatorDetector,
    SafetyReport,
)
from insideLLMs.types import ProbeResult, ResultStatus


def test_safety_report_and_detectors() -> None:
    report = SafetyReport(
        text="x",
        is_safe=True,
        overall_risk=RiskLevel.LOW,
        flags=[],
        scores={},
    )
    assert report.get_highest_risk_flag() is None

    flags = [
        SafetyFlag(
            category=SafetyCategory.PII_EXPOSURE,
            risk_level=RiskLevel.LOW,
            description="l",
            confidence=0.1,
        ),
        SafetyFlag(
            category=SafetyCategory.TOXICITY,
            risk_level=RiskLevel.HIGH,
            description="h",
            confidence=0.9,
        ),
    ]
    report2 = SafetyReport(
        text="x", is_safe=False, overall_risk=RiskLevel.HIGH, flags=flags, scores={}
    )
    assert report2.get_highest_risk_flag().risk_level == RiskLevel.HIGH

    hd = SafetyHallucinationIndicatorDetector()
    assert hd.get_risk_level({"risk_score": 0.1}) == RiskLevel.LOW
    assert hd.get_risk_level({"risk_score": 0.3}) == RiskLevel.MEDIUM
    assert hd.get_risk_level({"risk_score": 0.5}) == RiskLevel.HIGH
    assert hd.get_risk_level({"risk_score": 0.9}) == RiskLevel.CRITICAL

    bd = BiasDetector()
    text_stereo = "All women are naturally better at nursing than men."
    matches = bd.analyze_stereotypes(text_stereo)
    assert isinstance(matches, list)
    unbalanced = "He he he he he he he. " + text_stereo
    analysis = bd.analyze(unbalanced)
    assert "bias_score" in analysis

    sa = ContentSafetyAnalyzer()
    hall_text = (
        "Studies show that 97% of experts agree this unverified claim is definitely true "
        "according to research that proves it without any doubt whatsoever."
    )
    full = sa.analyze(
        hall_text + " " + unbalanced,
        check_toxicity=True,
        check_hallucination=True,
        check_bias=True,
    )
    assert full.overall_risk in (
        RiskLevel.NONE,
        RiskLevel.LOW,
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.CRITICAL,
    )
    low_only = SafetyReport(
        text="t",
        is_safe=True,
        overall_risk=RiskLevel.LOW,
        flags=[
            SafetyFlag(
                category=SafetyCategory.PII_EXPOSURE,
                risk_level=RiskLevel.LOW,
                description="l",
                confidence=0.2,
            )
        ],
        scores={},
    )
    assert low_only.get_highest_risk_flag().risk_level == RiskLevel.LOW


def test_high_level_coerce_and_create_experiment() -> None:
    from insideLLMs.models import DummyModel
    from insideLLMs.probes.logic import LogicProbe

    class WeirdStatus(Enum):
        X = "not-a-real-status"

    class BadEnum(Enum):
        Y = object()

    model = DummyModel()
    probe = LogicProbe()
    results = [
        {"input": "a", "output": "b", "status": WeirdStatus.X, "error": None},
        {"input": "c", "output": "d", "status": "success", "error": None},
        {"input": "e", "output": "f", "status": BadEnum.Y, "error": None},
    ]
    exp = hl.create_experiment_result(
        model=model,
        probe=probe,
        results=results,
        experiment_id=None,
    )
    assert exp.experiment_id
    assert all(isinstance(r, ProbeResult) for r in exp.results)

    pr = [ProbeResult(input="i", output="o", status=ResultStatus.SUCCESS)]
    exp2 = hl.create_experiment_result(model=model, probe=probe, results=pr)
    assert exp2.results

    exp3 = hl.create_experiment_result(model=model, probe=probe, results=[])
    assert exp3.experiment_id


@pytest.mark.asyncio
async def test_run_probe_async_wrapper() -> None:
    from insideLLMs.models import DummyModel
    from insideLLMs.probes.logic import LogicProbe

    out = await hl.run_probe_async(DummyModel(), LogicProbe(), ["hello"])
    assert out


def test_workflows_guards(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        wf._coerce_path("   ", name="x")
    with pytest.raises(ValueError, match="validation_mode"):
        wf.run_harness_to_dir(cfg, tmp_path / "r", validation_mode="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="track"):
        wf.run_harness_to_dir(cfg, tmp_path / "r", track="nope")
    with pytest.raises(FileNotFoundError):
        wf.run_harness_to_dir(tmp_path / "missing.yaml", tmp_path / "r")
    with pytest.raises(ValueError, match="output_format"):
        wf.diff_run_dirs(tmp_path, tmp_path, output_format="xml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit"):
        wf.diff_run_dirs(tmp_path, tmp_path, limit=0)
