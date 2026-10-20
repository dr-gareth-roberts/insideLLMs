"""Claims-quality acceptance is opt-in and distinct from evidence authenticity."""

import json
from pathlib import Path
from typing import Any

import pytest

from insideLLMs.contrib.claims.compiler import compile_claims
from insideLLMs.policy import run_policy


@pytest.fixture
def claim_run(tmp_path: Path) -> Path:
    """Create structurally complete unsigned evidence with real compiler results."""
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "records.jsonl").write_text('{"output":"4"}\n', encoding="utf-8")
    attestations = tmp_path / "attestations"
    attestations.mkdir()
    for stage in (
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
        "07.claims",
    ):
        (attestations / f"{stage}.dsse.json").write_text("{}", encoding="utf-8")
    claims = [
        {"id": "accuracy", "metric": "accuracy", "operator": ">=", "threshold": 0.9},
        {"id": "latency", "metric": "latency", "operator": "<", "threshold": 200},
    ]
    (tmp_path / "claims.yaml").write_text(json.dumps({"claims": claims}), encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps({"metrics": {"accuracy": {"mean": 0.95}, "latency": {"mean": 120}}}),
        encoding="utf-8",
    )
    compile_claims(tmp_path / "claims.yaml", tmp_path)
    return tmp_path


def _read(run_dir: Path, filename: str) -> Any:
    """Read a test fixture's JSON without using the policy implementation."""
    return json.loads((run_dir / filename).read_text(encoding="utf-8"))


def _write(run_dir: Path, filename: str, value: Any) -> None:
    """Write JSON, including deliberately malformed non-finite numeric fixtures."""
    (run_dir / filename).write_text(json.dumps(value), encoding="utf-8")


def _assert_quality_failure(run_dir: Path) -> dict[str, Any]:
    """Require an explicit quality rejection while structural checks still pass."""
    verdict = run_policy(run_dir, require_claims_pass=True)
    assert verdict["passed"] is False
    assert verdict["structural_passed"] is True
    assert verdict["checks"]["claims_quality"] is False
    assert verdict["claims_quality"]["passed"] is False
    assert verdict["claims_quality"]["reasons"]
    assert verdict["reasons"]
    assert verdict["assurance"] == "structural"
    assert verdict["claims_quality"]["assurance"] == "metric_acceptance"
    assert verdict["signature_authenticity"] == "not_checked"
    assert verdict["scitt_authenticity"] == "not_checked"
    return verdict


def test_compiler_success_and_quality_acceptance_are_distinct(claim_run: Path) -> None:
    """Successful compilation can contain one failed threshold among passing claims."""
    summary = _read(claim_run, "summary.json")
    summary["metrics"]["accuracy"]["mean"] = 0.8
    _write(claim_run, "summary.json", summary)
    compiled = compile_claims(claim_run / "claims.yaml", claim_run)
    assert compiled["status"] == "success"
    assert compiled["verification"]["accuracy"]["passed"] is False
    assert compiled["verification"]["latency"]["passed"] is True
    verdict = _assert_quality_failure(claim_run)
    assert verdict["claims_quality"]["checks"]["accuracy"] is False
    assert verdict["claims_quality"]["checks"]["latency"] is True


@pytest.mark.parametrize("declarations", ["empty", "duplicate"])
def test_compiler_cannot_create_vacuous_or_overwritten_acceptance(
    claim_run: Path, declarations: str
) -> None:
    """Successful compilation must not legitimise empty or duplicate declarations."""
    claim = _read(claim_run, "claims.json")["claims"][0]
    claims = [] if declarations == "empty" else [claim, dict(claim)]
    _write(claim_run, "claims.yaml", {"claims": claims})
    compiled = compile_claims(claim_run / "claims.yaml", claim_run)
    assert compiled["status"] == "success"
    assert len(compiled["verification"]) == (0 if declarations == "empty" else 1)
    _assert_quality_failure(claim_run)


def test_passing_compiler_results_do_not_claim_authenticity(claim_run: Path) -> None:
    """Accept valid metric evidence without treating unsigned envelopes as authentic."""
    verdict = run_policy(claim_run, require_claims_pass=True)
    assert verdict["passed"] is True
    assert verdict["structural_passed"] is True
    assert verdict["claims_quality"]["passed"] is True
    assert verdict["claims_quality"]["checks"] == {"accuracy": True, "latency": True}
    assert verdict["checks"]["claims_quality"] is True
    assert verdict["claims_quality"]["assurance"] == "metric_acceptance"
    assert verdict["claims_quality"]["reasons"] == []
    assert verdict["reasons"] == []
    assert verdict["assurance"] == "structural"
    assert verdict["signature_authenticity"] == "not_checked"
    assert verdict["scitt_authenticity"] == "not_checked"


def test_passing_claims_do_not_override_structural_failure(claim_run: Path) -> None:
    """Metric acceptance and structural completeness must both hold."""
    (claim_run / "records.jsonl").unlink()
    verdict = run_policy(claim_run, require_claims_pass=True)
    assert verdict["passed"] is False
    assert verdict["structural_passed"] is False
    assert verdict["claims_quality"]["passed"] is True
    assert verdict["checks"]["records"] is False


@pytest.mark.parametrize("filename", ["claims.json", "verification.json"])
def test_missing_claim_evidence_fails_closed(claim_run: Path, filename: str) -> None:
    """Neither an attestation filename nor the other evidence file can stand in."""
    (claim_run / filename).unlink()
    verdict = _assert_quality_failure(claim_run)
    assert any(filename in reason for reason in verdict["claims_quality"]["reasons"])


@pytest.mark.parametrize("filename", ["claims.json", "verification.json"])
@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]", b"null", b"true", b"{}"])
def test_unreadable_or_invalid_evidence_fails_closed(
    claim_run: Path, filename: str, raw: bytes
) -> None:
    """Parsing and shape errors must return a rejection, not escape or pass."""
    (claim_run / filename).write_bytes(raw)
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("filename", ["claims.json", "verification.json"])
def test_evidence_directory_fails_closed(claim_run: Path, filename: str) -> None:
    """An existing path is insufficient if its contents cannot be read as evidence."""
    (claim_run / filename).unlink()
    (claim_run / filename).mkdir()
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("duplicate", ["claim_id", "passed", "claims_list"])
def test_duplicate_json_keys_fail_closed(claim_run: Path, duplicate: str) -> None:
    """Reject ambiguity even when a last-key-wins parser would see passing evidence."""
    filename = "verification.json"
    if duplicate == "claim_id":
        verification = _read(claim_run, filename)
        raw = '{"accuracy":{"passed":false},' + json.dumps(verification)[1:]
    elif duplicate == "passed":
        raw = (
            (claim_run / filename)
            .read_text(encoding="utf-8")
            .replace('"passed": true', '"passed": false, "passed": true', 1)
        )
    else:
        filename = "claims.json"
        raw = '{"claims":[],' + json.dumps(_read(claim_run, filename))[1:]
    (claim_run / filename).write_text(raw, encoding="utf-8")
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("coverage", ["missing", "extra"])
def test_verification_must_cover_exactly_the_declared_claims(
    claim_run: Path, coverage: str
) -> None:
    """Partial evidence and evidence for undeclared claims are rejected."""
    verification = _read(claim_run, "verification.json")
    if coverage == "missing":
        del verification["latency"]
    else:
        verification["unlisted"] = dict(verification["accuracy"])
    _write(claim_run, "verification.json", verification)
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("result", [None, [], True, "passed"])
def test_each_verification_result_must_be_an_object(claim_run: Path, result: Any) -> None:
    """A truthy scalar is not a claim result."""
    verification = _read(claim_run, "verification.json")
    verification["accuracy"] = result
    _write(claim_run, "verification.json", verification)
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("field", ["passed", "metric_value", "operator", "threshold"])
def test_required_result_fields_cannot_be_omitted(claim_run: Path, field: str) -> None:
    """Incomplete result objects fail even when the remaining values look valid."""
    verification = _read(claim_run, "verification.json")
    del verification["accuracy"][field]
    _write(claim_run, "verification.json", verification)
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed", False),
        ("passed", "true"),
        ("passed", 1),
        ("passed", None),
        ("metric_value", "0.95"),
        ("metric_value", True),
        ("metric_value", None),
        ("metric_value", float("nan")),
        ("metric_value", float("inf")),
        ("metric_value", float("-inf")),
        ("metric_value", []),
        ("threshold", "0.9"),
        ("threshold", True),
        ("threshold", float("nan")),
        ("threshold", float("inf")),
        ("operator", "!="),
        ("operator", []),
        ("error", "metric calculation failed"),
        ("error", False),
        ("error", {}),
        ("threshold", 0.01),
        ("operator", ">"),
        ("metric_value", 0.01),
    ],
)
def test_malformed_or_inconsistent_result_fails_closed(
    claim_run: Path, field: str, value: Any
) -> None:
    """Strict acceptance rejects coercion, error flags and contradictory evidence."""
    verification = _read(claim_run, "verification.json")
    verification["accuracy"][field] = value
    _write(claim_run, "verification.json", verification)
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("error", [None, ""])
def test_empty_optional_error_does_not_reject_valid_results(claim_run: Path, error: Any) -> None:
    """A null or empty optional error denotes no reported calculation error."""
    verification = _read(claim_run, "verification.json")
    verification["accuracy"]["error"] = error
    _write(claim_run, "verification.json", verification)
    assert run_policy(claim_run, require_claims_pass=True)["passed"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("id", "  "),
        ("id", 1),
        ("id", "latency"),
        ("metric", ""),
        ("metric", None),
        ("operator", "!="),
        ("operator", []),
        ("threshold", True),
        ("threshold", "0.9"),
        ("threshold", float("inf")),
    ],
)
def test_invalid_declarations_fail_closed(claim_run: Path, field: str, value: Any) -> None:
    """Malformed or duplicate declarations cannot be hidden by valid result IDs."""
    claims = _read(claim_run, "claims.json")
    claims["claims"][0][field] = value
    _write(claim_run, "claims.json", claims)
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize("claims", [[], {}, None, [None]])
def test_claims_must_be_a_nonempty_list_of_objects(claim_run: Path, claims: Any) -> None:
    """No claims is insufficient evidence for an explicitly requested gate."""
    _write(claim_run, "claims.json", {"claims": claims})
    _assert_quality_failure(claim_run)


@pytest.mark.parametrize(
    ("operator", "threshold"), [(">", 0.94), (">=", 0.95), ("<", 0.96), ("<=", 0.95), ("==", 0.95)]
)
def test_supported_comparisons_accept_real_compiler_output(
    claim_run: Path, operator: str, threshold: float
) -> None:
    """The consumer accepts every comparison currently emitted by the compiler."""
    claims = _read(claim_run, "claims.json")
    claims["claims"][0].update(operator=operator, threshold=threshold)
    _write(claim_run, "claims.yaml", claims)
    compile_claims(claim_run / "claims.yaml", claim_run)
    assert run_policy(claim_run, require_claims_pass=True)["passed"] is True


@pytest.mark.parametrize("evidence", ["missing", "failed", "malformed", "empty"])
def test_disabled_gate_preserves_the_original_structural_verdict(
    claim_run: Path, evidence: str
) -> None:
    """Default and explicit False retain the exact legacy verdict for all evidence."""
    baseline = run_policy(claim_run)
    assert "claims_quality" not in baseline
    assert "structural_passed" not in baseline
    if evidence == "missing":
        (claim_run / "claims.json").unlink()
        (claim_run / "verification.json").unlink()
    elif evidence == "failed":
        verification = _read(claim_run, "verification.json")
        verification["accuracy"]["passed"] = False
        _write(claim_run, "verification.json", verification)
    elif evidence == "malformed":
        (claim_run / "claims.json").write_bytes(b"\xff")
        (claim_run / "verification.json").write_bytes(b"{")
    else:
        _write(claim_run, "claims.json", {"claims": []})
        _write(claim_run, "verification.json", {})
    assert run_policy(claim_run) == baseline
    assert run_policy(claim_run, require_claims_pass=False) == baseline


@pytest.mark.parametrize("value", [None, "false", "true", 0, 1, []])
def test_gate_configuration_requires_an_explicit_boolean(claim_run: Path, value: Any) -> None:
    """Ambiguous Python values cannot silently enable or disable the gate."""
    with pytest.raises((TypeError, ValueError)):
        run_policy(claim_run, require_claims_pass=value)
