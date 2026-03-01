import json
from pathlib import Path

import yaml

from insideLLMs.attestations.claims.compiler import compile_claims


def test_compile_claims(tmp_path):
    summary = {"metrics": {"accuracy": {"mean": 0.95}, "latency": {"mean": 120}}}
    claims_yaml = """
claims:
  - id: claim-1
    metric: accuracy
    operator: ">="
    threshold: 0.90
  - id: claim-2
    metric: latency
    operator: "<"
    threshold: 200
"""
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    claims_path = tmp_path / "claims.yaml"
    claims_path.write_text(claims_yaml)

    res = compile_claims(claims_path, tmp_path)

    assert "verification" in res
    assert res["verification"]["claim-1"]["passed"] is True
    assert res["verification"]["claim-2"]["passed"] is True
    assert (tmp_path / "claims.json").exists()
    assert (tmp_path / "verification.json").exists()
