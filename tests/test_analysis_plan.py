import json
from pathlib import Path
from insideLLMs.analysis.plan import write_plan

def test_write_plan(tmp_path):
    plan_dict = {"objective": "evaluate safety", "thresholds": {"accuracy": 0.9}}
    path, digest = write_plan(tmp_path, plan_dict)
    
    assert path.exists()
    assert path.parent.name == "analysis"
    assert path.name == "plan.json"
    assert json.loads(path.read_text()) == plan_dict
    assert isinstance(digest, str)
    assert len(digest) == 64 # sha256 length
