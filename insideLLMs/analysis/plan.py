"""Pre-registered analysis plan (signed before execution) for Ultimate mode.

Scoring attestation (05) references plan digest. Policy can require plan exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_plan(run_dir: Path | str, plan: dict[str, Any]) -> tuple[Path, str]:
    """Write analysis plan to run_dir/analysis/plan.json and return (path, digest)."""
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "plan.json"
    
    from insideLLMs.crypto.canonical import digest_obj
    plan_dict = json.loads(json.dumps(plan)) # normalize
    path.write_text(json.dumps(plan_dict, indent=2))
    
    digest_info = digest_obj(plan_dict, purpose="analysis_plan")
    
    return path, digest_info["digest"]
