"""Pre-registered analysis plan (signed before execution) for Ultimate mode.

Scoring attestation (05) references plan digest. Policy can require plan exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_plan(run_dir: Path | str, plan: dict[str, Any]) -> Path:
    """Write analysis plan to run_dir/analysis/plan.json (stub)."""
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "plan.json"
    path.write_text(json.dumps(plan, indent=2))
    return path
