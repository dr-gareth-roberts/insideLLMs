"""EvalBOM: CycloneDX and SPDX 3 emission from run (stub)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def emit_cyclonedx(run_dir: Path | str) -> dict[str, Any]:
    """Emit CycloneDX EvalBOM for run (stub)."""
    return {"bomFormat": "CycloneDX", "specVersion": "1.4", "version": 1, "components": []}


def emit_spdx3(run_dir: Path | str) -> dict[str, Any]:
    """Emit SPDX 3 EvalBOM for run (stub)."""
    return {"spdxVersion": "SPDX-3.0", "elements": []}
