"""EvalBOM: CycloneDX and SPDX 3 emission from run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def emit_cyclonedx(run_dir: Path | str) -> dict[str, Any]:
    """Emit CycloneDX EvalBOM for run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"

    components = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if "model" in manifest and "model_id" in manifest["model"]:
                components.append(
                    {
                        "type": "machine-learning-model",
                        "name": manifest["model"]["model_id"],
                        "supplier": {"name": manifest["model"].get("provider", "unknown")},
                    }
                )
            if "probe" in manifest and "probe_id" in manifest["probe"]:
                components.append(
                    {
                        "type": "application",
                        "name": manifest["probe"]["probe_id"],
                    }
                )
            if "dataset" in manifest and "dataset_id" in manifest["dataset"]:
                components.append(
                    {
                        "type": "data",
                        "name": manifest["dataset"]["dataset_id"],
                    }
                )
        except Exception:
            pass

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "version": 1,
        "components": components,
    }


def emit_spdx3(run_dir: Path | str) -> dict[str, Any]:
    """Emit SPDX 3 EvalBOM for run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"

    elements = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if "model" in manifest and "model_id" in manifest["model"]:
                elements.append(
                    {
                        "type": "Software",
                        "name": manifest["model"]["model_id"],
                        "primaryPurpose": "machine-learning-model",
                    }
                )
            if "probe" in manifest and "probe_id" in manifest["probe"]:
                elements.append(
                    {
                        "type": "Software",
                        "name": manifest["probe"]["probe_id"],
                        "primaryPurpose": "application",
                    }
                )
            if "dataset" in manifest and "dataset_id" in manifest["dataset"]:
                elements.append(
                    {
                        "type": "Dataset",
                        "name": manifest["dataset"]["dataset_id"],
                    }
                )
        except Exception:
            pass

    return {
        "spdxVersion": "SPDX-3.0",
        "elements": elements,
    }
