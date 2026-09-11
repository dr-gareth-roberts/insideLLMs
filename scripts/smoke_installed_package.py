"""Exercise an installed distribution outside the source tree, without API keys.

Run with the clean environment's interpreter after installing a wheel or sdist:
    /tmp/package-venv/bin/python scripts/smoke_installed_package.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    source_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["INSIDELLMS_DISABLE_PLUGINS"] = "1"
    with tempfile.TemporaryDirectory(prefix="insidellms-package-smoke-") as scratch:
        cwd = Path(scratch)
        env["MPLCONFIGDIR"] = str(cwd / "matplotlib")
        env["XDG_CACHE_HOME"] = str(cwd / "cache")
        imported = subprocess.check_output(
            [sys.executable, "-I", "-c", "import insideLLMs; print(insideLLMs.__file__)"],
            cwd=cwd,
            env=env,
            text=True,
        ).strip()
        if Path(imported).resolve().is_relative_to(source_root / "insideLLMs"):
            raise RuntimeError(f"Smoke test imported the source tree: {imported}")

        def cli(*args: str, expected_exit: int = 0) -> None:
            result = subprocess.run(
                [sys.executable, "-I", "-m", "insideLLMs.cli", *args],
                cwd=cwd,
                env=env,
                check=False,
            )
            if result.returncode != expected_exit:
                raise RuntimeError(
                    f"CLI {args} exited {result.returncode}, expected {expected_exit}"
                )

        cli("--version")
        cli("info", "dataset", "reasoning")
        for template in ("basic", "benchmark", "tracking", "full", "harness"):
            config = cwd / template / "config.yaml"
            cli("init", str(config), "--template", template, "--quiet")
            cli("validate", str(config))
            run_dir = cwd / template / "run"
            if template == "harness":
                cli("harness", str(config), "--run-dir", str(run_dir), "--skip-report")
            else:
                cli("run", str(config), "--run-dir", str(run_dir))
            cli("validate", str(run_dir))
            cli("diff", str(run_dir), str(run_dir), "--fail-on-changes")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            if not manifest.get("run_completed"):
                raise RuntimeError(f"Incomplete packaged run: {run_dir}")
        for name, response, accuracy in (("baseline", "Paris", 1.0), ("candidate", "Lyon", 0.0)):
            config = cwd / f"{name}.json"
            config.write_text(
                json.dumps(
                    {
                        "config_version": "1",
                        "model": {"type": "dummy", "args": {"canned_response": response}},
                        "probe": {"type": "logic"},
                        "dataset": {
                            "format": "inline",
                            "data": [
                                {
                                    "question": "What is the capital of France?",
                                    "reference_answer": "Paris",
                                }
                            ],
                        },
                    }
                )
            )
            cli("run", str(config), "--run-dir", str(cwd / name))
            record = json.loads((cwd / name / "records.jsonl").read_text())
            if record["scores"] != {"accuracy": accuracy}:
                raise RuntimeError(f"Incorrect packaged score: {record['scores']}")
        cli(
            "diff",
            str(cwd / "baseline"),
            str(cwd / "candidate"),
            "--fail-on-regressions",
            expected_exit=2,
        )
        print("Installed package smoke passed for all five templates and a real score regression.")


if __name__ == "__main__":
    main()
