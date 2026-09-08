from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def test_legacy_script_fails_closed_without_invoking_tools(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    invocation_log = tmp_path / "invocations.log"

    bash_path = shutil.which("bash")
    assert bash_path is not None
    (stub_dir / "bash").symlink_to(bash_path)

    for command in ("python", "python3", "git", "gh", "pip"):
        stub = stub_dir / command
        stub.write_text(
            f"#!/usr/bin/env bash\nprintf '%s\\n' '{command}' >> \"$INVOCATION_LOG\"\nexit 97\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    env = {
        "HOME": str(tmp_path),
        "INVOCATION_LOG": str(invocation_log),
        "PATH": str(stub_dir),
    }

    result = subprocess.run(
        ["/bin/bash", str(repo_root / ".github/actions/diff-gate/diff-gate.sh")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "legacy action is retired" in result.stderr
    assert "repository-root action" in result.stderr
    assert "wiki/tutorials/CI-Integration.md" in result.stderr
    assert not invocation_log.exists()
