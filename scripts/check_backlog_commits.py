#!/usr/bin/env python3
"""Verify that every fix_commit recorded in .loop/BACKLOG.json resolves.

An audit backlog whose commit hashes do not exist cannot be audited. Squash
merges routinely destroy the per-fix commits those hashes point at, which is
how ten of the thirteen hashes in this repo became unresolvable. A goal
inspection has already failed once for exactly this reason.

Exit status is 1 if any recorded hash is missing, so this can gate CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKLOG = Path(__file__).resolve().parents[1] / ".loop" / "BACKLOG.json"


def resolves(sha: str) -> bool:
    """Return True if sha names an object in this repository."""
    result = subprocess.run(
        ["git", "cat-file", "-t", sha],
        capture_output=True,
        text=True,
        cwd=BACKLOG.parents[1],
    )
    return result.returncode == 0 and result.stdout.strip() == "commit"


def main() -> int:
    if not BACKLOG.exists():
        print(f"no backlog at {BACKLOG}; nothing to check")
        return 0

    items = json.loads(BACKLOG.read_text(encoding="utf-8"))["items"]
    recorded = [(i["id"], i["fix_commit"]) for i in items if i.get("fix_commit")]

    missing = [(item_id, sha) for item_id, sha in recorded if not resolves(sha)]

    for item_id, sha in missing:
        print(f"MISSING {item_id}: {sha}")

    print(f"\n{len(recorded) - len(missing)}/{len(recorded)} recorded fix_commit hashes resolve")

    if missing:
        print(
            "\nA recorded hash that does not resolve makes that item's verification "
            "claim unauditable. If the commit was squash-merged, record the merge "
            "commit instead."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
