# MONSTER_LOOP

Exhaustive, resumable audit loop for `insideLLMs`: bugs, syntax/import errors, dead code, inefficiencies, structural simplification — run one verified increment at a time until the codebase clears its own bar for production quality.

Built to survive context resets. Every increment reads its state from disk, does exactly one thing, proves it, and stops.

## Trust goal

Every fix in this loop serves one outcome: someone who has never read this codebase can trust `make check` passing, `insidellms diff --fail-on-regressions` failing when it should, and the coverage number in CI — without re-deriving any of it by hand. insideLLMs' whole value proposition is that its outputs are trustworthy enough to gate someone else's CI pipeline on. A bug, a dead branch, an inefficient hot path, or a structural mess that makes the next contributor guess isn't just untidy — it's a direct hit against that specific promise. Use this as the test for any judgment call §1–§7 don't explicitly cover: does the change make the tool's outputs more trustworthy without a human re-checking them, or does it just make the code look tidier? If it's not clearly the first, it's not this loop's job — log it and move on (§1.2). §8 is the mechanical checklist for "done"; this is what that checklist is in service of.

## 0. Precedence and continuity

- `AGENTS.md` is authoritative for environment and commands. Read it first, every time. If anything here conflicts with it, `AGENTS.md` wins.
- This file is authoritative for *process*: how to find work, how much to do per turn, how to prove a fix, when to stop and ask instead of guessing.
- This is not the first audit pass on this repo. `docs/AUDIT_FINDINGS.md`, `docs/AUDIT_FIX_PLAN.md`, and `docs/plans/2026-02-26-audit-remaining-cleanup*.md` are prior waves. `tests/test_audit_wave{2,3,6}_regressions.py` are their landed proof. Before starting, run `ls tests/test_audit_wave*.py` and use the next free integer as this wave's number. Don't renumber or touch the old ones.
- Treat every "✅ Done" claim in those prior docs as a *lead to re-verify*, not ground truth. They go stale. See the worked example in §9.

## 1. Non-negotiables

1. **One verified increment per turn.** Pick one item, fix it, prove it, stop.
2. **No unrequested refactors.** Log new findings as backlog items; don't touch them now.
3. **No fix without evidence.** Before/after signals required.
4. **No comment or docstring padding.**
5. **No new dependencies without escalation** — see §7.
6. **`contrib/**` is lower-trust for deletions.** Confirm before removing.
7. **Check compatibility-shim pattern before calling something duplication.**
8. **`compliance_intelligence/` and `extensions/vscode-insidellms/` are satellite scope.**
9. **Anything touching a "Stable" row in `docs/STABILITY_MATRIX.md` is escalation**, unless purely internal with no signature or behavioural difference.

## 2. State

```
.loop/BACKLOG.json   — machine-readable findings
.loop/LOG.md         — human-readable append-only run log
```

## 3. The rail (A0–A7)

**A0 — Sync.** Read `AGENTS.md`, this file, `.loop/BACKLOG.json`, tail of `.loop/LOG.md`.

**A1 — Scan** (if backlog empty, every 15 fixed items, or on request):

```
ruff check .
ruff format --check .
mypy insideLLMs
vulture insideLLMs --min-confidence 80
pytest --collect-only -q
pytest -m "not slow and not integration" --cov=insideLLMs --cov-report=term-missing
```

**A2 — Select.** One item: severity, then oldest, then core > contrib > satellites.

**A3 — Diagnose.** Grep callers, registry, shim docstrings.

**A4 — Fix.** Smallest correct diff.

**A5 — Verify.** Match blast radius (see plan table in repo docs).

**A6 — Record.** Update backlog, append log, commit `fix(scope): summary [W<n>-<id>]`.

**A7 — Stop.**

## 4. Standing backlog

- Shrink `[tool.mypy] disable_error_code` in `pyproject.toml` (one code per increment).
- `make typecheck-strict-runtime` toward gated strict.
- Deprecation sunset versions for remaining shims.

## 5–10

See initial chat specification for scan categories, fix discipline, escalation rules, definition of done, worked example (trace shims / stale Done claims), and unattended/supervised/CI running modes.
