# Goal: Recover Trustworthy CI Coverage to 91 Percent

## User Request

Target genuine coverage gaps such as `caching.py`,
`cli/commands/doctor.py`, and `analysis/visualization.py`. Once coverage
reaches at least 91%, rerun the original acceptance gates in a new goal.

## Refined Goal

Create one dedicated Wave 7 `coverage_gap` backlog item using the next free
append-only ID, then add meaningful tests for real behavior in the three named
modules until exact project branch coverage is at least 91.0%. Do not change
production behavior merely to influence coverage. After reaching the threshold,
rerun the repository's complete trust gates, including behavioral diff gating,
and record reproducible evidence so a new contributor does not need to
recalculate the result by hand.

## Acceptance Criteria

- [ ] A new `coverage_gap` item is appended to `.loop/BACKLOG.json` using the
      actual next free Wave 7 ID; it records the 90.0781% starting measurement,
      the target, scope, evidence, final status, commit traceability, and exact
      verification commands/results.
- [ ] Tests added for `insideLLMs/caching.py`,
      `insideLLMs/cli/commands/doctor.py`, and/or
      `insideLLMs/analysis/visualization.py` exercise meaningful public,
      boundary, error, persistence, or diagnostic behavior. No test exists only
      to execute lines, and no assertion is weakened.
- [ ] The comparable command
      `python3 -m pytest -m "not slow and not integration" --cov=insideLLMs --cov-report=term-missing`
      reports exact combined line/branch coverage of at least 91.0%, with no
      omitted files, changed coverage configuration, new suppressions, or
      unexplained failures.
- [ ] `make check` passes cleanly in the restored optional-dependency
      environment, and `make golden-path`, `python3 -m pytest -m determinism`,
      and `python3 -m pytest -m contract` all pass.
- [ ] Existing or newly focused CLI tests prove
      `insidellms diff --fail-on-regressions` exits nonzero for a genuine
      regression and succeeds when no regression exists.
- [ ] `.loop/LOG.md` receives one append-only entry containing literal
      before/after coverage numerators, percentages, test counts, trust-gate
      results, and the selected Wave 7 ID.
- [ ] The work is delivered in one Builder commit titled
      `test(coverage): [B] recover CI coverage [W7-....]`, at most 72
      characters, followed by independent Inspector verification.

## Scope Boundaries

**In scope:**
- Tests and test fixtures for the three named modules, prioritizing the
  highest-value uncovered behavior.
- Minimal test-environment restoration using already-declared optional
  dependencies when a gate proves one is missing.
- One new Wave 7 `coverage_gap` backlog item and its append-only audit log.
- Existing diff CLI tests or a focused new test proving both exit-code paths.

**Out of scope:**
- Production-code changes, coverage configuration changes, exclusions,
  pragma-based suppression, assertion weakening, generated padding, or tests
  whose only purpose is line execution.
- Fixing bugs discovered while adding coverage; record them as separate
  backlog findings and return BLOCKED if they prevent honest completion.
- Any other existing Wave 7 backlog item, compatibility-shim removal, Stable
  surface changes, new project dependencies, or broad refactors.

## Applicable Project Conventions

**Quality gate command:**
- `make check`
- `make golden-path`
- `python3 -m pytest -m determinism`
- `python3 -m pytest -m contract`
- `python3 -m pytest -m "not slow and not integration" --cov=insideLLMs --cov-report=term-missing`
- Focused diff CLI tests proving `--fail-on-regressions` exit behavior.

**Commit convention:**
- Project convention: `type(scope): description`.
- Builder: `test(coverage): [B] recover CI coverage [W7-....]`, at most 72
  characters.
- Inspector: `chore(goal): [I] inspect coverage recovery`, at most 72
  characters.
- Assisted-by trailer required: `Assisted-by: Claude:Sonnet-4.6`

**Guidelines:**
- `AGENTS.md`
- `CONTRIBUTING.md`
- `.loop/BACKLOG.json` and `.loop/LOG.md`
- The MONSTER_LOOP rules captured by the prior goal and audit state.
- No `CONSTITUTION.md`, `.agents/guidelines/`, or `.github/guidelines/` exists.

**Rules:**
- Use `python3`; ensure `$HOME/.local/bin` is available when needed.
- Preserve exact coverage arithmetic. The starting evidence is
  23,178 covered items out of 25,731, or 90.0781%; 91.0% requires at least
  23,416 covered items, a gain of 238.
- Read existing tests before adding cases and reuse established fixtures and
  mocking patterns.
- Keep tests deterministic, offline, and independent of provider API keys.
- Do not modify the prior immutable goal or reinterpret its baseline.
