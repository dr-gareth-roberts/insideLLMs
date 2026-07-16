# Goal: Complete One Trustworthy CI Audit Increment

## User Request

Run the supplied MONSTER_LOOP process so that a contributor who has never read
the codebase can trust `make check`, `insidellms diff --fail-on-regressions`,
and the coverage number in CI without manually re-deriving their correctness.
Every judgment call must improve the trustworthiness of the tool's outputs,
not merely make the code look tidier.

## Refined Goal

Complete exactly one Wave 7 MONSTER_LOOP A0-A7 increment against the existing
audit backlog. Select the highest-priority eligible open item, diagnose it from
actual callers and tests, make the smallest correctness-focused change, and
record literal before/after evidence. Stop after that single item; do not begin
another backlog item or perform unrelated cleanup.

## Acceptance Criteria

- [ ] Exactly one eligible `.loop/BACKLOG.json` item is selected according to
      severity, age, and scope ordering, and no unrelated finding is fixed.
- [ ] The selected item's defect or trust gap has a reproducible failing signal
      before the change and a clean signal after the change; a behavioral fix
      includes a regression test that fails before and passes after.
- [ ] The blast-radius quality gates from MONSTER_LOOP A5 pass, including
      `make check` for any core `insideLLMs/**` change and the additional
      determinism/contract/golden-path gates when the artifact or diff spine is
      touched.
- [ ] Coverage is measured comparably and does not fall below the Wave 7
      baseline of 91.0%; no unexplained suppression or silent failure is added.
- [ ] `.loop/BACKLOG.json` and `.loop/LOG.md` contain the actual before/after
      evidence, verification commands/results, and traceability for the item.
- [ ] The implementation is delivered in one Builder commit whose title
      includes both `[B]` and the selected `[W7-....]` identifier, then the
      Builder stops without starting a second increment.

## Scope Boundaries

**In scope:**
- A0-A7 for one highest-priority eligible open Wave 7 backlog item.
- The smallest necessary production code, regression test, user-visible
  changelog entry when required, and audit state updates.
- Evidence that directly increases confidence in correctness, regression
  gating, deterministic artifacts, or CI coverage reporting.

**Out of scope:**
- A second backlog item, broad scans unless A1's cadence requires one, and
  unrelated refactors or cosmetic cleanup.
- Stable-surface behavior/signature changes, uncertain dynamically dispatched
  deletions, compatibility-shim removal without a sunset decision, new
  third-party dependencies, or structural changes spanning more than roughly
  five files; these must be recorded/escalated rather than guessed.
- Driving all of Wave 7 to Definition of Done in this single Goal run.
- Treating known optional-dependency failures in a `[dev]`-only environment as
  regressions.

## Applicable Project Conventions

**Quality gate command:**
- `make check` for core package changes.
- `make check-fast` for tests, `contrib/`, or satellite-only changes.
- `make check && make golden-path && python3 -m pytest -m determinism && python3 -m pytest -m contract`
  for determinism, diff, or artifact-spine changes.
- `make check && make typecheck-strict` for `injection.py` or `safety.py`.
- Wave baseline: 91.0% branch coverage as recorded in `.loop/BACKLOG.json`.

**Commit convention:**
- Project convention: `type(scope): description`.
- Builder iteration convention: `type(scope): [B] description [W7-....]`,
  imperative and at most 72 characters.
- Inspector convention: `chore(scope): [I] description`, at most 72 characters.
- Assisted-by trailer required: `Assisted-by: Claude:Sonnet-4.6`

**Guidelines:**
- `AGENTS.md`
- `CONTRIBUTING.md`
- The user-supplied MONSTER_LOOP specification captured in this goal.
- No `CONSTITUTION.md`, `.agents/guidelines/`, or `.github/guidelines/` exists.

**Rules:**
- Use `python3`; installed scripts may require `$HOME/.local/bin` on `PATH`.
- Do not add dependencies merely to run existing checks; optional integrations
  can fail in a `[dev]`-only environment as documented in `AGENTS.md`.
- Read `.loop/BACKLOG.json`, the tail of `.loop/LOG.md`, callers, registry or
  decorator dispatch, and relevant tests before editing.
- Compatibility shims are intentional; uncertain Stable or dynamically
  dispatched behavior is escalated, not changed.
- Comments explain why only; prefer deletion to padding.
- Audit history is append-only, and every fix needs literal before/after proof.
