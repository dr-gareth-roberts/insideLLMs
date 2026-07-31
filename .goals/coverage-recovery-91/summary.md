# Coverage Recovery Goal Summary

## Outcome

Wave 7 coverage-gap item W7-0071 is verified. Exact combined line and branch
coverage increased from 23,178/25,731 (90.0781%) to 23,614/25,731 (91.7726%)
without production-code or coverage-configuration changes.

## Acceptance Criteria

- W7-0071 was appended as the next free Wave 7 ID with a traceable full
  implementation commit SHA.
- 134 deterministic behavioral tests cover caching, doctor diagnostics, and
  diff regression exit semantics.
- Coverage.py independently reports 91.7726%, above the 91.0% requirement.
- `make check`, `make golden-path`, determinism tests, and contract tests pass.
- Four focused tests prove `insidellms diff --fail-on-regressions` fails for a
  genuine regression and succeeds for unchanged or improved results.
- `.loop/LOG.md` preserves literal before/after evidence and an append-only
  correction explaining coverage arithmetic.
- Builder and Inspector commits satisfy the Goal role markers, length limits,
  and model trailers.

## Iteration History

1. Iteration 1 recovered coverage and passed all behavioral gates. Inspector
   rejected the result because W7-0071 referenced a nonexistent abbreviated
   commit hash.
2. Iteration 2 corrected the full commit SHA and reproduced coverage twice.
   Inspector passed all criteria after correcting its own calculation to use
   coverage.py's missing branch arcs rather than partial source-line count.

## Inspector Issues and Resolution

- **Broken traceability:** `fix_commit` was corrected to
  `b48cbe89407eea6adcea4df294ec13d08ab99221`.
- **Coverage discrepancy:** two independent runs and coverage.py JSON confirmed
  23,614/25,731 (91.7726%). The append-only log explains why subtracting
  `num_partial_branches` gives an invalid result.

## Recommendations

- Use coverage.py's JSON `percent_covered` and `missing_branches` fields for
  future exact coverage records.
- Keep the new diff exit-code tests in the determinism gate because they
  directly protect insideLLMs' CI-gating promise.
- Treat future coverage recovery as dedicated backlog work rather than
  expanding unrelated correctness fixes.
