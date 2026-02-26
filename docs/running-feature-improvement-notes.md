# Running Feature & Improvement Notes

This file is a live log of noteworthy features and improvement opportunities observed while examining the codebase.

## 2026-02-26

### Implemented (Batch A)
- Entry 11: Added shared CLI run/harness utilities in `insideLLMs/cli/commands/_run_common.py` to centralize tracker initialization and standard artifact enumeration, reducing duplication between command modules.
- Entry 13: Updated `insideLLMs/cli/commands/harness.py` to reuse runtime artifact helpers (`_atomic_write_text`, `_atomic_write_yaml`, `_ensure_run_sentinel`, `_prepare_run_dir`, `_semver_tuple`) instead of local redefinitions.
- Entry 48: Enhanced `insidellms report` output so regenerated `summary.json` now includes `report_metadata` fields (`from_records`, `tool`, `tool_version`, `records_file`) for auditability.

### Implemented (Batch B)
- Entry 41: Extended `insidellms doctor` diagnostics with verifiable-evaluation readiness checks for `ultimate:tuf`, `ultimate:cosign`, and `ultimate:oras`.
- Entry 26: Added concise command documentation for `insidellms attest`, `insidellms sign`, and `insidellms verify-signatures` (including prerequisites) in `README.md` and `API_REFERENCE.md`.

### Implemented (Batch C)
- Entry 39: Preserved timeout diagnostics in runtime outputs by mapping timeout wrapper failures to `status="timeout"` and persisting timeout metadata into emitted records/manifests (`status_counts` and `timeout_count` in manifest `custom`).
- Entry 78: Added direct CLI command tests for `attest`, `sign`, and `verify-signatures` command flows and error paths.

### Implemented (Batch D)
- Entry 1: Added a dedicated artifact contract page at `docs/ARTIFACT_CONTRACT.md` as a single source of truth for canonical artifact fields and volatility rules.
- Entry 82: Added an "Advanced Assurance" navigation section in `wiki/advanced/index.md` linking determinism, tracing, and assurance CLI workflows.
- Entry 95: Added explicit canonical-source cross-references from wiki determinism pages back to `docs/DETERMINISM.md` and `docs/ARTIFACT_CONTRACT.md`.

### Implemented (Batch E)
- Entry 54: Expanded `DOCUMENTATION_INDEX.md` with dedicated determinism/artifact-contract/verifiable-evaluation entries and quick navigation for assurance workflows.
- Entry 3: Updated architecture/instruction docs to point to the CLI package structure (`insideLLMs/cli/__init__.py`, `insideLLMs/cli/_parsing.py`, `insideLLMs/cli/commands/*`) instead of legacy single-file references.
- Entry 62: Updated `CONTRIBUTING.md` architecture guidance to reflect current CLI/runtime package layout and added stable-vs-internal API usage notes.
- Entry 77: Extended stability contract tests to include verifiable-evaluation CLI command surfaces (`attest`, `sign`, `verify-signatures`).
- Entry 81: Added a dedicated wiki page (`wiki/advanced/Verifiable-Evaluation.md`) with practical attest/sign/verify quickstart guidance and cross-links.
- Entry 14: Included an Ultimate-mode quickstart sequence and expected artifact outputs in `wiki/advanced/Verifiable-Evaluation.md`.
- Entry 93: Added direct links from core wiki pages to verifiable-evaluation workflows (`wiki/index.md`, `wiki/Determinism-and-CI.md`).
- Entry 94: Refreshed legacy CLI path references in wiki performance guidance to point at current parser/module locations.

### Implemented (Batch F)
- Entry 53: Hardened report reconstruction for mixed/partial record metadata in `insideLLMs/cli/_report_builder.py` (safe model-spec extraction, robust `example_index` sorting fallback, and strict dict guards for `metadata` payloads).
- Entry 53: Added focused regression coverage in `tests/test_cli_remaining_coverage.py` to ensure `insidellms report` succeeds when records include partial harness metadata and mixed metadata shapes.

### Implemented (Batch G)
- Entry 92: Added top-level schema validation workflow examples (JSON + JSONL) in `README.md` and aligned schema command reference examples in `wiki/reference/CLI.md` with current parser semantics.
- Entry 92: Added fast navigation to schema-validation workflows in `DOCUMENTATION_INDEX.md`.

### Implemented (Batch H)
- Entry 40: Added "Which Config API to Use" section in `wiki/reference/Configuration.md` clarifying `config.py` (YAML) vs `config_types.py` (RunConfig) usage.
- Entry 47: Created `docs/IMPORT_PATHS.md` as a migration matrix for canonical vs deprecated vs shim import paths, and linked it from `DOCUMENTATION_INDEX.md`.

### Implemented (Batch I)
- Entry 55: Cross-linked `compliance_intelligence/` from `README.md` and `DOCUMENTATION_INDEX.md` with a short status note.
- Entry 57: Switched `report.py` and `harness.py` to canonical imports (`insideLLMs.analysis.visualization`) instead of the shim (`insideLLMs.visualization`).
- Entry 63: Added a compatibility matrix to `examples/README.md` indicating which examples are fully offline vs API-key dependent.

### Implemented (Batch J)
- Entry 59: Aligned AGENTS.md coverage gate with CI (`--cov-fail-under=95`).
- Entry 65: Added `make check-fast` for quick iterative pre-checks and documented it in CONTRIBUTING.md and scripts/checks.sh.
- Entry 67: Updated SECURITY.md supported-version matrix to include 0.2.x.
- Entry 69: Added troubleshooting section to docs/PLUGINS.md (discovery disabled, import errors, registration collisions).
- Entry 87: Added "Preferred Import Style" section to docs/IMPORT_PATHS.md.

### Implemented (Batch K)
- Entry 12: Added deprecation timeline for legacy `results.jsonl` alias in docs/ARTIFACT_CONTRACT.md (Legacy Artifact Aliases section).
- Entry 85: Added shim inventory with removal/indefinite-support status to docs/IMPORT_PATHS.md.
- Entry 91: Added ci/README.md documenting how to extend the CI harness with low-cost probes while preserving offline determinism; linked from DOCUMENTATION_INDEX.md.
- Entry 96: Added "Data Retention and Privacy" section to wiki/Experiment-Tracking.md for compliance-sensitive adopters.
- Entry 98: Added docs/plans/README.md labeling planning documents as archival and pointing to current docs.

### Implemented (Batch L)
- Entry 27: Added docs/CRYPTO_DIGEST_CONTRACT.md with digest descriptor fields (`purpose`, `algo`, `canon_version`) and Merkle root descriptors for external verifiers; linked from Verifiable-Evaluation and DOCUMENTATION_INDEX.
- Entry 33: Added `insideLLMs.pipeline` → `insideLLMs.runtime.pipeline` to docs/IMPORT_PATHS.md compatibility shims.
- Entry 90: Clarified in benchmarks/README.md that thresholds are environment-dependent and should be used for relative comparison, not absolute pass/fail.

### Implemented (Batch M)
- Entry 68: Refreshed bug_report.yml version placeholder from `0.1.0` to `0.2.x`.
- Entry 71: Added DOCUMENTATION_INDEX and Docs Site links to issue template config for self-service routing before filing.
- Entry 83: Added "Archival" section to DOCUMENTATION_INDEX separating docs/plans from primary user/contributor docs navigation.
- Entry 86: Added "Import path and reference hygiene" section to docs/DOCS_STANDARDS.md with canonical-path guidance and periodic lint recommendation.
- Entry 97: Included recurring check for stale path references in wiki/reference pages within DOCS_STANDARDS.md.

### Entry 1
- Feature noticed: The project has a strong deterministic artifact spine (`records.jsonl` / `summary.json` / `report.html` / `diff.json`) with CI-oriented behavior-regression workflows.
- Suggestion: Keep this as a first-class "contract" page in docs with one source of truth for fields/volatility rules so contributors can quickly verify determinism assumptions.

### Entry 2
- Feature noticed: Runtime logic is modularized cleanly (`insideLLMs/runtime/_*` modules), with `runtime/runner.py` serving as a compatibility re-export surface.
- Suggestion: Add a brief contributor note describing which APIs are stable vs internal-only, since the re-export layer exposes many private helpers and may increase accidental coupling in downstream code/tests.

### Entry 3
- Feature noticed: CLI is package-based (`insideLLMs/cli/` with command modules), which scales well for many subcommands.
- Suggestion: Align architecture docs/instructions to point at `insideLLMs/cli/__init__.py` and `insideLLMs/cli/commands/*` (instead of legacy single-file `insideLLMs/cli.py`) to reduce onboarding friction.

### Entry 4
- Feature noticed: CLI argument construction is comprehensive and supports many deterministic/runtime flags.
- Suggestion: Split `insideLLMs/cli/_parsing.py` (currently very large) into per-command parser builders to reduce merge conflicts and make new command contributions safer.

### Entry 5
- Feature noticed: `ProbeRunner.run(...)` offers rich configurability (resume, deterministic artifacts, schema validation, batch modes, etc.).
- Suggestion: Introduce a preferred "config-first" path in docs and gradually de-emphasize the many keyword overrides to lower API complexity and reduce parameter drift risk.

### Entry 6
- Feature noticed: `insideLLMs/__init__.py` provides a broad convenience export surface for users.
- Suggestion: Consider lazy imports or a slimmer default export surface to reduce import-time overhead and optional-dependency coupling for users who only need a subset of modules.

### Entry 7
- Feature noticed: The project contains extensive inline API documentation and examples directly in module docstrings.
- Suggestion: Evaluate moving long-form examples into docs pages and keeping module docstrings shorter, which can improve source readability and maintenance when behavior changes.

### Entry 8
- Feature noticed: Test coverage breadth is strong, with many targeted branch-coverage files and deterministic-path tests.
- Suggestion: Organize `tests/` into subdirectories by subsystem (CLI/runtime/models/probes/etc.) to improve discoverability and reduce cognitive load when navigating ~190 test files.

### Entry 9
- Feature noticed: Diff functionality appears primarily implemented through CLI command code and record utilities.
- Suggestion: Clarify or clean up the `insideLLMs/diff/` package (currently appearing as cache-only in workspace view) so source-of-truth locations are explicit for contributors.

### Entry 10
- Feature noticed: Tooling standardization is strong around Ruff + Mypy + Pytest via Make targets.
- Suggestion: Reassess whether all dev dependencies are still required (for example, overlapping formatter/import-sort tools) to keep contributor setup lean and reduce dependency churn.

### Entry 11
- Feature noticed: `run` and `harness` commands both include robust run-directory resolution, tracking setup, and post-run artifact UX hints.
- Suggestion: Consolidate shared command behaviors into a small CLI utility layer to reduce duplication and keep run/harness behavior aligned over time.

### Entry 12
- Feature noticed: Harness artifact emission explicitly preserves backward compatibility (`records.jsonl` plus legacy `results.jsonl`).
- Suggestion: Add/maintain a clear deprecation timeline for legacy artifact aliases so users know what will remain stable long-term.

### Entry 13
- Feature noticed: The runtime artifact utilities already provide reusable helpers (`_atomic_write_text`, `_atomic_write_yaml`, `_prepare_run_dir`, etc.).
- Suggestion: Prefer importing and reusing these helpers directly in command modules (instead of redefining local copies) to minimize divergence in safety behavior.

### Entry 14
- Feature noticed: Ultimate mode provides an end-to-end verifiable pipeline (integrity roots, DSSE attestations, policy verdict, optional SCITT/OCI).
- Suggestion: Add a focused docs quickstart for Ultimate mode with an example command sequence and expected run-directory tree to improve adoption.

### Entry 15
- Feature noticed: Attestation generation is structured into explicit step builders (`00`-`09`) with consistent naming.
- Suggestion: Add schema-level validation for each predicate payload so optional-field growth does not silently drift across steps.

### Entry 16
- Feature noticed: Policy evaluation already checks artifact presence and required attestation files.
- Suggestion: Evolve from hardcoded checks toward configurable policy profiles/rules so teams can tune admissibility requirements without code edits.

### Entry 17
- Feature noticed: SCITT receipt handling is integrated into both attestation and policy paths.
- Suggestion: Add a strict policy mode that requires SCITT receipts for designated attestations (not only validating them when present).

### Entry 18
- Feature noticed: SCITT client has retry/timeout behavior and clear error surfacing.
- Suggestion: Expand receipt verification beyond shape/digest checks to cryptographic/issuer validation (when service metadata is available) for stronger trust guarantees.

### Entry 19
- Feature noticed: OCI publish/pull is available through ORAS wrappers and integrates with attestation step `09.publish`.
- Suggestion: Capture and persist registry-returned OCI digest wherever possible so provenance can reference immutable content addresses instead of mutable refs.

### Entry 20
- Feature noticed: Sigstore signing and verification commands are straightforward and operationally useful.
- Suggestion: Consider richer verifier constraints (for example issuer constraints alongside identity) to make policy-grade signature verification easier.

### Entry 21
- Feature noticed: OpenVEX generation exists and automatically includes run components from `manifest.json`.
- Suggestion: Add a richer vulnerability mapping path (specific CVE IDs/statuses/justifications) and gate catch-all statements behind an explicit fallback mode.

### Entry 22
- Feature noticed: Claims compilation provides a practical threshold-check workflow from `claims.yaml` to `verification.json`.
- Suggestion: Support richer metric selectors (percentiles, counts, model/probe slices) and add explicit schema validation for claims files to improve robustness.

### Entry 23
- Feature noticed: Privacy layer includes redaction, encryption, and selective-disclosure utilities.
- Suggestion: Align package exports/documentation so all three capabilities are equally discoverable from the privacy module surface.

### Entry 24
- Feature noticed: Export command supports PII redaction and optional JSONL encryption hooks.
- Suggestion: For CSV exports, consider column union + stable column ordering across all rows instead of using only first-row keys.

### Entry 25
- Feature noticed: Ultimate-mode orchestration hooks are present in both sync and async runners.
- Suggestion: Centralize shared post-artifact invocation logic into a common helper to reduce divergence risk between runner implementations.

### Entry 26
- Feature noticed: CLI includes `attest`, `sign`, and `verify-signatures` commands in the parser and command map.
- Suggestion: Add concise README/API reference sections for these commands (with prerequisites like `cosign`) to improve discoverability for new users.

### Entry 27
- Feature noticed: Crypto primitives are well-separated (`canonical`, `merkle`, bundle ID) and include purpose/canonicalization metadata.
- Suggestion: Add a compact "digest descriptor contract" doc table so external verifiers can consistently interpret `purpose`, `algo`, and `canon_version` fields.

### Entry 28
- Feature noticed: Dataset commitment and TUF client modules exist as building blocks for provenance-aware dataset ingestion.
- Suggestion: Wire these modules into config-driven dataset loading (or clearly mark as standalone utilities) so users understand the supported end-to-end path.

### Entry 29
- Feature noticed: TUF fetching currently falls back to a mock verification path when TUF runtime support is unavailable.
- Suggestion: Offer an explicit strict mode that forbids mock fallback in security-sensitive environments.

### Entry 30
- Feature noticed: Dataset loaders are simple and reliable for CSV/JSONL/HF sources.
- Suggestion: Add optional streaming/iterator loading modes for very large local datasets to reduce memory pressure during long harness runs.

### Entry 31
- Feature noticed: Trace contracts provide deterministic, CI-friendly validation across stream/tool/generate flows.
- Suggestion: Emit a standardized machine-readable trace validation artifact in run directories to make contract gating easier in CI pipelines.

### Entry 32
- Feature noticed: Trace recording is sequence-based and deterministic, which is excellent for reproducibility.
- Suggestion: Add explicit guardrails for trace size growth (event caps/sampling/truncation policy) so long runs remain manageable.

### Entry 33
- Feature noticed: `insideLLMs/pipeline.py` acts as a compatibility shim over `insideLLMs/runtime/pipeline.py`.
- Suggestion: Document a single preferred import path to reduce duplicated documentation burden and future migration friction.

### Entry 34
- Feature noticed: Sync and async runners both implement mature artifact/validation flows.
- Suggestion: Factor more shared logic into common internals to reduce maintenance overhead and prevent subtle behavior drift.

### Entry 35
- Feature noticed: Models and probes have rich type interfaces and protocol support.
- Suggestion: Consider keeping source docstrings more concise and shifting expansive usage walkthroughs to docs pages to improve code readability.

### Entry 36
- Feature noticed: Export pipeline supports environment-based key loading for encryption.
- Suggestion: Add helper UX for key validation/generation guidance so encryption failures are easier for users to debug quickly.

### Entry 37
- Feature noticed: Dataset package exports currently expose commitments but not TUF fetch helpers.
- Suggestion: Revisit public exports for consistency (or document intentional scoping) so discoverability aligns with available dataset-security capabilities.

### Entry 38
- Feature noticed: LangChain integration provides both ChatModel and Runnable adapters with graceful optional-dependency handling.
- Suggestion: Add explicit docs for role/content conversion semantics (especially tool/multimodal message handling) to avoid integration surprises.

### Entry 39
- Feature noticed: Async execution includes a dedicated timeout wrapper and structured exception mapping.
- Suggestion: Persist timeout metadata in run artifacts/results for easier post-run diagnosis of intermittent execution failures.

### Entry 40
- Feature noticed: The project includes both `config.py` (Pydantic model family) and `config_types.py` (`RunConfig` dataclass-style runtime controls).
- Suggestion: Publish a clear "which config API to use" guideline (and long-term convergence plan) to reduce contributor confusion.

### Entry 41
- Feature noticed: `doctor` gives good coverage for common optional NLP/visualization dependencies and key env vars.
- Suggestion: Extend diagnostics for verifiable-evaluation tooling (`cosign`, `oras`, `tuf`) to improve readiness checks for Ultimate workflows.

### Entry 42
- Feature noticed: `validate` command supports both config-file and run-directory validation paths in one entrypoint.
- Suggestion: Consider split command UX (`validate-config` / `validate-run`) or clearer argument naming to simplify behavior expectations.

### Entry 43
- Feature noticed: Run-directory validation already leverages schema registry/versioning for `manifest.json` and `records.jsonl`.
- Suggestion: Expand config-file validation depth (for example dataset format-specific constraints) to catch more issues before execution.

### Entry 44
- Feature noticed: LangChain adapter has a practical fallback path for models without native chat support.
- Suggestion: Expose a configurable chat-to-prompt rendering strategy so advanced users can tune prompt framing for their models.

### Entry 45
- Feature noticed: Pipeline/middleware architecture is feature-rich and documented comprehensively.
- Suggestion: Add a concise "recommended middleware order" checker or warning utility in code to prevent accidental misordering in production setups.

### Entry 46
- Feature noticed: Analysis capabilities are extensive and mature (statistics, export, visualization, comparison, evaluation).
- Suggestion: Break very large analysis modules into smaller focused units to improve maintainability and reduce merge-conflict pressure.

### Entry 47
- Feature noticed: Backward-compatibility shims (`insideLLMs.export`, `insideLLMs.visualization`) preserve old import paths cleanly.
- Suggestion: Keep a small import-path migration matrix in docs so users can quickly identify canonical module locations.

### Entry 48
- Feature noticed: `report` command can regenerate `summary.json`/`report.html` directly from `records.jsonl`, which is operationally helpful.
- Suggestion: Add explicit metadata fields indicating whether summary/report were regenerated and by which tool version for audit clarity.

### Entry 49
- Feature noticed: Basic HTML report fallback exists when interactive visualization deps are missing.
- Suggestion: Move fallback HTML rendering to templated assets to simplify styling updates and keep command code focused on data assembly.

### Entry 50
- Feature noticed: `validate` command supports schema-version override and warn/strict modes for run artifacts.
- Suggestion: Add a concise per-file error summary (counts by failure type) to make large-run validation triage faster.

### Entry 51
- Feature noticed: Configuration support gracefully falls back when Pydantic is unavailable.
- Suggestion: Emit a stronger runtime warning in fallback mode so users know schema/type validation guarantees are reduced.

### Entry 52
- Feature noticed: Dataset JSONL loading currently surfaces precise line-number errors for malformed records.
- Suggestion: Consider an optional "skip bad lines" mode for exploratory workflows while keeping strict mode as default for CI paths.

### Entry 53
- Feature noticed: Report reconstruction logic in `_report_builder` handles both harness-style and single-run record layouts.
- Suggestion: Add focused tests around mixed/partial record metadata edge cases to harden reconstruction behavior as formats evolve.

### Entry 54
- Feature noticed: `DOCUMENTATION_INDEX.md` provides strong navigation for core docs and common workflows.
- Suggestion: Add dedicated index entries for verifiable-evaluation topics (`attest`, signing, SCITT, OpenVEX, OCI publish) for faster discovery.

### Entry 55
- Feature noticed: The repository includes a substantial `compliance_intelligence/` demonstration with clear architecture and scenario coverage.
- Suggestion: Cross-link this subproject from the root README/docs index with a short status note so users know how it relates to core `insideLLMs` scope.

### Entry 56
- Feature noticed: Several documentation files include detailed static metadata (line counts/sizes/section inventories).
- Suggestion: Consider generating these metadata sections automatically to reduce drift as documentation evolves.

### Entry 57
- Feature noticed: Report generation currently relies on compatibility imports that still work reliably.
- Suggestion: Prefer canonical module paths in command implementations where practical to reduce long-term shim coupling.

### Entry 58
- Feature noticed: Feature coverage spans core evaluation plus advanced provenance/security workflows.
- Suggestion: Revisit extras/dependency grouping so users can install a dedicated profile for verifiable-evaluation features without pulling unrelated stacks.

### Entry 59
- Feature noticed: CI workflow is comprehensive (lint/format/typecheck/tests/docs build/wiki link checks) and exercises multiple Python versions.
- Suggestion: Keep local guidance and AGENTS/README references aligned with CI thresholds and gates (for example coverage floor) to avoid contributor confusion.

### Entry 60
- Feature noticed: Release workflow cleanly builds artifacts and publishes GitHub release + optional PyPI.
- Suggestion: Add provenance/supply-chain release steps (artifact attestations, signature verification) to align with the project’s verifiable-evaluation direction.

### Entry 61
- Feature noticed: Pages workflow is minimal and focused on `wiki/` publication.
- Suggestion: Add lightweight docs link/lint validation in the Pages workflow too (not only CI) to catch publishing regressions close to deployment.

### Entry 62
- Feature noticed: Contributor docs provide clear setup and quality-check commands.
- Suggestion: Update architecture path references in `CONTRIBUTING.md` to match current package layout (e.g., CLI package and runtime modules) for smoother onboarding.

### Entry 63
- Feature noticed: `examples/` offers practical runnable workflows from quickstart to deterministic golden-path and programmatic harnesses.
- Suggestion: Add a compact compatibility matrix indicating which examples are fully offline vs API-key dependent so users can pick the right starting point quickly.

### Entry 64
- Feature noticed: Automation workflows include both ad-hoc Claude invocation and PR review plugin execution.
- Suggestion: Document governance/expected behavior for these bots (scope, etiquette, required secrets) to keep maintainer expectations explicit.

### Entry 65
- Feature noticed: Root helper script (`scripts/checks.sh`) mirrors essential local quality gates.
- Suggestion: Consider adding a "fast" variant (or documented subset) for quick iterative pre-checks while retaining full checks for final verification.

### Entry 66
- Feature noticed: Stability policy docs and PR template include clear contract-impact checkpoints.
- Suggestion: Add direct references to concrete example PRs (or mini playbooks) so contributors can model the expected stability-change workflow.

### Entry 67
- Feature noticed: `SECURITY.md` provides a responsible disclosure process and practical secure-usage tips.
- Suggestion: Keep supported-version matrix synchronized with actual release line (`0.2.x` currently present) to avoid ambiguity during vulnerability reporting.

### Entry 68
- Feature noticed: Issue templates are well-structured and capture reproducibility details.
- Suggestion: Refresh default version placeholders/examples periodically (for example `0.1.0` -> current release line) to reduce stale metadata in incoming reports.

### Entry 69
- Feature noticed: Plugin docs clearly define entry point group and registration signatures.
- Suggestion: Add a small troubleshooting section (discovery disabled, import errors, registration collisions) to reduce friction for third-party plugin authors.

### Entry 70
- Feature noticed: Project workflows include AI-assisted automation paths for both on-demand and PR review contexts.
- Suggestion: Document fallback reviewer expectations when automation secrets/plugins are unavailable so review quality remains predictable.

### Entry 71
- Feature noticed: Issue template config includes Discussions and README contact links.
- Suggestion: Consider linking directly to the docs site or documentation index for faster self-service routing before issue filing.

### Entry 72
- Feature noticed: The codebase has many very large modules (3k-4k+ LOC) across analysis and core feature areas.
- Suggestion: Prioritize incremental module decomposition around the largest files to improve reviewability, test targeting, and contributor throughput.

### Entry 73
- Feature noticed: Receipt middleware captures per-call hashes/latency with sync + async support and canonical JSONL emission.
- Suggestion: Factor shared receipt construction into common helpers to reduce duplicated code paths and keep sync/async behavior tightly aligned.

### Entry 74
- Feature noticed: Chat request hashing in receipt middleware performs message normalization inline.
- Suggestion: Reuse a single robust message-normalization utility (shared with other adapters) to reduce edge-case risk across message object types.

### Entry 75
- Feature noticed: Infrastructure modules (`caching`, `rate_limiting`, `cost_tracking`) are feature-rich and comprehensive.
- Suggestion: Consider publishing focused "recommended subset" guides/presets for common production scenarios so users can adopt capabilities progressively.

### Entry 76
- Feature noticed: Registry system supports plugin discovery and flexible factory signatures.
- Suggestion: Add optional diagnostics that report plugin load results/conflicts at startup to improve operability in extension-heavy deployments.

### Entry 77
- Feature noticed: Stability contract tests exist and already assert core CLI/schema invariants.
- Suggestion: Expand these tests to include newer user-facing commands/flags (e.g., attest/sign/verify-signatures surfaces) to keep guarantees current.

### Entry 78
- Feature noticed: Ultimate-mode integration tests cover artifact/attestation flows at runner level.
- Suggestion: Add direct CLI command tests for `attest`, `sign`, and `verify-signatures` to validate end-user command UX and error handling paths.

### Entry 79
- Feature noticed: Changelog follows Keep a Changelog structure with clear Added/Changed/Removed sections.
- Suggestion: Keep release links/version ranges synchronized with latest tags to avoid stale compare targets and improve release traceability.

### Entry 80
- Feature noticed: Security policy includes clear handling expectations and SLA-style response windows.
- Suggestion: Add explicit mapping between vulnerability severity tiers and target remediation timelines for more precise maintainer/respondent expectations.

### Entry 81
- Feature noticed: Verifiable-evaluation capabilities (attestations/signing/SCITT/OpenVEX/OCI publish) are implemented in code and CLI.
- Suggestion: Add dedicated user docs/wiki pages and cross-links for these workflows so the feature set is discoverable beyond source inspection.

### Entry 82
- Feature noticed: Documentation index is strong for core probe/run/diff workflows.
- Suggestion: Include a focused "advanced assurance" section grouping determinism, policy, attestations, signatures, and transparency topics in one place.

### Entry 83
- Feature noticed: `docs/plans/` contains extensive historical implementation planning artifacts.
- Suggestion: Consider separating archival planning docs from primary user/contributor docs navigation to reduce noise for first-time readers.

### Entry 84
- Feature noticed: Existing docs mention plugin extensibility with concise setup examples.
- Suggestion: Add one end-to-end external plugin mini-repo example (with tests/packaging) to demonstrate best-practice extension structure.

### Entry 85
- Feature noticed: Multiple compatibility shims are intentionally maintained across runtime/analysis/trace surfaces.
- Suggestion: Track shim inventory centrally (with intended removal/indefinite-support status) so maintenance decisions stay explicit.

### Entry 86
- Feature noticed: Some module docstrings and historical docs still reference older import paths (e.g., `insideLLMs.statistics` / `insideLLMs.trace_config`).
- Suggestion: Run periodic import-path lint checks across docs/docstrings to keep guidance aligned with canonical module layout.

### Entry 87
- Feature noticed: Runtime package re-exports many symbols for convenience.
- Suggestion: Document a concise "preferred import style" for new code to balance ergonomics with long-term API clarity.

### Entry 88
- Feature noticed: Reproducibility and observability compatibility modules are richly documented and onboarding-friendly.
- Suggestion: Add a shorter "quick reference first" section at top of long shim docs so readers can find canonical imports immediately.

### Entry 89
- Feature noticed: Benchmark scripts provide practical import/probe/runtime measurements with clear human-readable output.
- Suggestion: Add machine-readable benchmark output mode (JSON) to support trend tracking in CI or local performance dashboards.

### Entry 90
- Feature noticed: Benchmark README includes threshold guidance for import performance.
- Suggestion: Clarify that thresholds are environment-dependent (and optionally provide normalized baselines) to reduce misleading interpretations across machines.

### Entry 91
- Feature noticed: CI harness assets (`ci/harness.yaml`, dataset) provide an excellent deterministic offline gate.
- Suggestion: Add a documented way to extend this harness with additional low-cost probes while preserving offline determinism.

### Entry 92
- Feature noticed: Schema command supports listing, dumping, and validating versioned artifacts with warn/strict modes.
- Suggestion: Add examples in top-level docs showing common `insidellms schema validate` workflows for both JSON and JSONL inputs.

### Entry 93
- Feature noticed: Wiki home and determinism pages are clear and compelling for core CI-diff workflows.
- Suggestion: Add direct links from these pages to advanced verifiable-evaluation workflows once those docs are introduced.

### Entry 94
- Feature noticed: Wiki performance/caching guidance is detailed and operationally practical.
- Suggestion: Refresh internal code-path references periodically (e.g., legacy `insideLLMs/cli.py` mentions) to keep docs aligned with current package structure.

### Entry 95
- Feature noticed: Determinism guidance exists in both `docs/DETERMINISM.md` and wiki pages.
- Suggestion: Define one canonical source and cross-reference the other to reduce drift in terminology and guarantees over time.

### Entry 96
- Feature noticed: Experiment tracking docs are comprehensive and include backend-specific behavior details.
- Suggestion: Add a short section on data-retention/privacy considerations for tracked artifacts to support compliance-sensitive adopters.

### Entry 97
- Feature noticed: Most current top-level docs appear aligned with canonical module paths after prior cleanup.
- Suggestion: Keep a lightweight recurring check for stale path references in wiki/reference pages (for example performance docs mentioning legacy CLI file paths).

### Entry 98
- Feature noticed: Historical planning documents intentionally reference removed/deprecated modules and migration steps.
- Suggestion: Clearly label these plan docs as archival in navigation/search contexts so users do not mistake them for current API guidance.

### Entry 99
- Feature noticed: The repository already has tooling for wiki link validation in CI.
- Suggestion: Extend docs checks to include optional "canonical import path" linting to catch drift early as modules evolve.
