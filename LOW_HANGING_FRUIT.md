# Low Hanging Fruit - insideLLMs

Analysis date: 2026-01-26

## 1. Duplicate Utility Functions (High Impact, Easy Fix)

**Issue:** `_stable_json_dumps` and `_fingerprint_value` are duplicated in two files:
- `insideLLMs/cli.py` (lines 973, 1005)
- `insideLLMs/runtime/runner.py` (lines 1480, 1566)

**Fix:** Extract these to a shared utility module (e.g., `utils.py` or `reproducibility.py`) and import from both locations. This eliminates code drift and reduces maintenance burden.

---

## 2. Missing `__all__` Exports in Key Modules (Low Effort, Cleaner API)

**Issue:** Two important modules lack explicit `__all__` definitions:
- `cli.py` - Contains many internal `_`-prefixed helpers alongside public classes like `ProgressBar`, `Spinner`
- `exceptions.py` - Defines a large exception hierarchy without clarifying which are public

**Fix:** Add `__all__` lists to both files. This clarifies the public API and prevents accidental use of internal utilities.

---

## 3. Test Coverage Gaps for Core Modules (Easy to Add)

**Issue:** Several source modules have no corresponding test files:
- `caching_unified.py` (109KB) - no `test_caching_unified.py`
- `config_types.py` (75KB) - no `test_config_types.py`

Additionally, many NLP tests skip when optional dependencies aren't installed, reducing coverage in CI.

**Fix:** Add basic test stubs for the missing modules. For `config_types.py`, this would primarily be dataclass instantiation and validation tests which are quick to write.

---

## Bonus: Deprecation Warning for runner.py Shim

**Issue:** The `runner.py` shim documents that it's deprecated but doesn't emit a `DeprecationWarning`.

**Fix:** Add a 2-line warning to guide users to the new import path (`insideLLMs.runtime.runner`).
