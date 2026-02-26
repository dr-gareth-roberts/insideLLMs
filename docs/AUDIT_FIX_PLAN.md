# Audit Fix Plan

Companion to `AUDIT_FINDINGS.md`. Each section is a self-contained, reviewable change set ordered by impact.

---

## Fix 1 — evalbom SPDX3 type casing (🔴 HIGH — 1 failing test)

**Root cause**: `emit_spdx3()` emits `"type": "software"` but SPDX 3.0 uses `"Software"`.

```diff
# insideLLMs/evalbom.py
-                    "type": "software",
+                    "type": "Software",

-                    "type": "software",   # probe element (line ~63)
+                    "type": "Software",
```

> Note: Dataset elements use `"type": "dataset"` — verify against SPDX 3.0 spec whether that should also be `"Dataset"`.

**Test impact**: `tests/test_evalbom.py::test_emit_spdx3` → PASS

---

## Fix 2 — Matplotlib `labels=` deprecated parameter (🟡 LOW — future break)

**Root cause**: `plt.boxplot(labels=...)` renamed to `tick_labels=` in Matplotlib 3.9; removed in 3.11.

**File**: `insideLLMs/analysis/visualization.py`

```diff
# Line ~1207-1210
-        plt.boxplot(
-            latencies_by_model.values(),
-            labels=latencies_by_model.keys(),
-        )
+        plt.boxplot(
+            latencies_by_model.values(),
+            tick_labels=list(latencies_by_model.keys()),
+        )

# Line ~1680
-        plt.boxplot(by_cat.values(), labels=by_cat.keys())
+        plt.boxplot(list(by_cat.values()), tick_labels=list(by_cat.keys()))
```

---

## Fix 3 — Add `serving` optional extras to pyproject.toml (🟠 MEDIUM)

**Root cause**: FastAPI + uvicorn used in `deployment.py` but undeclared as optional deps.

```diff
# pyproject.toml
 [project.optional-dependencies]
 nlp = [...]
 visualization = [...]
 dev = [...]
 langchain = [...]
+serving = [
+    "fastapi>=0.100.0",
+    "uvicorn>=0.22.0",
+]
 all = [
-    "insideLLMs[nlp,visualization,dev]",
+    "insideLLMs[nlp,visualization,dev,serving]",
 ]
```

Also update `.github/workflows/ci.yml` test step to use the extras instead of ad-hoc install:
```diff
-          pip install -e ".[dev,nlp,visualization]"
-          pip install fastapi uvicorn
+          pip install -e ".[dev,nlp,visualization,serving]"
```

---

## Fix 4 — Pydantic v2 `protected_namespaces` warnings (🟡 LOW)

Add `protected_namespaces = ()` to every model that uses `model_*` field names.

### `insideLLMs/config.py`
```diff
 class ModelConfig(BaseModel):
+    model_config = ConfigDict(protected_namespaces=(), extra="allow", validate_default=True)
     model_id: str = Field(...)
```

### `insideLLMs/schemas/v1_0_0.py`
The `_BaseSchema` root class sets `extra="forbid"`. Add namespace suppression there so all subclasses inherit it:
```diff
 class _BaseSchema(BaseModel):
     if _PYDANTIC_V2:
-        model_config = ConfigDict(extra="forbid")
+        model_config = ConfigDict(extra="forbid", protected_namespaces=())
```

### `insideLLMs/deployment.py` — `GenerateResponse`, `HealthResponse`
```diff
 class GenerateResponse(BaseModel):
+    model_config = ConfigDict(protected_namespaces=())
     response: str
     model_id: Optional[str] = None
```

---

## Fix 5 — pytest-asyncio loop scope (🟡 LOW)

```diff
# pyproject.toml [tool.pytest.ini_options]
 addopts = ["-v", "--tb=short", "--strict-markers"]
 asyncio_mode = "auto"
+asyncio_default_fixture_loop_scope = "function"
```

---

## Fix 6 — Stale DBSCAN docstring (🟡 LOW)

**File**: `insideLLMs/semantic_analysis.py` — `ClusteringMethod` enum

```diff
     DBSCAN : str
         Density-Based Spatial Clustering of Applications with Noise.
         Finds clusters of arbitrary shape based on density.
         Best for: Unknown number of clusters, noise detection.
-        Note: Currently not implemented, raises NotImplementedError.
```

---

## Fix 7 — Pydantic v1 `class Config` in deployment.py (ℹ️ INFO)

**File**: `insideLLMs/deployment.py` — `GenerateRequest` inner class

```diff
-        class Config:
-            json_schema_extra = {
-                "example": {
-                    "prompt": "What is the capital of France?",
-                    "temperature": 0.7,
-                    "max_tokens": 100,
-                }
-            }
+        model_config = ConfigDict(
+            json_schema_extra={
+                "example": {
+                    "prompt": "What is the capital of France?",
+                    "temperature": 0.7,
+                    "max_tokens": 100,
+                }
+            }
+        )
```

---

## Fix 8 — Remove "stub" label from `claims/compiler.py` (ℹ️ INFO)

```diff
-"""Claims compiler: read claims.yaml, compute effects/CIs, emit claims.json + verification.json (stub)."""
+"""Claims compiler: read claims.yaml, evaluate metric thresholds, emit claims.json + verification.json."""
```

---

## Fix 9 — Mark process-pool tests with environment skip guard (ℹ️ INFO)

**File**: `tests/test_misc_coverage.py`

```python
import os
import pytest

_CAN_SPAWN = True
try:
    import multiprocessing
    p = multiprocessing.Process(target=lambda: None)
    p.start()
    p.join()
except (PermissionError, OSError):
    _CAN_SPAWN = False

requires_process_spawn = pytest.mark.skipif(
    not _CAN_SPAWN, reason="ProcessPoolExecutor not permitted in this environment"
)
```

Then decorate the 5 affected test methods with `@requires_process_spawn`.

---

## Fix 10 — Delete stray `package-lock.json` (ℹ️ INFO)

```bash
rm package-lock.json
echo "package-lock.json" >> .gitignore
```

---

## Fix 11 — Clarify `datasets/tuf_client.py` stub label (ℹ️ INFO)

```diff
-"""TUF client for verified dataset fetch (stub).
-
-Fetch dataset by name/version and verify TUF metadata before run.
-"""
+"""TUF client for verified dataset fetch.
+
+Fetches a dataset by name/version and verifies TUF metadata before a run.
+When the ``tuf`` package is available, uses ``tuf.ngclient.Updater`` for
+real verification. Falls back to a mock implementation for testing/offline use.
+"""
```

---

## Verification Checklist

After applying all fixes, run:

```bash
# All tests should pass (except env-restricted ones now marked skip)
python -m pytest tests/ -q

# Confirm zero DeprecationWarning from Pydantic
python -m pytest tests/ -W error::DeprecationWarning -q --ignore=tests/test_caching.py --ignore=tests/test_cache.py

# Confirm zero evalbom failure
python -m pytest tests/test_evalbom.py -v

# Confirm matplotlib warning gone
python -m pytest tests/test_visualization.py tests/test_interactive_visualization.py -W error::DeprecationWarning -v
```
