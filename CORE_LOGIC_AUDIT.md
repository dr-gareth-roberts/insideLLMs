# Core Logic Audit - Bug and Improvement Analysis

**Date:** 2026-01-26  
**Scope:** Core framework components (runner, models, probes, registry, CLI)

---

## Critical Issues

### 1. Race Condition in AsyncProbeRunner.completed Counter

**Location:** `insideLLMs/runtime/runner.py:2968`

**Issue:**
```python
async def run_single(index: int, item: Any) -> None:
    nonlocal completed
    async with semaphore:
        try:
            # ... probe execution ...
        finally:
            completed += 1  # NOT THREAD-SAFE
```

**Problem:** `completed += 1` is not atomic. Multiple coroutines can read-modify-write simultaneously, causing lost increments.

**Impact:** Progress reporting incorrect. Potential infinite loops if completion tracking fails.

**Fix:**
```python
# Option 1: Use asyncio.Lock
write_lock = asyncio.Lock()
async with write_lock:
    completed += 1

# Option 2: Use atomic counter
from threading import Lock
completed_lock = Lock()
with completed_lock:
    completed += 1
```

**Priority:** HIGH - Can cause subtle bugs in concurrent execution

---

### 2. Bare Exception Handlers Throughout

**Location:** 15+ instances in `runtime/runner.py`, 100+ across codebase

**Examples:**
```python
# Line 929
try:
    manifest["library_version"] = getattr(insideLLMs, "__version__", None)
except Exception:  # TOO BROAD
    pass

# Line 1644
try:
    marker.write_text("insideLLMs run directory\n", encoding="utf-8")
except Exception:  # TOO BROAD
    pass

# Line 2953
except Exception as e:  # Should be more specific
    errors[index] = e
```

**Problem:** Catches everything including KeyboardInterrupt, SystemExit, programming errors. Masks bugs.

**Fix:** Be specific about what you're catching:
```python
# Good
except (IOError, OSError) as e:
    logger.warning(f"Failed to write marker: {e}")

# Good
except (ImportError, AttributeError):
    manifest["library_version"] = None
```

**Priority:** HIGH - Masks bugs, makes debugging harder

---

### 3. latency_ms Always None

**Location:** Throughout `runtime/runner.py` (8+ instances)

**Issue:**
```python
probe_result = ProbeResult(
    input=item,
    output=output,
    status=ResultStatus.SUCCESS,
    latency_ms=None,  # ALWAYS None
    metadata={},
)
```

**Problem:** Latency is never measured, always set to `None`. Field exists but unused.

**Impact:** 
- Users can't analyse performance
- Can't identify slow prompts
- Can't optimise for latency

**Fix:**
```python
import time

start = time.perf_counter()
output = self.probe.run(self.model, item, **probe_kwargs)
latency_ms = (time.perf_counter() - start) * 1000

probe_result = ProbeResult(
    input=item,
    output=output,
    status=ResultStatus.SUCCESS,
    latency_ms=latency_ms,
    metadata={},
)
```

**Note:** Documentation says latency is omitted for determinism. But it could be:
- Measured and stored in `metadata` (not in deterministic fields)
- Optionally enabled with `--track-latency` flag
- Stored in separate `latency.jsonl` file

**Priority:** MEDIUM - Feature exists but unused

---

### 4. Silent Failures in Critical Paths

**Location:** Multiple locations

**Examples:**
```python
# Line 1644 - Marker write failure silently ignored
try:
    marker.write_text("insideLLMs run directory\n", encoding="utf-8")
except Exception:
    pass  # Silent failure

# Line 3490 - Dataset hash failure silently ignored
try:
    dataset["dataset_hash"] = f"sha256:{hasher.hexdigest()}"
except Exception:
    pass  # Silent failure - breaks determinism!
```

**Problem:** Critical operations fail silently. No logging. User unaware.

**Fix:**
```python
import logging
logger = logging.getLogger(__name__)

try:
    marker.write_text("insideLLMs run directory\n", encoding="utf-8")
except (IOError, OSError) as e:
    logger.warning(f"Failed to write run marker: {e}")
    # Continue - not critical

try:
    dataset["dataset_hash"] = f"sha256:{hasher.hexdigest()}"
except Exception as e:
    logger.error(f"Failed to hash dataset: {e}")
    # This IS critical for determinism - should raise or warn loudly
```

**Priority:** HIGH - Silent failures break determinism guarantees

---

### 5. Inconsistent Error Handling in CLI

**Location:** `insideLLMs/cli.py` (multiple commands)

**Issue:** Some commands catch broad `Exception`, others catch specific errors. No consistency.

**Examples:**
```python
# cmd_run - catches broad Exception
except Exception as e:
    print_error(f"Error running experiment: {e}")

# cmd_validate - catches specific OutputValidationError
except OutputValidationError as e:
    errors += 1

# cmd_diff - catches broad Exception
except Exception as e:
    print_error(f"Could not read records.jsonl: {e}")
```

**Problem:** Inconsistent UX. Some errors give context, others don't.

**Fix:** Standardise error handling:
```python
try:
    # operation
except SpecificError as e:
    print_error(f"Operation failed: {e}")
    if args.verbose:
        traceback.print_exc()
    return 1
except Exception as e:
    print_error(f"Unexpected error: {e}")
    print_error("Please report this issue with --verbose output")
    if args.verbose:
        traceback.print_exc()
    return 2
```

**Priority:** MEDIUM - UX issue, not correctness

---

## Design Issues

### 6. No Structured Logging

**Location:** Entire codebase

**Issue:** Uses `print()` and `print_error()` instead of structured logging.

**Problem:**
- Can't filter by severity
- Can't route to different outputs
- Can't integrate with observability tools
- Difficult to debug in production

**Fix:**
```python
import logging

logger = logging.getLogger(__name__)

# Instead of
print_error(f"Failed: {e}")

# Use
logger.error("Operation failed", exc_info=e, extra={"operation": "run", "run_id": run_id})
```

**Priority:** HIGH - Critical for production use

---

### 7. Missing Input Validation in Critical Paths

**Location:** `AsyncProbeRunner.run()`, `ProbeRunner.run()`

**Issue:** Minimal validation before expensive operations.

**Examples:**
```python
# No validation of concurrency value
semaphore = asyncio.Semaphore(concurrency)  # What if concurrency is 0? Negative?

# No validation of batch_workers
resolved_batch_workers = batch_workers if batch_workers is not None else max(1, concurrency)
# What if batch_workers is 0? Negative?
```

**Fix:**
```python
if concurrency < 1:
    raise ValueError(f"concurrency must be >= 1, got {concurrency}")

if batch_workers is not None and batch_workers < 1:
    raise ValueError(f"batch_workers must be >= 1, got {batch_workers}")
```

**Priority:** MEDIUM - Edge case handling

---

### 8. Inconsistent Type Hints

**Location:** Throughout codebase

**Issue:** Some functions have complete type hints, others have `Any` or missing hints.

**Examples:**
```python
# Good
def _build_model_spec(model: Any) -> dict[str, Any]:

# Bad - should specify return type
def _normalize_info_obj_to_dict(info_obj: Any):  # Missing return type

# Bad - Any is too broad
def _build_result_record(..., item: Any, ...):  # Could be more specific
```

**Fix:** Add complete type hints:
```python
def _normalize_info_obj_to_dict(info_obj: Any) -> dict[str, Any]:
    ...

# Or use Protocol
from typing import Protocol

class PromptItem(Protocol):
    ...

def _build_result_record(..., item: PromptItem, ...):
```

**Priority:** LOW - Code quality, not correctness

---

### 9. No Timeout Handling in Async Operations

**Location:** `AsyncProbeRunner.run()`

**Issue:** No timeout for individual probe executions. Can hang indefinitely.

**Example:**
```python
output = await loop.run_in_executor(
    None,
    lambda: self.probe.run(self.model, item, **probe_kwargs),
)
# No timeout - can hang forever
```

**Fix:**
```python
try:
    output = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: self.probe.run(self.model, item, **probe_kwargs),
        ),
        timeout=timeout_seconds
    )
except asyncio.TimeoutError:
    raise ProbeExecutionError(f"Probe execution timed out after {timeout_seconds}s")
```

**Priority:** HIGH - Can cause hangs in production

---

### 10. write_ready_records() Not Actually Async

**Location:** `runtime/runner.py:2887-2930`

**Issue:**
```python
async def write_ready_records() -> None:
    nonlocal next_write_index
    async with write_lock:
        while next_write_index < len(results) and results[next_write_index] is not None:
            # ... synchronous file I/O ...
            records_fp.write(_stable_json_dumps(record) + "\n")  # BLOCKING
            records_fp.flush()  # BLOCKING
```

**Problem:** Declared `async` but performs blocking I/O. Defeats purpose of async.

**Fix:**
```python
# Option 1: Use aiofiles
import aiofiles

async with aiofiles.open(records_path, mode) as records_fp:
    await records_fp.write(_stable_json_dumps(record) + "\n")
    await records_fp.flush()

# Option 2: Run in executor
await loop.run_in_executor(
    None,
    lambda: records_fp.write(_stable_json_dumps(record) + "\n")
)
```

**Priority:** MEDIUM - Performance issue in async code

---

## Code Quality Issues

### 11. Overly Long Functions

**Location:** `AsyncProbeRunner.run()` is 400+ lines

**Issue:** Single function does too much:
- Config merging
- Run ID generation
- Directory preparation
- Async execution
- Progress tracking
- Artefact writing
- Manifest generation

**Fix:** Extract into smaller functions:
```python
async def run(self, ...):
    config = self._merge_config(...)
    run_id = self._generate_run_id(...)
    run_dir = self._prepare_run_dir(...)
    results = await self._execute_probes(...)
    await self._write_artefacts(...)
    return self._build_experiment_result(...)
```

**Priority:** LOW - Maintainability, not correctness

---

### 12. Duplicate Code Between ProbeRunner and AsyncProbeRunner

**Location:** `runtime/runner.py`

**Issue:** Significant duplication in:
- Config merging logic
- Run ID generation
- Directory preparation
- Artefact writing
- Manifest generation

**Fix:** Extract shared logic to helper functions or base class.

**Priority:** LOW - Code quality

---

### 13. No Logging in Critical Paths

**Location:** Throughout runner.py

**Issue:** Silent execution. No debug logging. Hard to troubleshoot.

**Examples:**
```python
# No logging
resolved_run_id = _deterministic_run_id_from_inputs(...)

# Should log
logger.debug(f"Generated run_id: {resolved_run_id}")

# No logging
_prepare_run_dir(resolved_run_dir, overwrite=overwrite, run_root=root)

# Should log
logger.info(f"Prepared run directory: {resolved_run_dir}")
```

**Fix:** Add structured logging throughout:
```python
logger.info("Starting probe run", extra={
    "run_id": resolved_run_id,
    "model": model_spec["model_id"],
    "probe": probe_spec["probe_id"],
    "examples": len(prompt_set)
})
```

**Priority:** HIGH - Essential for debugging production issues

---

## Prioritised Fix List

### P0 - Critical (Fix Immediately)

1. **Race condition in `completed` counter** - Use lock or atomic counter
2. **Silent dataset hash failures** - Log or raise, don't silently pass
3. **No timeout in async probe execution** - Add `asyncio.wait_for()`
4. **Add structured logging** - Replace print() with logging

### P1 - High (Fix Soon)

5. **Bare exception handlers** - Make specific (IOError, OSError, etc.)
6. **Blocking I/O in async function** - Use aiofiles or run_in_executor
7. **Missing input validation** - Validate concurrency, batch_workers

### P2 - Medium (Improve Quality)

8. **latency_ms always None** - Either remove field or implement measurement
9. **Inconsistent CLI error handling** - Standardise pattern
10. **Missing type hints** - Add return types, use Protocols

### P3 - Low (Technical Debt)

11. **Long functions** - Extract AsyncProbeRunner.run() into smaller pieces
12. **Code duplication** - Share logic between sync/async runners
13. **No timeout configuration** - Add timeout parameters

---

## Recommended Immediate Fixes

### Fix 1: Race Condition

```python
# In AsyncProbeRunner.run()
completed_lock = asyncio.Lock()

async def run_single(index: int, item: Any) -> None:
    nonlocal completed
    async with semaphore:
        try:
            # ... execution ...
        finally:
            async with completed_lock:
                completed += 1
            # ... progress callback ...
```

### Fix 2: Structured Logging

```python
# At module level
import logging
logger = logging.getLogger(__name__)

# Replace print_error/print_warning
logger.error("Operation failed", exc_info=e, extra={"context": "..."})
logger.warning("Non-critical issue", extra={"context": "..."})
logger.info("Operation started", extra={"run_id": run_id})
logger.debug("Detailed state", extra={"state": state_dict})
```

### Fix 3: Specific Exception Handling

```python
# Instead of
except Exception:
    pass

# Use
except (IOError, OSError) as e:
    logger.warning(f"I/O operation failed: {e}")
except (ImportError, AttributeError):
    # Expected in some environments
    pass
```

### Fix 4: Input Validation

```python
# At start of AsyncProbeRunner.run()
if concurrency < 1:
    raise ValueError(f"concurrency must be >= 1, got {concurrency}")

if batch_workers is not None and batch_workers < 1:
    raise ValueError(f"batch_workers must be >= 1, got {batch_workers}")

if not prompt_set:
    raise ValueError("prompt_set cannot be empty")
```

---

## Non-Critical Improvements

### Improvement 1: Latency Tracking

**Option A:** Remove the field entirely (it's always None)

**Option B:** Implement it properly:
```python
# Add flag
track_latency: bool = False

# Measure if enabled
if track_latency:
    start = time.perf_counter()
    output = self.probe.run(self.model, item, **probe_kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
else:
    output = self.probe.run(self.model, item, **probe_kwargs)
    latency_ms = None
```

**Option C:** Store in metadata (preserve determinism):
```python
metadata = {}
if track_latency:
    metadata["latency_ms"] = latency_ms

probe_result = ProbeResult(
    ...,
    latency_ms=None,  # Deterministic field stays None
    metadata=metadata  # Non-deterministic data here
)
```

### Improvement 2: Better Error Context

```python
# Current
except Exception as e:
    errors[index] = e

# Better
except Exception as e:
    logger.error(
        "Probe execution failed",
        exc_info=e,
        extra={
            "example_id": index,
            "input": item,
            "model": model_spec["model_id"],
            "probe": probe_spec["probe_id"]
        }
    )
    errors[index] = e
```

### Improvement 3: Progress Callback Error Handling

```python
# Current - no error handling
_invoke_progress_callback(progress_callback, ...)

# Better
try:
    _invoke_progress_callback(progress_callback, ...)
except Exception as e:
    logger.warning(f"Progress callback failed: {e}")
    # Don't let callback errors break the run
```

---

## Testing Gaps

### Missing Tests

1. **Race condition test** - Concurrent execution with high concurrency
2. **Timeout test** - Slow probe execution
3. **Error recovery test** - Partial failures in async execution
4. **Resume validation** - Edge cases in resume logic

### Recommended Tests

```python
# Test race condition
async def test_concurrent_completion_tracking():
    # Run with high concurrency, verify completed count matches total
    runner = AsyncProbeRunner(model, probe)
    results = await runner.run(prompts, concurrency=50)
    assert len(results) == len(prompts)

# Test timeout
async def test_probe_timeout():
    slow_probe = SlowProbe(delay=10)
    runner = AsyncProbeRunner(model, slow_probe)
    with pytest.raises(TimeoutError):
        await runner.run(prompts, timeout=1)

# Test error recovery
async def test_partial_failures():
    failing_probe = PartiallyFailingProbe(fail_rate=0.5)
    runner = AsyncProbeRunner(model, failing_probe)
    results = await runner.run(prompts)
    # Should have mix of success and error
    assert any(r["status"] == "success" for r in results)
    assert any(r["status"] == "error" for r in results)
```

---

## Summary

**Critical bugs:** 4 (race condition, silent failures, no timeouts, bare exceptions)

**Design issues:** 3 (no logging, inconsistent error handling, latency unused)

**Code quality:** 3 (long functions, duplication, missing type hints)

**Estimated fix time:**
- P0 fixes: 2-3 hours
- P1 fixes: 3-4 hours
- P2 fixes: 2-3 hours
- Total: 7-10 hours for complete hardening

**Recommendation:** Fix P0 issues immediately. They can cause subtle bugs in production.
