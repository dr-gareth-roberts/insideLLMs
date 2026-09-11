"""Regression tests for production-quality audit wave 7 fixes."""

from __future__ import annotations

import importlib
import json
import sys
import warnings
from pathlib import Path

import pytest


# W7-0001 — the SafetyHallucinationIndicatorDetector percentage pattern must
# match normally-formatted percentages. The old pattern r"\b\d+(?:\.\d+)?%\b"
# ended in \b immediately after '%'; since '%' is a non-word char, that word
# boundary required a word char right after '%', so "95%" (followed by space/
# punctuation/end) never matched and percentages were silently ignored.
def test_percentage_counts_as_specific_claim():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    detector = SafetyHallucinationIndicatorDetector()
    # Text whose ONLY specific claim is a percentage — before the fix this
    # returned specific_claims == [] and has_specific_claims is False.
    result = detector.analyze("Roughly 95% of users prefer this option.")

    assert "95%" in result["specific_claims"]
    assert result["indicators"]["has_specific_claims"] is True


# W7-0001 — the raw pattern must match the common percentage forms (trailing
# space, punctuation, end-of-string) that the stray \b previously excluded.
def test_specific_claim_percentage_pattern_matches_common_forms():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    pattern = SafetyHallucinationIndicatorDetector.SPECIFIC_CLAIM_PATTERNS[0]
    assert pattern.findall("95% of users") == ["95%"]
    assert pattern.findall("growth of 12.5%.") == ["12.5%"]
    assert pattern.findall("hit 100%!") == ["100%"]


# W7-0001 — the module's own docstring example
# ("Studies show 75% of experts agree this is definitely true.") documents
# has_specific_claims == True; that intent must hold in behaviour.
def test_docstring_percentage_example_holds():
    from insideLLMs.safety import SafetyHallucinationIndicatorDetector

    detector = SafetyHallucinationIndicatorDetector()
    result = detector.analyze("Studies show 75% of experts agree this is definitely true.")
    assert result["indicators"]["has_specific_claims"] is True


# W7-0009 — a blank cell in a markdown table row must not shift the following
# values left. The old parser filtered empty cells before zip(headers, cells),
# so a blank cell silently moved every later value into the wrong column and
# dropped the last column.
def test_markdown_table_blank_cell_keeps_columns_aligned():
    from insideLLMs.structured_extraction import TableExtractor

    text = "| Name | Age | City |\n|------|-----|------|\n| Bob  |     | NYC  |\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    # 'NYC' is a City value and must stay in City, not slide into Age.
    assert rows == [{"Name": "Bob", "Age": "", "City": "NYC"}]


# W7-0009 — rows without outer border pipes must still parse correctly.
def test_markdown_table_without_outer_pipes():
    from insideLLMs.structured_extraction import TableExtractor

    text = "Name | Age | City\nBob | 30 | NYC\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    assert rows == [{"Name": "Bob", "Age": "30", "City": "NYC"}]


# W7-0009 — a border-less row ending in a pipe keeps its trailing empty cell
# instead of dropping the last column (raised in review of the table fix).
def test_markdown_table_trailing_blank_cell_on_borderless_row():
    from insideLLMs.structured_extraction import TableExtractor

    text = "Name | Age | City\nBob | 30 |\n"
    rows = TableExtractor().extract(text).extracted_data["rows"]
    assert rows == [{"Name": "Bob", "Age": "30", "City": ""}]


# W7-0006 — a non-positive max_size must fail fast at construction instead of
# hanging forever. The eviction loop `while len(cache) >= max_size` could never
# make progress when max_size <= 0 (an empty cache is already >= 0 and there is
# nothing to evict), so `.set()` spun indefinitely.
def test_inmemory_cache_rejects_nonpositive_max_size():
    import pytest

    from insideLLMs.caching import InMemoryCache

    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_size"):
            InMemoryCache(max_size=bad)
    # A valid cache still constructs and stores.
    cache = InMemoryCache(max_size=2)
    cache.set("a", 1)
    assert cache.get("a") == 1


# W7-0006 — the same guard must protect StrategyCache (max_size via CacheConfig).
def test_strategy_cache_rejects_nonpositive_max_size():
    import pytest

    from insideLLMs.caching import StrategyCache

    with pytest.raises(ValueError, match="max_size"):
        StrategyCache(max_size=0)


# W7-0008 — ContentDetector.check re-scanned the whole rolling buffer every call
# and re-appended matches from earlier chunks, double-counting them. A match must
# be reported exactly once, while patterns spanning two chunks still complete.
def test_content_detector_does_not_double_count_across_chunks():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("num", r"\d+")
    first = detector.check("First: 123")
    second = detector.check(" Second: 456")

    assert [d["match"] for d in first] == ["123"]
    assert [d["match"] for d in second] == ["456"]  # 123 must not reappear
    assert [d["match"] for d in detector.get_all_detections()] == ["123", "456"]


# W7-0008 — a pattern split across two check() calls must still be detected once.
def test_content_detector_matches_pattern_spanning_chunks():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("greeting", r"hello")
    assert detector.check("hel") == []
    assert [d["match"] for d in detector.check("lo world")] == ["hello"]


# W7-0008 — redefining a pattern under an existing name must reset that name's
# scan offset, or matches near the buffer start are wrongly skipped as already
# reported (raised in review of the streaming fix).
def test_content_detector_resets_scan_pos_when_pattern_redefined():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("p", r"\d+")
    assert [d["match"] for d in detector.check("abc 123")] == ["123"]  # advances offset
    detector.add_pattern("p", r"[a-z]+")  # same name, different pattern
    # 'abc' sits at the buffer start (before the old offset); it must still match.
    assert [d["match"] for d in detector.check("")] == ["abc"]


# W7-0008 — clear() must also reset the per-pattern scan offsets, or a fresh
# check() after clear() skips matches at the start of the new buffer (raised in
# review of the streaming fix).
def test_content_detector_clear_resets_scan_pos():
    from insideLLMs.streaming import ContentDetector

    detector = ContentDetector()
    detector.add_pattern("num", r"\d+")
    detector.check("first 123")  # advances the scan offset past index 0
    detector.clear()
    # After clear the buffer restarts at 0; the new match must be found.
    assert [d["match"] for d in detector.check("456")] == ["456"]


# W7-0010 — async_timeout must raise asyncio.TimeoutError (as documented), not
# asyncio.CancelledError. The old implementation called task.cancel() without
# translating the resulting CancelledError, so callers that catch TimeoutError
# silently missed the timeout and an external cancel was indistinguishable from
# an internal one.
def test_async_timeout_raises_TimeoutError_not_CancelledError():
    import asyncio

    from insideLLMs.async_utils import async_timeout

    async def _run() -> str:
        try:
            async with async_timeout(0.05):
                await asyncio.sleep(5)
        # The builtin, not asyncio.TimeoutError: before Python 3.11 those are
        # distinct classes, and async_timeout normalizes to the builtin so every
        # layer (including RetryConfig.retryable_exceptions) sees one type.
        except TimeoutError:
            return "TimeoutError"
        except asyncio.CancelledError:
            return "CancelledError"
        return "no_exception"

    assert asyncio.run(_run()) == "TimeoutError"


# W7-0010 — an external task cancellation must still propagate as CancelledError,
# not be swallowed or mistakenly converted to TimeoutError.
def test_async_timeout_external_cancel_propagates_CancelledError():
    import asyncio

    from insideLLMs.async_utils import async_timeout

    async def _inner() -> str:
        try:
            async with async_timeout(10.0):  # long — will not fire
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            return "CancelledError"
        except asyncio.TimeoutError:
            return "TimeoutError"
        return "no_exception"

    async def _run() -> str:
        task = asyncio.create_task(_inner())
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return "CancelledError_from_task"

    assert asyncio.run(_run()) == "CancelledError"


# W7-0011 — the DSSE Pre-Authentication Encoding must use the spec version tag
# "DSSEv1" (lowercase v). The old "DSSEV1" produced signatures no spec-compliant
# verifier (cosign, in-toto) would accept.
def test_dsse_pae_uses_spec_version_tag():
    from insideLLMs.attestations.dsse import pae

    out = pae("application/vnd.in-toto+json", b"hello")
    # Full DSSE PAE: "DSSEv1" SP LEN(type) SP type SP LEN(body) SP body
    assert out == b"DSSEv1 28 application/vnd.in-toto+json 5 hello"
    assert not out.startswith(b"DSSEV1 ")


# ---------------------------------------------------------------------------
# W7-0072 / W7-0002 — visualization shim sunset. The conflict between the
# parallel branch's "deprecate, remove at v2.0.0" policy and W7-0002's
# "indefinite support" stance is settled in favour of the former: the shim
# emits a DeprecationWarning, and CHANGELOG, docs/IMPORT_PATHS.md and the shim
# docstring all name v2.0.0 as the removal release.
# ---------------------------------------------------------------------------
def test_visualization_shim_sunset_documented_consistently():
    """IMPORT_PATHS, CHANGELOG, and shim docstring must agree on v2.0.0 removal."""
    repo_root = Path(__file__).resolve().parents[1]
    import_paths = (repo_root / "docs" / "IMPORT_PATHS.md").read_text(encoding="utf-8")
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    shim_doc = (repo_root / "insideLLMs" / "visualization.py").read_text(encoding="utf-8")

    assert "Deprecated; removal in v2.0.0" in import_paths
    assert "removed in v2.0.0" in changelog
    assert "removed in v2.0.0" in shim_doc
    assert "DeprecationWarning" in shim_doc
    assert "indefinitely" not in shim_doc.lower()
    assert "is not deprecated" not in shim_doc


def test_changelog_migration_timeline_uses_the_real_package_version():
    """W7-0002 - the timeline named a fictional v1.1.0 as the current release."""
    from insideLLMs import __version__

    repo_root = Path(__file__).resolve().parents[1]
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    section = changelog.split("### Visualization Module")[1].split("\n## ")[0]

    assert f"v{__version__} (current)" in section
    # The project has never shipped a v1.x; the timeline must not imply otherwise.
    assert "v1.1.0" not in section
    assert "v1.2.0" not in section


def test_visualization_shim_emits_deprecation_warning_on_import():
    """CHANGELOG migration timeline requires a DeprecationWarning on shim import."""
    sys.modules.pop("insideLLMs.visualization", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mod = importlib.import_module("insideLLMs.visualization")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning when importing insideLLMs.visualization"
    message = str(deprecations[0].message)
    assert "v2.0.0" in message
    assert "insideLLMs.analysis.visualization" in message
    # Shim still aliases the canonical module object.
    canonical = importlib.import_module("insideLLMs.analysis.visualization")
    assert mod is canonical


# W7-0007 — on the async stop_on_error path, run_single used to write a
# status="skipped" placeholder record for every item queued behind the first
# failure. Those items never executed, yet write_ready_records persisted them,
# so records.jsonl depended on whether the run was sync or async and a later
# resume counted them as completed work.
class _FailOnSecondItem:
    """Probe that raises on the second prompt and records what it executed."""

    name = "fail-on-second"

    def __init__(self, fail_on: str = "p1") -> None:
        self.fail_on = fail_on
        self.executed: list[str] = []

    def run(self, _model, item, **_kwargs):
        self.executed.append(item)
        if item == self.fail_on:
            raise ValueError(f"boom on {item}")
        return f"ok:{item}"


_STOP_PROMPTS = ["p0", "p1", "p2", "p3", "p4"]


def _read_records(run_dir: Path) -> list[dict]:
    lines = (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def _run_async_until_stop(probe, run_dir: Path, run_id: str = "stop-run") -> None:
    from insideLLMs.exceptions import RunnerExecutionError
    from insideLLMs.models import DummyModel
    from insideLLMs.runtime.runner import AsyncProbeRunner

    with pytest.raises(RunnerExecutionError):
        await AsyncProbeRunner(DummyModel(), probe).run(
            _STOP_PROMPTS,
            stop_on_error=True,
            concurrency=1,
            emit_run_artifacts=True,
            run_dir=run_dir,
            run_id=run_id,
            overwrite=True,
            return_experiment=False,
            deterministic_artifacts=True,
        )


async def test_async_stop_on_error_writes_no_records_for_unexecuted_items(tmp_path: Path):
    probe = _FailOnSecondItem()
    run_dir = tmp_path / "async-stop"

    await _run_async_until_stop(probe, run_dir)

    assert probe.executed == ["p0", "p1"]
    records = _read_records(run_dir)
    # One record per item that actually ran, and nothing past the failure.
    assert [r["status"] for r in records] == ["success", "error"]
    assert not any(r["status"] == "skipped" for r in records)


async def test_sync_and_async_stop_on_error_produce_identical_records(tmp_path: Path):
    """records.jsonl is Stable and deterministic for identical inputs/config."""
    from insideLLMs.exceptions import RunnerExecutionError
    from insideLLMs.models import DummyModel
    from insideLLMs.runtime.runner import ProbeRunner

    async_dir = tmp_path / "async"
    sync_dir = tmp_path / "sync"

    await _run_async_until_stop(_FailOnSecondItem(), async_dir, run_id="same-run")

    sync_probe = _FailOnSecondItem()
    with pytest.raises(RunnerExecutionError):
        ProbeRunner(DummyModel(), sync_probe).run(
            _STOP_PROMPTS,
            stop_on_error=True,
            emit_run_artifacts=True,
            run_dir=sync_dir,
            run_id="same-run",
            overwrite=True,
            return_experiment=False,
            deterministic_artifacts=True,
        )

    assert sync_probe.executed == ["p0", "p1"]
    assert (async_dir / "records.jsonl").read_bytes() == (sync_dir / "records.jsonl").read_bytes()


async def test_resume_after_async_stop_on_error_reexecutes_remaining_items(tmp_path: Path):
    """Items behind a stop_on_error failure must re-run, not resume as done."""
    from insideLLMs.models import DummyModel
    from insideLLMs.runtime.runner import AsyncProbeRunner

    run_dir = tmp_path / "resume"
    await _run_async_until_stop(_FailOnSecondItem(), run_dir)

    resume_probe = _FailOnSecondItem(fail_on="never-fails")
    results = await AsyncProbeRunner(DummyModel(), resume_probe).run(
        _STOP_PROMPTS,
        concurrency=1,
        resume=True,
        emit_run_artifacts=True,
        run_dir=run_dir,
        run_id="stop-run",
        return_experiment=False,
        deterministic_artifacts=True,
    )

    # p0/p1 are a valid completed prefix; p2-p4 never executed and must re-run.
    assert resume_probe.executed == ["p2", "p3", "p4"]
    assert [r["status"] for r in results] == ["success", "error", "success", "success", "success"]


def test_validate_resume_record_rejects_skipped_placeholder():
    """A 'skipped' record describes an item that never ran; resume must refuse it."""
    from insideLLMs.runtime.runner import _validate_resume_record

    record = {"custom": {"record_index": 0}, "input": "p0", "status": "skipped"}

    with pytest.raises(ValueError, match="never executed"):
        _validate_resume_record(record, expected_index=0, expected_item="p0", run_id=None)

    # The same record with a real status still validates.
    _validate_resume_record(
        {**record, "status": "success"}, expected_index=0, expected_item="p0", run_id=None
    )


# W7-0081 - the batch path had a sync/async divergence of its own: the sync
# runner used to write the failing record and break, while the async runner
# wrote every result run_batch returned, so a stop_on_error run over 8 items
# produced 2 records synchronously and 8 asynchronously. The divergence closed
# in the other direction on rebase: the audit remediation (A07/A13) made the
# sync runner persist every completed batch result too, because run_batch has
# already attempted every item by the time the failure is observed and a
# resume must never repeat attempted work. What this locks is parity.
class _BatchFailOnSecondItem:
    """run_batch returns an error for the second prompt, success for the rest."""

    name = "batch-fail-on-second"

    def run(self, _model, item, **_kwargs):
        if item == "p1":
            raise ValueError("boom")
        return f"ok:{item}"

    def run_batch(self, _model, items, **_kwargs):
        from insideLLMs.probes.base import ProbeResult
        from insideLLMs.types import ResultStatus

        results = []
        for item in items:
            if item == "p1":
                results.append(
                    ProbeResult(
                        input=item,
                        status=ResultStatus.ERROR,
                        error="boom",
                        latency_ms=None,
                        metadata={"error_type": "ValueError"},
                    )
                )
            else:
                results.append(
                    ProbeResult(
                        input=item,
                        output=f"ok:{item}",
                        status=ResultStatus.SUCCESS,
                        latency_ms=None,
                        metadata={},
                    )
                )
        return results


_BATCH_PROMPTS = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]


async def test_batch_stop_on_error_matches_between_sync_and_async(tmp_path: Path):
    """use_probe_batch must persist the same attempted results on both runners."""
    from insideLLMs.exceptions import RunnerExecutionError
    from insideLLMs.models import DummyModel
    from insideLLMs.runtime.runner import AsyncProbeRunner, ProbeRunner

    kwargs = dict(
        stop_on_error=True,
        use_probe_batch=True,
        emit_run_artifacts=True,
        run_id="batch-run",
        overwrite=True,
        return_experiment=False,
        deterministic_artifacts=True,
    )

    async_dir = tmp_path / "async"
    sync_dir = tmp_path / "sync"

    with pytest.raises(RunnerExecutionError):
        await AsyncProbeRunner(DummyModel(), _BatchFailOnSecondItem()).run(
            _BATCH_PROMPTS, run_dir=async_dir, **kwargs
        )
    with pytest.raises(RunnerExecutionError):
        ProbeRunner(DummyModel(), _BatchFailOnSecondItem()).run(
            _BATCH_PROMPTS, run_dir=sync_dir, **kwargs
        )

    # run_batch attempted all eight items, so all eight outcomes are evidence;
    # dropping the six behind the failure would make a resume re-run attempted
    # work. Both runners persist the same eight records, byte for byte.
    assert [r["status"] for r in _read_records(async_dir)] == ["success", "error"] + ["success"] * 6
    assert [r["status"] for r in _read_records(sync_dir)] == ["success", "error"] + ["success"] * 6
    assert (async_dir / "records.jsonl").read_bytes() == (sync_dir / "records.jsonl").read_bytes()
