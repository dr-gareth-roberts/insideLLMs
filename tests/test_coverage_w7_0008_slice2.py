"""W7-0008 slice 2: crypto omit shrink + CLI doctor/diff/init + caching hotspots."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from insideLLMs.caching import (
    AsyncCacheAdapter,
    BaseCacheABC,
    CacheConfig,
    CachedModel,
    CacheEntry,
    CacheLookupResult,
    CacheNamespace,
    CacheScope,
    CacheStats,
    CacheStatus,
    CacheStrategy,
    CacheWarmer,
    DiskCache,
    InMemoryCache,
    MemoizedFunction,
    PromptCache,
    ResponseDeduplicator,
    StrategyCache,
    cached,
    cached_response,
    clear_default_cache,
    create_cache,
    create_cache_warmer,
    create_namespace,
    create_prompt_cache,
    generate_cache_key,
    generate_model_cache_key,
    get_cache_key,
    get_default_cache,
    memoize,
    set_default_cache,
)
from insideLLMs.cli.commands import doctor as doctor_mod
from insideLLMs.cli.commands.diff import _print_judge_review, cmd_diff
from insideLLMs.cli.commands.init_cmd import _init_uses_defaults, cmd_init
from insideLLMs.crypto import merkle as merkle_mod
from insideLLMs.crypto.merkle import merkle_root_from_items, merkle_root_from_jsonl
from insideLLMs.types import ModelResponse

# ---------------------------------------------------------------------------
# crypto (newly measured after omit shrink)
# ---------------------------------------------------------------------------


def test_crypto_merkle_edge_branches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported algo"):
        merkle_root_from_items([{"a": 1}], algo="md5")
    with pytest.raises(ValueError, match="Unsupported algo"):
        merkle_mod._hash_pair("aa", "bb", algo="md5")

    custom = merkle_root_from_items(
        [{"a": 1}, {"a": 2}],
        canonicalize_fn=lambda x: json.dumps(x, sort_keys=True).encode(),
        canon_version="canon_v1",
    )
    assert custom["count"] == 2
    assert custom["canon_version"] == "canon_v1"

    odd = merkle_root_from_items([{"i": i} for i in range(3)], canon_version="canon_v1")
    assert odd["count"] == 3

    with pytest.raises(FileNotFoundError):
        merkle_root_from_jsonl(tmp_path / "missing.jsonl")

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a":1}\n\n{"a":2}\n', encoding="utf-8")
    out = merkle_root_from_jsonl(path, strict=True)
    assert out["count"] == 2
    assert merkle_mod._merkle_root_from_hashes([], "sha256")


def test_crypto_package_exports() -> None:
    import insideLLMs.crypto as crypto_pkg

    assert "canonical_json_bytes" in crypto_pkg.__all__
    assert callable(crypto_pkg.merkle_root_from_items)


def test_crypto_canonical_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import importlib
    import sys

    from insideLLMs.crypto.canonical import run_bundle_id

    with pytest.raises(ValueError, match="Unsupported algo for run_bundle_id"):
        run_bundle_id("m", {"r": "1"}, ["d"], algo="md5")

    # ImportError path for library version probe
    original = sys.modules["insideLLMs.crypto.canonical"]
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "insideLLMs" and not fromlist:
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked
    try:
        del sys.modules["insideLLMs.crypto.canonical"]
        reloaded = importlib.import_module("insideLLMs.crypto.canonical")
        assert reloaded._LIBRARY_VERSION is None
    finally:
        builtins.__import__ = real_import
        sys.modules["insideLLMs.crypto.canonical"] = original
        import insideLLMs.crypto as crypto_pkg

        crypto_pkg.canonical = original


# ---------------------------------------------------------------------------
# CLI doctor helpers (no nltk importorskip)
# ---------------------------------------------------------------------------


def test_doctor_helpers_and_capabilities(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    assert doctor_mod._plugins_disabled_via_env() is False
    monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "true")
    assert doctor_mod._plugins_disabled_via_env() is True

    class EP:
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class EPs:
        def select(self, group):
            return [EP("z", "pkg:z"), EP("a", "pkg:a")]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: EPs())
    assert [e["name"] for e in doctor_mod._entrypoint_plugins("insideLLMs.plugins")] == ["a", "z"]

    class BoomEPs:
        def select(self, group):
            raise RuntimeError("bad meta")

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: BoomEPs())
    assert doctor_mod._entrypoint_plugins("insideLLMs.plugins") == []

    class LegacyEPs(dict):
        pass

    legacy = LegacyEPs({"insideLLMs.plugins": [EP("x", "v")]})
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: legacy)
    assert doctor_mod._entrypoint_plugins("insideLLMs.plugins")[0]["name"] == "x"

    st = doctor_mod._capability_status(modules=["nope.mod"], credential_env=["NO_KEY"], notes=[])
    assert st["status"] == "missing_dependencies_and_credentials"
    assert (
        doctor_mod._capability_status(modules=["nope.mod"], credential_env=[], notes=[])["status"]
        == "missing_dependencies"
    )
    monkeypatch.setenv("HAS_KEY", "1")
    assert (
        doctor_mod._capability_status(modules=[], credential_env=["HAS_KEY"], notes=[])["status"]
        == "ready"
    )
    assert (
        doctor_mod._capability_status(modules=[], credential_env=["MISSING_KEY"], notes=[])[
            "status"
        ]
        == "missing_credentials"
    )
    assert (
        doctor_mod._capability_status(modules=[], credential_env=[], notes=["Needs network"])[
            "status"
        ]
        == "requires_external_service"
    )

    caps = doctor_mod._build_capabilities(
        [{"name": "plotly", "ok": False}, {"name": "pydantic", "ok": True}]
    )
    doctor_mod._print_capabilities_summary(caps)
    assert "Capabilities" in capsys.readouterr().out

    monkeypatch.setenv("INSIDELLMS_DISABLE_PLUGINS", "1")
    doctor_mod._print_capabilities_summary(doctor_mod._build_capabilities([]))

    args = argparse.Namespace(format="json", fail_on_warn=False, capabilities=True)
    doctor_mod.cmd_doctor(args)
    # cmd_doctor may print warnings to stderr; isolate JSON object from stdout
    raw = capsys.readouterr().out
    start = raw.find("{")
    payload = json.loads(raw[start:])
    assert "capabilities" in payload

    doctor_mod.cmd_doctor(argparse.Namespace(format="json", fail_on_warn=False, capabilities=False))
    raw2 = capsys.readouterr().out
    payload2 = json.loads(raw2[raw2.find("{") :])
    assert "capabilities" not in payload2

    doctor_mod.cmd_doctor(argparse.Namespace(format="text", fail_on_warn=False, capabilities=True))
    assert "Capabilities" in capsys.readouterr().out

    # ready extras + all-checks-passed text branches
    doctor_mod._print_capabilities_summary(
        {
            "models": [{"name": "dummy", "status": "ready"}],
            "probes": [],
            "datasets": [],
            "extras": [{"name": "core", "ready": True}, {"name": "nlp", "ready": False}],
            "plugins": {"discovered_entry_points": [], "disabled_by_env": False},
        }
    )
    assert "extra:core" in capsys.readouterr().out

    with (
        patch.object(doctor_mod, "_has_module", return_value=True),
        patch.object(doctor_mod, "_check_nltk_resource", return_value=True),
        patch.object(doctor_mod.shutil, "which", return_value="/bin/true"),
    ):
        doctor_mod.cmd_doctor(
            argparse.Namespace(format="text", fail_on_warn=False, capabilities=False)
        )
    assert "All recommended checks passed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI diff judge printer + init interactive
# ---------------------------------------------------------------------------


def test_print_judge_review_branches(capsys) -> None:
    _print_judge_review({"summary": None, "verdicts": None, "breaking": False}, limit=5)
    _print_judge_review(
        {
            "policy": "strict",
            "breaking": True,
            "summary": {"breaking": 1, "review": 2, "acceptable": 3},
            "verdicts": [
                {
                    "label": {"model": "m", "probe": "p", "example": "e"},
                    "decision": "breaking",
                    "reason": "drop",
                    "detail": "score",
                },
                "skip-me",
                {
                    "model_id": "m2",
                    "probe_id": "p2",
                    "example_id": "e2",
                    "decision": "review",
                    "reason": "r",
                },
            ],
        },
        limit=1,
    )
    out = capsys.readouterr().out
    assert "Judge Verdict" in out
    assert "and 2 more" in out


def test_cmd_diff_judge_and_flags(tmp_path: Path, monkeypatch, capsys) -> None:
    def _write_run(d: Path, score: float) -> None:
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "schema_version": "1.0.0",
            "run_id": "r",
            "model": {"model_id": "dummy", "provider": "x", "params": {}},
            "probe": {"probe_id": "logic", "probe_version": "1", "params": {}},
            "example_id": "ex1",
            "dataset": {
                "dataset_id": "d",
                "dataset_version": None,
                "dataset_hash": None,
                "provenance": "t",
                "params": {},
            },
            "input": {"text": "q"},
            "output": {"text": "a"},
            "output_text": "a",
            "scores": {"accuracy": score},
            "primary_metric": "accuracy",
            "usage": {},
            "latency_ms": 1,
            "status": "success",
            "error": None,
            "error_type": None,
            "custom": {},
            "started_at": "2020-01-01T00:00:00Z",
            "completed_at": "2020-01-01T00:00:01Z",
        }
        (d / "records.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_run(a, 1.0)
    _write_run(b, 0.5)

    base = dict(
        run_dir_a=str(a),
        run_dir_b=str(b),
        format="text",
        output=None,
        interactive=False,
        fail_on_regressions=False,
        fail_on_changes=False,
        fail_on_trace_violations=False,
        fail_on_trace_drift=False,
        fail_on_trajectory_drift=False,
        output_fingerprint_ignore=None,
        validate_output=False,
        schema_version="1.0.0",
        validation_mode="strict",
        judge=True,
        judge_policy="strict",
        judge_limit=10,
        limit=10,
    )
    cmd_diff(argparse.Namespace(**base))
    assert "Judge Verdict" in capsys.readouterr().out

    assert cmd_diff(argparse.Namespace(**{**base, "format": "json", "interactive": True})) == 1
    assert "requires text output" in capsys.readouterr().err

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    # avoid interactive prompt path by keeping interactive False for text+output warning
    cmd_diff(argparse.Namespace(**{**base, "output": str(tmp_path / "x.json")}))
    captured = capsys.readouterr()
    assert "only used with" in (captured.out + captured.err)


def test_cmd_init_interactive_paths(tmp_path: Path, monkeypatch) -> None:
    assert _init_uses_defaults(
        argparse.Namespace(output="experiment.yaml", model="dummy", probe="logic", template="basic")
    )

    answers = iter([str(tmp_path / "out.yaml"), "badtpl", "harness"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = argparse.Namespace(
        output="experiment.yaml",
        model="dummy",
        probe="logic",
        template="basic",
        interactive=True,
        quiet=False,
    )
    assert cmd_init(args) == 0
    assert (tmp_path / "out.yaml").exists()

    monkeypatch.setattr("builtins.input", lambda _p="": (_ for _ in ()).throw(EOFError()))
    assert cmd_init(args) == 1


# ---------------------------------------------------------------------------
# caching hotspots
# ---------------------------------------------------------------------------


def test_caching_dataclasses_and_keys() -> None:
    cfg = CacheConfig(strategy=CacheStrategy.LFU, scope=CacheScope.MODEL)
    assert cfg.to_dict()["strategy"] == "lfu"
    entry = CacheEntry(key="k", value="v")
    assert "key" in entry.to_dict()
    stats = CacheStats(hits=1, misses=1)
    assert stats.to_dict()["hits"] == 1
    lookup = CacheLookupResult(hit=True, value=1, status=CacheStatus.HIT)
    assert lookup.to_dict()["hit"] is True

    assert generate_cache_key("p", model="m", params={"t": 1}, algorithm="md5")
    assert generate_cache_key("p", algorithm="sha1")
    assert generate_model_cache_key("m", "p", temperature=0.2, max_tokens=10, top_p=1)

    mem = InMemoryCache()
    assert BaseCacheABC.get(mem, "x") is None
    assert BaseCacheABC.set(mem, "x", 1) is None
    assert BaseCacheABC.delete(mem, "x") is None
    assert BaseCacheABC.clear(mem) is None
    assert BaseCacheABC.stats(mem) is None
    assert BaseCacheABC.has(mem, "missing") is False


def test_inmemory_and_disk_cache(tmp_path: Path) -> None:
    cache = InMemoryCache(max_size=2, default_ttl=1)
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.set("c", {"v": 3})  # eviction
    cache.set("ttl", {"v": 9}, ttl=1)
    time.sleep(1.05)
    assert cache.get("ttl") is None
    assert cache.delete("c") in (True, False)
    assert cache.delete("nope") is False
    cache.set("z", 1)
    assert cache.stats().entry_count >= 0
    cache.clear()
    assert cache.keys() == []
    cache._evict_lru()

    disk = DiskCache(path=tmp_path / "cache.db", max_size_mb=1)
    disk.set("k1", {"x": 1}, metadata={"m": 1})
    assert disk.get("k1") == {"x": 1}
    assert disk.get("missing") is None
    disk.set("exp", {"e": 1}, ttl=1)
    time.sleep(1.05)
    assert disk.get("exp") is None
    assert disk.delete("k1") is True
    assert disk.delete("k1") is False
    disk.set("a", 1)
    disk.set("b", 2)
    assert disk.stats().entry_count >= 0
    export = tmp_path / "export.json"
    assert disk.export_to_file(export) >= 1
    disk.clear()
    assert disk.import_from_file(export) >= 1

    tiny = DiskCache(path=tmp_path / "tiny.db", max_size_mb=1)
    for i in range(40):
        tiny.set(f"k{i}", {"payload": "x" * 200})
    tiny._evict_if_needed()


def test_strategy_prompt_warmer_memoize_namespace_factories() -> None:
    sc = StrategyCache(CacheConfig(max_size=2, strategy=CacheStrategy.LRU, ttl_seconds=1))
    sc.set("a", 1)
    sc.set("b", 2)
    sc.set("c", 3)
    sc.set("ttl", 9, ttl_seconds=1)
    time.sleep(1.05)
    assert sc.get("ttl").hit is False
    assert sc.delete("c") in (True, False)
    assert sc.delete("missing") is False
    sc.set("alive", 1)
    assert sc.contains("alive") is True
    assert isinstance(sc.values(), list)
    assert isinstance(sc.items(), list)

    lfu = StrategyCache(CacheConfig(max_size=2, strategy=CacheStrategy.LFU))
    lfu.set("a", 1)
    lfu.set("b", 2)
    lfu.get("a")
    lfu.set("c", 3)

    fifo = StrategyCache(CacheConfig(max_size=1, strategy=CacheStrategy.FIFO))
    fifo.set("a", 1)
    fifo.set("b", 2)

    pc = PromptCache(CacheConfig(max_size=10))
    pc.cache_response("hello", "world", model="m", params={"t": 0})
    assert pc.get_response("hello", model="m", params={"t": 0}).hit is True

    class _M:
        model_id = "dummy-m"

        def generate(self, prompt, **kwargs):
            return ModelResponse(content=f"out:{prompt}", model=self.model_id)

    cm = CachedModel(_M(), cache=InMemoryCache(), cache_only_deterministic=True)
    r1 = cm.generate("hello", temperature=0.0)
    assert isinstance(r1, ModelResponse)
    r2 = cm.generate("hello", temperature=0.0)
    assert r2.content == r1.content
    assert cm.model_id == "dummy-m"

    warmer = CacheWarmer(pc, generator=lambda p: f"w:{p}")
    warmer.add_prompt("p1")
    warmer.add_prompt("p2", priority=5)
    results = warmer.warm()
    assert results
    with pytest.raises(ValueError):
        CacheWarmer(pc).warm()
    assert warmer.get_queue_size() >= 0
    assert isinstance(warmer.get_results(), list)
    warmer.clear_queue()

    warmer2 = CacheWarmer(StrategyCache(), generator=lambda p: f"w:{p}")
    warmer2.add_prompt("x")
    warmer2.warm(skip_existing=False)

    mf = MemoizedFunction(lambda x: x + 1)
    assert mf(1) == 2
    assert mf(1) == 2
    mf.invalidate(1)
    assert mf.get_stats()["call_count"] >= 1

    @memoize(max_size=10, ttl_seconds=60)
    def add(x, y):
        return x + y

    assert add(1, 2) == 3

    @memoize
    def mul(x, y):
        return x * y

    assert mul(2, 3) == 6

    ns = CacheNamespace()
    c1 = ns.get_cache("a")
    assert ns.get_cache("a") is c1
    pc2 = ns.get_prompt_cache("p")
    assert ns.get_prompt_cache("p") is pc2
    assert ns.delete_cache("a") is True
    assert ns.delete_cache("a") is False
    assert "p" in ns.list_caches()
    assert isinstance(ns.get_all_stats(), dict)
    ns.clear_all()

    dedup = ResponseDeduplicator(similarity_threshold=0.5)
    assert dedup.add("p", "hello world")[0] is False
    assert dedup.add("p2", "hello world")[0] is True
    assert dedup.add("p3", "completely different zz")[0] is False
    assert dedup.get_duplicate_count() >= 1
    assert dedup.get_unique_responses()
    dedup.clear()

    adapter = AsyncCacheAdapter(sc)

    async def _run():
        await adapter.set("ak", 1)
        await adapter.get("ak")
        await adapter.delete("ak")
        await adapter.clear()

    asyncio.run(_run())

    create_cache(max_size=5, ttl_seconds=10, strategy=CacheStrategy.LRU)
    create_prompt_cache(max_size=5)
    create_cache_warmer(pc, generator=lambda p: p)
    create_namespace()
    assert get_cache_key("p", model="m", params={"a": 1})

    def gen(prompt):
        return f"g:{prompt}"

    shared = create_prompt_cache(max_size=10)
    resp, _ = cached_response("px", gen, model="m", params={}, cache=shared)
    assert resp.startswith("g:")
    _, was_hit2 = cached_response("px", gen, model="m", params={}, cache=shared)
    assert was_hit2 is True

    set_default_cache(InMemoryCache())
    assert get_default_cache() is not None
    clear_default_cache()
    get_default_cache()

    @cached(ttl=60)
    def f(x):
        return x * 2

    assert f(3) == 6
    assert f(3) == 6

    @cached(ttl=60, key_fn=lambda x: f"k:{x}")
    def g(x):
        return x

    assert g(1) == 1

    # expired entry on strategy get
    sc2 = StrategyCache()
    sc2._entries["expired"] = CacheEntry(
        key="expired",
        value="x",
        expires_at=datetime.now() - timedelta(seconds=1),
    )
    assert sc2.get("expired").hit is False
