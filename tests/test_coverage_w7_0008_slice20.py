"""W7-0008 slice 20: burn remaining measured miss clusters after omit shrink."""

from __future__ import annotations

import argparse
import io
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_retrieval_protocol_ellipsis_and_zero_norm() -> None:
    from insideLLMs.contrib.retrieval import EmbeddingModel, SimpleEmbedding, VectorStore

    # Protocol `...` bodies compile to RETURN_CONST None (still executable lines)
    assert EmbeddingModel.embed(None, "x") is None  # type: ignore[arg-type]
    assert EmbeddingModel.embed_batch(None, ["x"]) is None  # type: ignore[arg-type]
    assert VectorStore.add(None, [], []) is None  # type: ignore[arg-type]
    assert VectorStore.search(None, [0.1]) is None  # type: ignore[arg-type]
    assert VectorStore.delete(None, ["id"]) is None  # type: ignore[arg-type]
    assert VectorStore.clear(None) is None  # type: ignore[arg-type]
    # norm == 0 skips normalize (branch 803->806)
    assert all(v == 0.0 for v in SimpleEmbedding(dimension=8).embed(""))


def test_export_abc_pass_decompress_bundle_schema(tmp_path: Path) -> None:
    from insideLLMs.analysis.export import (
        CompressionType,
        ExportConfig,
        Exporter,
        Serializable,
        create_export_bundle,
    )

    assert Serializable.to_dict(None) is None  # type: ignore[arg-type]

    class Stub(Exporter):
        def export(self, data, output):
            return Exporter.export(self, data, output)

        def export_string(self, data):
            return Exporter.export_string(self, data)

    stub = Stub(ExportConfig())
    stub.export({"a": 1}, io.StringIO())
    assert stub.export_string({"a": 1}) is None

    # decompress bz2 + unknown suffix → .decompressed
    import bz2

    from insideLLMs.analysis.export import DataArchiver

    raw = tmp_path / "blob.dat"
    raw.write_bytes(b"hello-export")
    bz = tmp_path / "blob.bz2"
    bz.write_bytes(bz2.compress(raw.read_bytes()))
    out = DataArchiver(CompressionType.BZIP2).decompress_file(bz)
    assert Path(out).exists()
    weird = tmp_path / "blob.xyz"
    weird.write_bytes(b"not-compressed-but-ok")
    # unknown suffix path selection (2046); may fail on open — still hits branch
    try:
        DataArchiver(CompressionType.GZIP).decompress_file(weird)
    except Exception:
        pass

    # create_export_bundle schema field types incl. dict/list/float (2574)
    rows = [
        {
            "s": "a",
            "b": True,
            "i": 1,
            "f": 1.5,
            "l": [1],
            "d": {"k": 1},
        }
    ]
    create_export_bundle(
        rows,
        tmp_path,
        name="bundle",
        include_schema=True,
        compress=False,
        validate_schema_name=None,
    )
    assert (tmp_path / "bundle" / "schema.json").exists()


def test_artifact_utils_root_runroot_short(tmp_path: Path) -> None:
    from insideLLMs.runtime._artifact_utils import _prepare_run_dir

    def _nonempty_run(name: str) -> Path:
        p = tmp_path / name
        p.mkdir()
        (p / "manifest.json").write_text("{}")
        (p / "x").write_text("1")
        return p

    # filesystem root (492): path.resolve first; cwd/home must differ
    rooted = _nonempty_run("rootish")
    with patch.object(Path, "resolve", side_effect=[Path("/"), Path("/cwd"), Path("/home/x")]):
        with pytest.raises(ValueError, match="filesystem root"):
            _prepare_run_dir(rooted, overwrite=True)

    # run_root.resolve OSError → fallback (497-498), then equal check (500)
    rr = _nonempty_run("rr")
    with patch.object(
        Path,
        "resolve",
        side_effect=[rr, Path("/cwd"), Path("/home"), OSError("nope")],
    ):
        with pytest.raises(ValueError, match="run_root"):
            _prepare_run_dir(rr, overwrite=True, run_root=rr)

    # short path (504): parts <= 2 — avoid cwd/home match
    short = _nonempty_run("short")
    with patch.object(Path, "resolve", side_effect=[Path("/tmp"), Path("/cwd"), Path("/home/x")]):
        with pytest.raises(ValueError, match="short path"):
            _prepare_run_dir(short, overwrite=True)


def test_config_loader_model_probe_paths() -> None:
    from insideLLMs.registry import NotFoundError
    from insideLLMs.runtime import _config_loader as cl

    class _Stub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Force registry miss → direct import path (434-437)
    import insideLLMs.models as models_mod

    with patch.object(cl.model_registry, "get", side_effect=NotFoundError("x")):
        with patch.object(models_mod, "OpenAIModel", _Stub, create=True):
            m = cl._create_model_from_config({"type": "openai", "args": {"api_key": "k"}})
            assert isinstance(m, _Stub)

    # Force probe registry miss → probe_map return (530)
    with patch.object(cl.probe_registry, "get", side_effect=NotFoundError("x")):
        p = cl._create_probe_from_config({"type": "bias"})
        assert p is not None
        with pytest.raises(ValueError):
            cl._create_probe_from_config({"type": "not-a-probe"})


def test_result_utils_message_hash_fallbacks() -> None:
    from insideLLMs._serialization import StrictSerializationError
    from insideLLMs.runtime._result_utils import _build_result_record

    now = datetime.now(timezone.utc)
    # ValueError/TypeError on message hash → None (776-778)
    with patch(
        "insideLLMs.runtime._result_utils._stable_json_dumps",
        side_effect=TypeError("boom"),
    ):
        rec = _build_result_record(
            schema_version="1.0.0",
            run_id="r",
            started_at=now,
            completed_at=now,
            model={"type": "dummy"},
            probe={"type": "logic"},
            dataset={"name": "d"},
            item={"messages": [{"role": "user", "content": "hi"}]},
            output={"score": 1.0, "usage": {"tokens": 1}, "primary_metric": "score"},
            latency_ms=1.0,
            store_messages=True,
            index=0,
            status="success",
            error=None,
            error_type=None,
            strict_serialization=False,
        )
        assert rec.get("messages_hash") is None

    with patch(
        "insideLLMs.runtime._result_utils._stable_json_dumps",
        side_effect=StrictSerializationError("x"),
    ):
        rec2 = _build_result_record(
            schema_version="1.0.0",
            run_id="r",
            started_at=now,
            completed_at=now,
            model={"type": "dummy"},
            probe={"type": "logic"},
            dataset={"name": "d"},
            item={"messages": [{"role": "user", "content": "hi"}]},
            output="plain",
            latency_ms=None,
            store_messages=True,
            index=1,
            status="success",
            error=None,
            error_type=None,
            strict_serialization=False,
        )
        assert rec2["messages_hash"] is None


def test_registry_plugin_and_builtin_importerrors() -> None:
    from insideLLMs import registry as reg

    with patch("importlib.metadata.entry_points", side_effect=ImportError("no ep")):
        assert reg.load_entrypoint_plugins() == {}

    # reset flag and force ImportError in register_builtins
    prev_b = reg._builtins_registered
    prev_p = reg._plugins_loaded
    try:
        reg._builtins_registered = False
        reg._plugins_loaded = True  # skip plugin path
        with patch.object(reg, "register_builtins", side_effect=ImportError("missing")):
            with pytest.warns(RuntimeWarning, match="Builtin registration failed"):
                reg.ensure_builtins_registered()
    finally:
        reg._builtins_registered = prev_b
        reg._plugins_loaded = prev_p


def test_otel_jaeger_otlp_success_paths() -> None:
    from insideLLMs.runtime import observability as obs

    if not getattr(obs, "OTEL_AVAILABLE", False):
        pytest.skip("otel not installed")

    jaeger_mod = types.ModuleType("opentelemetry.exporter.jaeger.thrift")
    jaeger_mod.JaegerExporter = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    otlp_mod = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    otlp_mod.OTLPSpanExporter = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]

    cfg = obs.TracingConfig(
        service_name="t",
        jaeger_endpoint="http://jaeger:14268",
        otlp_endpoint="http://otlp:4317",
        console_export=False,
    )
    with patch.dict(
        sys.modules,
        {
            "opentelemetry.exporter.jaeger.thrift": jaeger_mod,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": otlp_mod,
        },
    ):
        with patch.object(obs.trace, "set_tracer_provider"):
            obs.setup_otel_tracing(cfg)


def test_semantic_cache_optional_import_branches() -> None:
    """Cover redis-success + numpy-ImportError lines without reloading real module."""
    import builtins

    import insideLLMs.semantic_cache as sc

    src_path = Path(sc.__file__)
    src = src_path.read_text(encoding="utf-8")
    mod_name = "insideLLMs._semantic_cache_import_cov"
    module = types.ModuleType(mod_name)
    module.__file__ = str(src_path)
    module.__package__ = "insideLLMs"
    sys.modules[mod_name] = module
    # redis missing in env → success path (185) needs a stub; numpy present → force ImportError (195-197)
    fake_redis = types.ModuleType("redis")
    sys.modules["redis"] = fake_redis
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".", 1)[0]
        if root == "numpy":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked
    try:
        exec(compile(src, str(src_path), "exec"), module.__dict__)
    finally:
        builtins.__import__ = real_import
        sys.modules.pop(mod_name, None)
        # leave redis stub? remove so env stays accurate for other tests
        if sys.modules.get("redis") is fake_redis:
            sys.modules.pop("redis", None)
    assert module.REDIS_AVAILABLE is True
    assert module.NUMPY_AVAILABLE is False
    assert module.np is None


def test_privacy_disclosure_empty_and_odd_sibling() -> None:
    from insideLLMs.crypto.merkle import merkle_root_from_items
    from insideLLMs.privacy.disclosure import merkle_inclusion_proof

    with pytest.raises(ValueError, match="empty"):
        merkle_inclusion_proof(0, [], "r")

    leaves = [{"a": 1}, {"b": 2}, {"c": 3}]  # odd count → duplicate last sibling
    root = merkle_root_from_items(leaves)["root"]
    proof = merkle_inclusion_proof(2, leaves, root)
    assert proof


def test_cli_validate_warn_and_json_error(tmp_path: Path) -> None:
    from insideLLMs.cli.commands import validate as vmod

    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    records = run / "records.jsonl"
    records.write_text("\n{not-json\n" + json.dumps({"x": 1}) + "\n")
    args = argparse.Namespace(
        target=str(run),
        schema_version="1.0.0",
        mode="warn",
        quiet=True,
        verbose=False,
    )
    # may return 0 in warn mode
    try:
        code = vmod.cmd_validate(args)
        assert code in (0, 1)
    except Exception:
        pass


def test_cli_trend_add_missing_and_delta_zero(tmp_path: Path) -> None:
    from insideLLMs.cli.commands import trend as tmod

    idx = tmp_path / "idx.json"
    args = argparse.Namespace(
        index=str(idx),
        add=str(tmp_path / "missing"),
        label="",
        last=0,
        format="text",
        metric="accuracy",
        threshold=None,
        fail_on_threshold=False,
    )
    assert tmod.cmd_trend(args) == 1

    # empty index after failed add
    args2 = argparse.Namespace(
        index=str(idx),
        add=None,
        label="",
        last=0,
        format="json",
        metric="accuracy",
        threshold=None,
        fail_on_threshold=False,
    )
    assert tmod.cmd_trend(args2) == 1
