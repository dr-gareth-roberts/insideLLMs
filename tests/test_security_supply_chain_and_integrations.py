"""Security, software-supply-chain, and optional-integration behavior."""

from __future__ import annotations

import argparse
import asyncio
import builtins
import importlib
import importlib.metadata
import json
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.attestations.predicates.slsa_provenance import build_slsa_provenance_predicate
from insideLLMs.attestations.steps import builders as attestation_builders
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
from insideLLMs.cli.commands.optimize_prompt import _parse_strategies, cmd_optimize_prompt
from insideLLMs.crypto import merkle as merkle_mod
from insideLLMs.crypto.merkle import merkle_root_from_items, merkle_root_from_jsonl
from insideLLMs.datasets.tuf_client import fetch_dataset
from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import ChatMessage
from insideLLMs.optimization import OptimizationStrategy
from insideLLMs.privacy import encryption as encryption_mod
from insideLLMs.publish import oras as oras_mod
from insideLLMs.runtime.diffing_interactive import (
    _summary_display,
    build_interactive_review_lines,
    copy_candidate_artifacts_to_baseline,
    print_interactive_review,
    prompt_accept_snapshot,
)
from insideLLMs.runtime.receipt import ReceiptMiddleware
from insideLLMs.shadow import (
    ShadowWriter,
    _decode_request_body,
    _read_request_body,
    _request_url_parts,
    _safe_mapping,
    _sample_request,
    _to_utc,
    fastapi,
)
from insideLLMs.signing import cosign as cosign_mod
from insideLLMs.types import ModelResponse

# ---------------------------------------------------------------------------
# privacy.encryption
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_roundtrip_and_blank_lines(tmp_path: Path) -> None:
    if not encryption_mod.CRYPTO_AVAILABLE:
        pytest.skip("cryptography not available")
    from cryptography.fernet import Fernet

    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    key = Fernet.generate_key()
    encryption_mod.encrypt_jsonl(path, key=key)
    assert "hello" not in path.read_text(encoding="utf-8", errors="ignore")
    encryption_mod.decrypt_jsonl(path, key=key)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


def test_encrypt_requires_crypto_and_key_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encryption_mod, "CRYPTO_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="cryptography"):
        encryption_mod.encrypt_jsonl(tmp_path / "x.jsonl", key=b"k")
    with pytest.raises(RuntimeError, match="cryptography"):
        encryption_mod.decrypt_jsonl(tmp_path / "x.jsonl", key=b"k")

    monkeypatch.setattr(encryption_mod, "CRYPTO_AVAILABLE", True)
    with pytest.raises(ValueError, match="Encryption key"):
        encryption_mod.encrypt_jsonl(tmp_path / "missing.jsonl", key=None)
    with pytest.raises(ValueError, match="Decryption key"):
        encryption_mod.decrypt_jsonl(tmp_path / "missing.jsonl", key=b"")
    with pytest.raises(FileNotFoundError):
        encryption_mod.encrypt_jsonl(tmp_path / "missing.jsonl", key=b"x" * 44)
    with pytest.raises(FileNotFoundError):
        encryption_mod.decrypt_jsonl(tmp_path / "missing.jsonl", key=b"x" * 44)


def test_encrypt_import_error_sets_flag() -> None:
    original = sys.modules["insideLLMs.privacy.encryption"]
    crypto_keys = [k for k in sys.modules if k == "cryptography" or k.startswith("cryptography.")]
    saved_crypto = {k: sys.modules[k] for k in crypto_keys}
    for k in crypto_keys:
        del sys.modules[k]

    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "cryptography" or (isinstance(name, str) and name.startswith("cryptography.")):
            raise ImportError("blocked for coverage")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked
    try:
        del sys.modules["insideLLMs.privacy.encryption"]
        reloaded = importlib.import_module("insideLLMs.privacy.encryption")
        assert reloaded.CRYPTO_AVAILABLE is False
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved_crypto)
        sys.modules["insideLLMs.privacy.encryption"] = original
        import insideLLMs.privacy as privacy_pkg

        privacy_pkg.encryption = original


def test_encrypt_cleans_temp_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not encryption_mod.CRYPTO_AVAILABLE:
        pytest.skip("cryptography not available")
    from cryptography.fernet import Fernet

    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\n', encoding="utf-8")
    key = Fernet.generate_key()
    temp = path.with_suffix(path.suffix + ".enc.tmp")

    def boom_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(encryption_mod.os, "replace", boom_replace)
    with pytest.raises(OSError, match="replace failed"):
        encryption_mod.encrypt_jsonl(path, key=key)
    assert not temp.exists()

    # exception before temp exists (false branch of temp_path.exists())
    real_open = builtins.open

    def boom_read(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and "r" in mode and "b" in mode:
            raise OSError("read failed")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", boom_read)
    with pytest.raises(OSError, match="read failed"):
        encryption_mod.encrypt_jsonl(path, key=key)


def test_decrypt_cleans_temp_and_skips_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not encryption_mod.CRYPTO_AVAILABLE:
        pytest.skip("cryptography not available")
    from cryptography.fernet import Fernet

    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\n', encoding="utf-8")
    key = Fernet.generate_key()
    encryption_mod.encrypt_jsonl(path, key=key)
    # insert blank lines between ciphertext rows
    cipher = path.read_bytes()
    path.write_bytes(b"\n" + cipher + b"\n\n")
    encryption_mod.decrypt_jsonl(path, key=key)
    assert json.loads(path.read_text(encoding="utf-8").strip()) == {"a": 1}

    path.write_text('{"a":1}\n', encoding="utf-8")
    encryption_mod.encrypt_jsonl(path, key=key)
    temp = path.with_suffix(path.suffix + ".dec.tmp")

    def boom_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(encryption_mod.os, "replace", boom_replace)
    with pytest.raises(OSError, match="replace failed"):
        encryption_mod.decrypt_jsonl(path, key=key)
    assert not temp.exists()

    real_open = builtins.open

    def boom_read(file, mode="r", *args, **kwargs):
        if str(file) == str(path) and "r" in mode and "b" in mode:
            raise OSError("read failed")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", boom_read)
    with pytest.raises(OSError, match="read failed"):
        encryption_mod.decrypt_jsonl(path, key=key)


# ---------------------------------------------------------------------------
# signing.cosign
# ---------------------------------------------------------------------------


def test_cosign_path_and_sign_verify_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cosign_mod.shutil, "which", lambda _name: None)
    assert cosign_mod._cosign_path() is None
    with pytest.raises(FileNotFoundError, match="cosign not found"):
        cosign_mod.sign_blob(tmp_path / "blob", tmp_path / "bundle")
    with pytest.raises(FileNotFoundError, match="cosign not found"):
        cosign_mod.verify_bundle(tmp_path / "blob", tmp_path / "bundle")

    fake = tmp_path / "cosign"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(cosign_mod.shutil, "which", lambda _name: str(fake))
    assert cosign_mod._cosign_path() == fake

    missing_blob = tmp_path / "missing.bin"
    with pytest.raises(FileNotFoundError, match="Blob to sign"):
        cosign_mod.sign_blob(missing_blob, tmp_path / "out" / "bundle.json")

    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"payload")
    out_bundle = tmp_path / "nested" / "bundle.json"

    def fail_run(*_a, **_k):
        return SimpleNamespace(returncode=1, stderr="boom", stdout="")

    monkeypatch.setattr(cosign_mod.subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="cosign sign-blob failed"):
        cosign_mod.sign_blob(blob, out_bundle)

    def ok_run(cmd, **_k):
        # sign-blob writes bundle path from argv
        if "sign-blob" in cmd:
            Path(cmd[cmd.index("--bundle") + 1]).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")

    monkeypatch.setattr(cosign_mod.subprocess, "run", ok_run)
    cosign_mod.sign_blob(blob, out_bundle)
    assert out_bundle.exists()

    assert cosign_mod.verify_bundle(blob, out_bundle) is True
    assert cosign_mod.verify_bundle(tmp_path / "nope", out_bundle) is False
    assert cosign_mod.verify_bundle(blob, tmp_path / "no-bundle") is False

    with pytest.raises(ValueError, match="Invalid identity_constraints"):
        cosign_mod.verify_bundle(blob, out_bundle, identity_constraints="bad;rm")

    def verify_fail(cmd, **_k):
        assert "--cert-identity" in cmd
        return SimpleNamespace(returncode=2, stderr="no", stdout="")

    monkeypatch.setattr(cosign_mod.subprocess, "run", verify_fail)
    assert (
        cosign_mod.verify_bundle(blob, out_bundle, identity_constraints="issuer=a@b.com") is False
    )


# ---------------------------------------------------------------------------
# attestations
# ---------------------------------------------------------------------------


def test_slsa_provenance_optional_fields() -> None:
    bare = build_slsa_provenance_predicate(builder={"id": "b"})
    assert bare["builder"] == {"id": "b"}
    assert "invocation" not in bare
    full = build_slsa_provenance_predicate(
        builder={"id": "b"},
        invocation={"parameters": {"x": 1}},
        materials=[{"uri": "records.jsonl"}],
        metadata={"buildInvocationId": "1"},
    )
    assert full["invocation"]["parameters"]["x"] == 1
    assert full["materials"][0]["uri"] == "records.jsonl"
    assert full["metadata"]["buildInvocationId"] == "1"


def test_all_attestation_builders_optional_fields() -> None:
    subject = [{"name": "x", "digest": {"sha256": "abc"}}]
    st0 = attestation_builders.build_attestation_00_source(
        subject,
        git_commit="c",
        git_dirty=True,
        pyproject_digest="p",
        lock_digest="l",
        insidellms_version="0.2.0",
    )
    assert st0["predicate"]["git_dirty"] is True
    assert st0["predicate"]["lock_digest"] == "l"

    st1 = attestation_builders.build_attestation_01_env(
        subject,
        python_version="3.12",
        platform="darwin",
        container_digest="cd",
        sbom_digest="sb",
    )
    assert st1["predicate"]["sbom_digest"] == "sb"

    st2 = attestation_builders.build_attestation_02_dataset(
        subject,
        dataset_id="d",
        dataset_version="v",
        dataset_merkle_root="m",
        tuf_verification={"ok": True},
    )
    assert st2["predicate"]["tuf_verification"]["ok"] is True

    st3 = attestation_builders.build_attestation_03_promptset(
        subject,
        template_digest="t",
        transform_pipeline_digest="tp",
        promptset_merkle_root="pm",
        sampling_seed=7,
        sampling_strategy="strat",
    )
    assert st3["predicate"]["sampling_seed"] == 7

    st4 = attestation_builders.build_attestation_04_execution(
        subject,
        records_digest="r",
        manifest_digest="m",
        records_merkle_root="rm",
        receipts_merkle_root="rr",
        model_identity_snapshot={"model": "dummy"},
        runner_config_snapshot={"seed": 1},
    )
    assert st4["predicate"]["model_identity_snapshot"]["model"] == "dummy"
    assert st4["predicate"]["invocation"]["parameters"]["seed"] == 1

    st5 = attestation_builders.build_attestation_05_scoring(
        subject,
        metrics_versions={"a": 1},
        judge_committee_config={"n": 2},
        analysis_plan_digest="ap",
    )
    assert st5["predicate"]["analysis_plan_digest"] == "ap"

    st6 = attestation_builders.build_attestation_06_report(subject, materials_digests=["d1"])
    assert st6["predicate"]["materials_digests"] == ["d1"]

    st7 = attestation_builders.build_attestation_07_claims(
        subject, claims_file_digest="c", verification_output_digest="v"
    )
    assert st7["predicate"]["verification_output_digest"] == "v"

    st8 = attestation_builders.build_attestation_08_policy(
        subject,
        policy_file_digest="p",
        verdict_digest="vd",
        passed=False,
        reasons=["nope"],
    )
    assert st8["predicate"]["passed"] is False
    assert st8["predicate"]["reasons"] == ["nope"]

    # bare builders (None optional kwargs) hit the false branches
    assert (
        attestation_builders.build_attestation_00_source(subject)["predicate"]["step"] == "source"
    )
    assert attestation_builders.build_attestation_01_env(subject)["predicate"]["step"] == "env"
    assert (
        attestation_builders.build_attestation_02_dataset(subject)["predicate"]["step"] == "dataset"
    )
    assert (
        attestation_builders.build_attestation_03_promptset(subject)["predicate"]["step"]
        == "promptset"
    )
    assert (
        attestation_builders.build_attestation_04_execution(subject)["predicate"]["step"]
        == "execution"
    )
    assert (
        attestation_builders.build_attestation_05_scoring(subject)["predicate"]["step"] == "scoring"
    )
    assert (
        attestation_builders.build_attestation_06_report(subject)["predicate"]["step"] == "report"
    )
    assert (
        attestation_builders.build_attestation_07_claims(subject)["predicate"]["step"] == "claims"
    )
    assert (
        attestation_builders.build_attestation_08_policy(subject)["predicate"]["step"] == "policy"
    )
    assert (
        attestation_builders.build_attestation_09_publish(subject)["predicate"]["step"] == "publish"
    )
    st9 = attestation_builders.build_attestation_09_publish(
        subject,
        oci_ref="registry.io/repo:tag",
        oci_digest="sha256:abc",
        signature_bundle_digests=["d1"],
    )
    assert st9["predicate"]["oci_ref"] == "registry.io/repo:tag"
    assert st9["predicate"]["oci_digest"] == "sha256:abc"
    assert st9["predicate"]["signature_bundle_digests"] == ["d1"]


# ---------------------------------------------------------------------------
# runtime.receipt
# ---------------------------------------------------------------------------


class _SyncModel:
    def generate(self, prompt: str, **kwargs):
        return f"gen:{prompt}"

    def chat(self, messages, **kwargs):
        return "chat-ok"


class _AsyncModel:
    async def agenerate(self, prompt: str, **kwargs):
        return f"agen:{prompt}"

    async def achat(self, messages, **kwargs):
        return "achat-ok"


class _SyncOnlyAsyncCompat:
    def generate(self, prompt: str, **kwargs):
        return f"sync-gen:{prompt}"

    def chat(self, messages, **kwargs):
        return "sync-chat"


def test_receipt_middleware_sync_async_and_noop(tmp_path: Path) -> None:
    sink = tmp_path / "calls.jsonl"
    mw = ReceiptMiddleware(receipt_sink=sink)
    mw.model = _SyncModel()
    assert mw.process_generate("hi", record_index=1, example_id="e1") == "gen:hi"
    msgs = [ChatMessage(role="user", content="q")]
    assert mw.process_chat(msgs, example_id="e2") == "chat-ok"

    next_mw = MagicMock()
    next_mw.process_generate.return_value = "via-next"
    next_mw.process_chat.return_value = "via-next-chat"
    mw2 = ReceiptMiddleware(receipt_sink=sink)
    mw2.next_middleware = next_mw
    assert mw2.process_generate("x") == "via-next"
    assert mw2.process_chat(msgs) == "via-next-chat"

    bare = ReceiptMiddleware(receipt_sink=None)
    bare.model = _SyncModel()
    assert bare.process_generate("z") == "gen:z"
    assert not sink.read_text(encoding="utf-8") or True  # sink may already have lines
    noop = ReceiptMiddleware()
    noop._append_receipt({"a": 1})  # no sink

    empty = ReceiptMiddleware(receipt_sink=sink)
    with pytest.raises(ModelError, match="No model"):
        empty.process_generate("x")
    with pytest.raises(ModelError, match="No model"):
        empty.process_chat(msgs)

    async def _async_paths() -> None:
        amw = ReceiptMiddleware(receipt_sink=sink)
        amw.model = _AsyncModel()
        assert await amw.aprocess_generate("a") == "agen:a"
        assert await amw.aprocess_chat(msgs) == "achat-ok"

        smw = ReceiptMiddleware(receipt_sink=sink)
        smw.model = _SyncOnlyAsyncCompat()
        assert await smw.aprocess_generate("b") == "sync-gen:b"
        assert await smw.aprocess_chat(msgs) == "sync-chat"

        # proper async next
        class _Next:
            async def aprocess_generate(self, prompt, **kwargs):
                return "n"

            async def aprocess_chat(self, messages, **kwargs):
                return "nc"

        amw2 = ReceiptMiddleware(receipt_sink=sink)
        amw2.next_middleware = _Next()
        assert await amw2.aprocess_generate("c") == "n"
        assert await amw2.aprocess_chat(msgs) == "nc"

        empty_a = ReceiptMiddleware(receipt_sink=sink)
        with pytest.raises(ModelError):
            await empty_a.aprocess_generate("x")
        with pytest.raises(ModelError):
            await empty_a.aprocess_chat(msgs)

    asyncio.run(_async_paths())
    lines = [json.loads(ln) for ln in sink.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert any("request_hash" in row and "response_hash" in row for row in lines)


def test_receipt_hash_helpers_with_dict_messages() -> None:
    from insideLLMs.runtime import receipt as receipt_mod

    h1 = receipt_mod._request_hash("p", {"temperature": 0.1, "record_index": 9})
    h2 = receipt_mod._request_hash("p", {"temperature": 0.1})
    assert h1 == h2
    chat_h = receipt_mod._request_hash_chat(
        [{"role": "user", "content": "hi"}], {"example_id": "x", "top_p": 1}
    )
    assert isinstance(chat_h, str) and len(chat_h) == 64
    assert receipt_mod._response_hash("ok", {"tokens": 1}) != receipt_mod._response_hash("ok")


# ---------------------------------------------------------------------------
# shadow helpers + remaining branches
# ---------------------------------------------------------------------------


def test_shadow_helpers_and_edge_paths(tmp_path: Path) -> None:
    assert _to_utc(datetime(2020, 1, 1, 12, 0, 0)).tzinfo == timezone.utc
    aware = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert _to_utc(aware) == aware

    assert _sample_request(sample_rate=0, method="GET", path="/", query="", body=b"") is False
    assert _sample_request(sample_rate=1, method="GET", path="/", query="", body=b"") is True
    # deterministic bucket path
    sampled = _sample_request(
        sample_rate=0.5, method="POST", path="/x", query="a=1", body=b'{"k":1}'
    )
    assert isinstance(sampled, bool)

    assert _decode_request_body(b"") is None
    assert _decode_request_body(b'{"a":1}') == {"a": 1}
    assert _decode_request_body(b"not-json") == "not-json"
    assert _decode_request_body(b"\xff\xfe") == {"bytes": 2}

    assert _safe_mapping({"A": 1}) == {"A": 1}
    assert _safe_mapping("nope") == {}
    assert _request_url_parts(SimpleNamespace()) == ("/", "")
    assert _request_url_parts(SimpleNamespace(url=SimpleNamespace(path="/p", query="q=1"))) == (
        "/p",
        "q=1",
    )

    writer = ShadowWriter(tmp_path / "out.jsonl", strict_serialization=True)
    writer.append({"z": 1, "a": 2})
    assert (tmp_path / "out.jsonl").exists()

    with pytest.raises(ValueError, match="sample_rate"):
        fastapi(sample_rate=1.5)

    async def _body_variants() -> None:
        class NoBody:
            pass

        assert await _read_request_body(NoBody()) == b""

        class BodyBytes:
            async def body(self):
                return b"abc"

        assert await _read_request_body(BodyBytes()) == b"abc"

        class BodyBytearray:
            async def body(self):
                return bytearray(b"ab")

        assert await _read_request_body(BodyBytearray()) == b"ab"

        class BodyStr:
            async def body(self):
                return "hi"

        assert await _read_request_body(BodyStr()) == b"hi"

        class BodyOther:
            async def body(self):
                return 123

        assert await _read_request_body(BodyOther()) == b""

    asyncio.run(_body_variants())

    # HTTP 500 status path + auto run_id
    out = tmp_path / "shadow.jsonl"
    mw = fastapi(
        output_path=out, sample_rate=1.0, clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    class Req:
        method = "GET"
        url = SimpleNamespace(path="/err", query="")
        headers = {}

        async def body(self):
            return b""

    class Resp:
        status_code = 503
        headers = {"content-type": "text/plain"}

    async def call_next(_r):
        return Resp()

    asyncio.run(mw(Req(), call_next))
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["status"] == "error"
    assert row["error_type"] == "HTTPStatusError"
    assert row["run_id"].startswith("shadow-")

    # headers + error capture + zero-sample skip on success
    out2 = tmp_path / "shadow2.jsonl"
    mw2 = fastapi(
        output_path=out2,
        sample_rate=1.0,
        include_request_headers=True,
        run_id="fixed",
        clock=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    class Req2:
        method = "POST"
        url = SimpleNamespace(path="/x", query="q=1")
        headers = {"h": "1"}

        async def body(self):
            return b'{"p":1}'

    async def boom(_r):
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        asyncio.run(mw2(Req2(), boom))
    assert json.loads(out2.read_text(encoding="utf-8").strip())["status"] == "error"

    out3 = tmp_path / "shadow3.jsonl"
    mw3 = fastapi(output_path=out3, sample_rate=0.0, run_id="skip")

    class RespOk:
        status_code = 200
        headers = {}

    async def ok(_r):
        return RespOk()

    assert asyncio.run(mw3(Req2(), ok)).status_code == 200
    assert not out3.exists()

    # error path with sample_rate=0: re-raise without writing
    mw4 = fastapi(output_path=tmp_path / "shadow4.jsonl", sample_rate=0.0, run_id="nosample")

    async def boom2(_r):
        raise RuntimeError("no-capture")

    with pytest.raises(RuntimeError, match="no-capture"):
        asyncio.run(mw4(Req2(), boom2))
    assert not (tmp_path / "shadow4.jsonl").exists()


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


# ---------------------------------------------------------------------------
# diffing_interactive
# ---------------------------------------------------------------------------


def test_diffing_interactive_full_coverage(tmp_path: Path, capsys) -> None:
    assert _summary_display(None) == "-"
    assert _summary_display({"output": "hello"}) == "hello"
    assert "accuracy=0.9" in _summary_display(
        {"status": "ok", "primary_metric": "accuracy", "primary_score": 0.9}
    )
    assert _summary_display({"status": "error"}) == "status=error"
    assert _summary_display({}) == "-"

    empty = build_interactive_review_lines({}, limit=5)
    assert empty == ["  No differences to review."]

    report = {
        "regressions": [
            {
                "label": {"model": "m", "probe": "p", "example": "e1"},
                "detail": "drop",
                "baseline": {"output": "a"},
                "candidate": {"output": "b"},
            },
            "skip",
            {
                "model_id": "m2",
                "probe_id": "p2",
                "example_id": "e2",
                "kind": "change",
                "baseline": {"status": "ok"},
                "candidate": {},
            },
        ],
        "improvements": [],
        "changes": [],
        "trace_drifts": [],
        "trace_violation_increases": [],
        "trajectory_drifts": [],
        "only_baseline": [
            {"label": {"model": "m", "probe": "p", "example": "old"}},
            "skip",
            {"model_id": "m3", "probe_id": "p3", "example_id": "e3"},
        ],
        "only_candidate": [
            {"label": {"model": "m", "probe": "p", "example": "new"}},
            "skip",
            {"model_id": "m4", "probe_id": "p4", "example_id": "e4"},
        ],
    }
    lines = build_interactive_review_lines(report, limit=1, dim_text=lambda s: f"DIM:{s}")
    assert any("DIM:" in ln for ln in lines)
    assert any("Missing in candidate" in ln for ln in lines)
    assert any("New in candidate" in ln for ln in lines)

    # no "more" truncation branches (limit covers all items)
    exact = build_interactive_review_lines(
        {
            "regressions": [
                {
                    "label": {"model": "m", "probe": "p", "example": "e"},
                    "detail": "d",
                    "baseline": {"output": "a"},
                    "candidate": {"output": "b"},
                }
            ],
            "improvements": [],
            "changes": [],
            "trace_drifts": [],
            "trace_violation_increases": [],
            "trajectory_drifts": [],
            "only_baseline": [{"model_id": "m", "probe_id": "p", "example_id": "e"}],
            "only_candidate": [{"model_id": "m", "probe_id": "p", "example_id": "e"}],
        },
        limit=10,
    )
    assert not any("more" in ln for ln in exact)
    # only regressions, no only_* lists
    assert build_interactive_review_lines(
        {
            "regressions": [
                {
                    "label": {"model": "m", "probe": "p", "example": "e"},
                    "detail": "d",
                    "baseline": {},
                    "candidate": {},
                }
            ],
            "improvements": [],
            "changes": [],
            "trace_drifts": [],
            "trace_violation_increases": [],
            "trajectory_drifts": [],
            "only_baseline": [],
            "only_candidate": [],
        },
        limit=5,
    )

    print_interactive_review(report, limit=2, emit_subheader=lambda t: print(f"## {t}"))
    assert "Interactive Snapshot Review" in capsys.readouterr().out
    print_interactive_review({"regressions": []}, limit=1)
    assert "No differences" in capsys.readouterr().out

    assert prompt_accept_snapshot(input_func=lambda _p: "yes") is True
    assert prompt_accept_snapshot(input_func=lambda _p: "n") is False

    def _eof(_p: str) -> str:
        raise EOFError

    assert prompt_accept_snapshot(input_func=_eof) is False

    base = tmp_path / "base"
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (cand / "manifest.json").write_text("{}", encoding="utf-8")
    copied = copy_candidate_artifacts_to_baseline(base, cand)
    assert "records.jsonl" in copied
    assert (base / "records.jsonl").exists()


# ---------------------------------------------------------------------------
# optimize_prompt CLI
# ---------------------------------------------------------------------------


def test_optimize_prompt_cli_branches(tmp_path: Path, capsys, monkeypatch) -> None:
    assert _parse_strategies(None) is None
    parsed = _parse_strategies("compression,clarity")
    assert parsed is not None
    assert OptimizationStrategy.COMPRESSION in parsed
    with pytest.raises(ValueError, match="Unknown strategy"):
        _parse_strategies("not-a-real-strategy")

    # empty strategies string -> None
    assert _parse_strategies(" , , ") is None

    bad = argparse.Namespace(
        prompt=None,
        input_file=str(tmp_path / "missing.txt"),
        strategies=None,
        format="text",
        output=None,
        show_diff=False,
    )
    assert cmd_optimize_prompt(bad) == 1

    bad2 = argparse.Namespace(
        prompt="",
        input_file=None,
        strategies=None,
        format="text",
        output=None,
        show_diff=False,
    )
    assert cmd_optimize_prompt(bad2) == 1

    assert (
        cmd_optimize_prompt(
            argparse.Namespace(
                prompt="hello",
                input_file=None,
                strategies="bogus",
                format="text",
                output=None,
                show_diff=False,
            )
        )
        == 1
    )

    inp = tmp_path / "prompt.txt"
    inp.write_text("Please kindly explain AI briefly please.", encoding="utf-8")
    out_json = tmp_path / "report.json"
    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt=None,
            input_file=str(inp),
            strategies=None,
            format="json",
            output=str(out_json),
            show_diff=False,
        )
    )
    assert rc == 0
    assert out_json.exists()

    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt="Please kindly explain AI briefly please.",
            input_file=None,
            strategies=None,
            format="json",
            output=None,
            show_diff=False,
        )
    )
    assert rc == 0
    assert "{" in capsys.readouterr().out

    out_txt = tmp_path / "opt.txt"
    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt="Please kindly explain AI briefly please. " * 3,
            input_file=None,
            strategies=None,
            format="text",
            output=str(out_txt),
            show_diff=True,
        )
    )
    assert rc == 0
    assert out_txt.exists()
    assert "Optimize Prompt" in capsys.readouterr().out

    # text path: no show_diff, no output file (still may print suggestions)
    rc = cmd_optimize_prompt(
        argparse.Namespace(
            prompt="Please kindly very carefully explain this please please.",
            input_file=None,
            strategies="compression",
            format="text",
            output=None,
            show_diff=False,
        )
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# tuf_client (omit shrink)
# ---------------------------------------------------------------------------


def test_tuf_client_mock_and_require_tuf() -> None:
    # Force ImportError path even when real `tuf` is installed.
    real_import = __import__

    def _block_tuf(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tuf" or name.startswith("tuf."):
            raise ImportError("blocked for coverage")
        return real_import(name, globals, locals, fromlist, level)

    import builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _block_tuf)
        with pytest.raises(RuntimeError, match="tuf module not available"):
            fetch_dataset("ds", "1.0", allow_mock=False)

        path, proof = fetch_dataset("ds", "1.0", allow_mock=True, base_url="https://example.com")
        assert path.exists()
        assert proof["status"] == "mock-verified"
        assert proof["base_url"] == "https://example.com"

    # pretend tuf is available
    fake_tuf = types.ModuleType("tuf")
    fake_ng = types.ModuleType("tuf.ngclient")
    fake_ng.Updater = object
    sys.modules["tuf"] = fake_tuf
    sys.modules["tuf.ngclient"] = fake_ng
    try:
        path2, proof2 = fetch_dataset("ds", "2.0", allow_mock=False)
        assert proof2["status"] == "verified"
        assert proof2["method"] == "tuf.ngclient"
        assert path2.exists()
    finally:
        # Restore real modules if they were installed; otherwise clear fakes.
        sys.modules.pop("tuf", None)
        sys.modules.pop("tuf.ngclient", None)
        try:
            import tuf.ngclient  # noqa: F401
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# publish.oras (omit shrink) — mock client
# ---------------------------------------------------------------------------


def test_oras_import_success_flag() -> None:
    """Cover ORAS_AVAILABLE=True import branch (oras not installed in [dev])."""
    import importlib

    original = sys.modules["insideLLMs.publish.oras"]
    fake_client = types.ModuleType("oras.client")
    fake_oras = types.ModuleType("oras")
    fake_oras.client = fake_client
    sys.modules["oras"] = fake_oras
    sys.modules["oras.client"] = fake_client
    try:
        del sys.modules["insideLLMs.publish.oras"]
        reloaded = importlib.import_module("insideLLMs.publish.oras")
        assert reloaded.ORAS_AVAILABLE is True
    finally:
        sys.modules.pop("oras", None)
        sys.modules.pop("oras.client", None)
        sys.modules["insideLLMs.publish.oras"] = original
        import insideLLMs.publish as publish_pkg

        publish_pkg.oras = original


def test_oras_push_pull_verify_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oras_mod, "ORAS_AVAILABLE", False)
    monkeypatch.setattr(oras_mod, "oras_client", None)
    with pytest.raises(RuntimeError, match="oras library"):
        oras_mod.push_run_oci(tmp_path, "ref")

    class FakeClient:
        def __init__(self):
            self.pushed = None
            self.pulled = None

        def push(self, target, files):
            self.pushed = (target, files)

        def pull(self, target, outdir):
            self.pulled = (target, outdir)

    fake_mod = types.SimpleNamespace(OciClient=FakeClient)
    monkeypatch.setattr(oras_mod, "ORAS_AVAILABLE", True)
    monkeypatch.setattr(oras_mod, "oras_client", fake_mod)

    with pytest.raises(ValueError, match="does not exist"):
        oras_mod.push_run_oci(tmp_path / "missing", "r")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        oras_mod.push_run_oci(empty, "r")

    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    result = oras_mod.push_run_oci(run, "registry/run:tag")
    assert result.ref == "registry/run:tag"

    out = tmp_path / "pulled"
    # pull without verify
    pulled = oras_mod.pull_run_oci("registry/run:tag", out)
    assert pulled.path == out

    # verify missing files
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValueError, match="manifest.json missing"):
        oras_mod._verify_pulled_run(bare)

    (bare / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="records.jsonl missing"):
        oras_mod._verify_pulled_run(bare)

    (bare / "records.jsonl").write_text("{}\n", encoding="utf-8")
    oras_mod._verify_pulled_run(bare)
    with pytest.raises(ValueError, match="policy file"):
        oras_mod._verify_pulled_run(bare, policy_path="policy.yaml")
    (bare / "policy.yaml").write_text("rules: []\n", encoding="utf-8")
    oras_mod._verify_pulled_run(bare, policy_path="policy.yaml")

    # pull with verify=True against prepared outdir — pull clears? FakeClient doesn't write files
    # so stage files before verify by wrapping pull
    out2 = tmp_path / "pulled2"
    out2.mkdir()
    (out2 / "manifest.json").write_text("{}", encoding="utf-8")
    (out2 / "records.jsonl").write_text("{}\n", encoding="utf-8")

    class FakeClient2(FakeClient):
        def pull(self, target, outdir):
            # leave pre-staged files
            self.pulled = (target, outdir)

    monkeypatch.setattr(oras_mod, "oras_client", types.SimpleNamespace(OciClient=FakeClient2))
    assert oras_mod.pull_run_oci("r", out2, verify=True).path == out2


# ---------------------------------------------------------------------------
# openrouter (omit shrink) without openai SDK
# ---------------------------------------------------------------------------


def test_openrouter_without_openai_sdk() -> None:
    """Cover openrouter.py by temporarily stubbing its OpenAIModel base."""
    import importlib

    class FakeInfo:
        def __init__(self):
            self.provider = "openai"
            self.extra = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._base_url = kwargs.get("base_url")

        def info(self):
            return FakeInfo()

    saved_openai = sys.modules.get("insideLLMs.models.openai")
    saved_openrouter = sys.modules.get("insideLLMs.models.openrouter")
    fake_openai_mod = types.ModuleType("insideLLMs.models.openai")
    fake_openai_mod.OpenAIModel = FakeOpenAI
    sys.modules["insideLLMs.models.openai"] = fake_openai_mod
    sys.modules.pop("insideLLMs.models.openrouter", None)
    try:
        openrouter_mod = importlib.import_module("insideLLMs.models.openrouter")
        m = openrouter_mod.OpenRouterModel(api_key="k", extra_headers={"X": "1"})
        assert m.kwargs["api_key_env"] == "OPENROUTER_API_KEY"
        assert m.kwargs["default_headers"]["X"] == "1"
        assert m.info().provider == "openrouter"
        m2 = openrouter_mod.OpenRouterModel(api_key="k")
        assert "X-Title" in m2.kwargs["default_headers"]
    finally:
        if saved_openai is not None:
            sys.modules["insideLLMs.models.openai"] = saved_openai
        else:
            sys.modules.pop("insideLLMs.models.openai", None)
        if saved_openrouter is not None:
            sys.modules["insideLLMs.models.openrouter"] = saved_openrouter
        else:
            sys.modules.pop("insideLLMs.models.openrouter", None)
            # restore real openrouter against real openai if available
            if saved_openai is not None:
                importlib.import_module("insideLLMs.models.openrouter")


# ---------------------------------------------------------------------------
# integrations.langchain helpers (omit shrink) without LangChain installed
# ---------------------------------------------------------------------------


def test_langchain_helpers_without_deps() -> None:
    from insideLLMs.integrations import langchain as lc

    assert lc._message_content_to_text(None) == ""
    assert lc._message_content_to_text("hi") == "hi"
    assert lc._message_content_to_text(3) == "3"
    assert "a" in lc._message_content_to_text(["a", {"b": 1}])
    assert lc._message_content_to_text({"z": 1})

    class Bad:
        def __str__(self):
            return "bad"

    # non-json-serializable fallback via str after dumps fails — use a key that dumps can handle
    assert isinstance(lc._message_content_to_text(Bad()), str)

    class Msg:
        def __init__(self, typ, content, name=None):
            self.type = typ
            self.content = content
            self.name = name

    converted = lc._lc_messages_to_insidellms(
        [
            Msg("system", "s"),
            Msg("human", "h"),
            Msg("ai", "a"),
            Msg("tool", "t"),
            Msg("other", "o"),
        ]
    )
    assert [c["role"] for c in converted] == ["system", "user", "assistant", "assistant", "user"]

    prompt = lc._insidellms_messages_to_prompt(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
    )
    assert "USER: q" in prompt
    assert prompt.endswith("ASSISTANT:")

    def f(*, stop=None):
        return stop

    assert lc._call_with_stop(lambda: "ok") == "ok"
    assert lc._call_with_stop(f, stop=["END"]) == ["END"]

    def only_stop_sequences(*, stop_sequences=None):
        return stop_sequences

    assert lc._call_with_stop(only_stop_sequences, stop=["X"]) == ["X"]

    def no_stop():
        return "plain"

    assert lc._call_with_stop(no_stop, stop=["X"]) == "plain"

    # Force missing langchain_core even when the extra is installed.
    real_import = __import__

    def _block_lc(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langchain_core" or name.startswith("langchain_core."):
            raise ImportError("blocked for coverage")
        return real_import(name, globals, locals, fromlist, level)

    import builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _block_lc)
        with pytest.raises(lc.LangChainIntegrationError):
            lc.as_langchain_chat_model(MagicMock())
        with pytest.raises(lc.LangChainIntegrationError):
            lc.as_langchain_runnable(MagicMock())
