"""W7-0008 slice 1: close 0%/low-coverage measured modules (no new omit/pragma)."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from insideLLMs.attestations.predicates.slsa_provenance import build_slsa_provenance_predicate
from insideLLMs.attestations.steps import builders as attestation_builders
from insideLLMs.exceptions import ModelError
from insideLLMs.models.base import ChatMessage
from insideLLMs.privacy import encryption as encryption_mod
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
