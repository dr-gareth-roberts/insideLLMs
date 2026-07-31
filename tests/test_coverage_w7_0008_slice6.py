"""W7-0008 slice 6: CLI/attestations/policy/structured/runtime measured gaps."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.attestations.dsse import build_dsse_envelope, parse_dsse_envelope
from insideLLMs.cli.commands.attest import cmd_attest
from insideLLMs.cli.commands.generate_suite import (
    _normalize_records,
    cmd_generate_suite,
)
from insideLLMs.cli.commands.sign import cmd_sign
from insideLLMs.cli.commands.verify import cmd_verify_signatures
from insideLLMs.datasets.commitments import dataset_merkle_root
from insideLLMs.policy.engine import run_policy
from insideLLMs.privacy.redaction import redact_pii
from insideLLMs.structured import ParsingError, StructuredResult, extract_json, parse_to_type


def test_dsse_parse_error_paths() -> None:
    env = build_dsse_envelope({"a": 1}, signatures=[{"keyid": "k", "sig": "cw=="}])
    payload, ptype = parse_dsse_envelope(env)
    assert payload["a"] == 1
    assert "in-toto" in ptype

    with pytest.raises(ValueError, match="payloadType"):
        parse_dsse_envelope({"payload": "e30="})
    with pytest.raises(ValueError, match="Invalid base64"):
        parse_dsse_envelope({"payloadType": "t", "payload": "!!!notb64!!!"})
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_dsse_envelope(
            {
                "payloadType": "t",
                "payload": base64.standard_b64encode(b"not-json").decode("ascii"),
            }
        )


def test_commitments_and_redaction() -> None:
    root = dataset_merkle_root([{"x": 1}, {"x": 2}])
    assert "root" in root
    assert redact_pii(("a@b.com", 1)) == (redact_pii("a@b.com"), 1)


def test_policy_engine_scitt_and_missing(tmp_path: Path) -> None:
    # empty → all missing
    v = run_policy(tmp_path)
    assert v["passed"] is False
    assert "manifest.json missing" in v["reasons"]

    # full-ish tree with bad scitt
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "records.jsonl").write_text("{}\n", encoding="utf-8")
    att = tmp_path / "attestations"
    att.mkdir()
    for name in (
        "00.source",
        "01.env",
        "02.dataset",
        "03.promptset",
        "04.execution",
        "05.scoring",
        "06.report",
        "07.claims",
    ):
        (att / f"{name}.dsse.json").write_text('{"payload":"e30="}', encoding="utf-8")
    (tmp_path / "integrity").mkdir()
    (tmp_path / "integrity" / "records.merkle.json").write_text("{}", encoding="utf-8")

    scitt = tmp_path / "receipts" / "scitt"
    scitt.mkdir(parents=True)
    (scitt / "04.execution.receipt.json").write_text("{}", encoding="utf-8")
    # receipt without attestation
    (scitt / "07.claims.receipt.json").write_text("{}", encoding="utf-8")
    (att / "07.claims.dsse.json").unlink()

    with patch("insideLLMs.policy.engine.verify_receipt", return_value=False):
        v2 = run_policy(tmp_path)
    assert v2["passed"] is False
    assert any("scitt" in r for r in v2["reasons"])

    # valid receipt path
    (att / "07.claims.dsse.json").write_text('{"payload":"e30="}', encoding="utf-8")
    with patch("insideLLMs.policy.engine.verify_receipt", return_value=True):
        with patch(
            "insideLLMs.policy.engine.digest_obj",
            return_value={"digest": "d"},
        ):
            v3 = run_policy(tmp_path)
    assert v3["checks"].get("scitt_04.execution") is True


def test_cli_sign_verify_attest(tmp_path: Path) -> None:
    assert cmd_sign(argparse.Namespace(run_dir=str(tmp_path / "missing"))) == 1
    run = tmp_path / "run"
    run.mkdir()
    assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 1  # no attestations/
    att = run / "attestations"
    att.mkdir()
    assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 1  # no dsse files

    dsse = att / "00.source.dsse.json"
    dsse.write_text("{}", encoding="utf-8")
    with patch("insideLLMs.cli.commands.sign.sign_blob", side_effect=RuntimeError("no")):
        assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 1
    with patch("insideLLMs.cli.commands.sign.sign_blob"):
        assert cmd_sign(argparse.Namespace(run_dir=str(run))) == 0

    assert (
        cmd_verify_signatures(argparse.Namespace(run_dir=str(tmp_path / "x"), identity=None)) == 1
    )
    signing = run / "signing"
    if signing.exists():
        for p in signing.iterdir():
            p.unlink()
        signing.rmdir()
    assert (
        cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    )  # no signing/
    signing.mkdir()
    # missing bundle
    assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    bundle = signing / "00.source.dsse.sigstore.bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    with patch("insideLLMs.cli.commands.verify.verify_bundle", return_value=False):
        assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    with patch("insideLLMs.cli.commands.verify.verify_bundle", side_effect=RuntimeError("boom")):
        assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 1
    with patch("insideLLMs.cli.commands.verify.verify_bundle", return_value=True):
        assert cmd_verify_signatures(argparse.Namespace(run_dir=str(run), identity=None)) == 0

    assert cmd_attest(argparse.Namespace(run_dir=str(tmp_path / "nope"))) == 1
    assert cmd_attest(argparse.Namespace(run_dir=str(run))) == 1  # no manifest
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    with patch(
        "insideLLMs.cli.commands.attest.run_ultimate_post_artifact",
        side_effect=RuntimeError("x"),
    ):
        assert cmd_attest(argparse.Namespace(run_dir=str(run))) == 1
    with patch("insideLLMs.cli.commands.attest.run_ultimate_post_artifact"):
        assert cmd_attest(argparse.Namespace(run_dir=str(run))) == 0


def test_generate_suite_paths(tmp_path: Path) -> None:
    # normalize empty / non-str prompt
    recs = _normalize_records([{"text": "  "}, {"foo": 1, "adversarial": True}], target="ops")
    assert recs[0]["prompt"]
    assert recs[1]["adversarial"] is True

    out = tmp_path / "suite.jsonl"
    args = argparse.Namespace(
        target="ops",
        num_cases=2,
        model="dummy",
        include_adversarial=True,
        output=str(out),
        model_args="[]",  # invalid object
        seed_example=["  hi  ", ""],
        format="jsonl",
    )
    assert cmd_generate_suite(args) == 1

    args.model_args = "{}"
    with patch(
        "insideLLMs.cli.commands.generate_suite.resolve_registered_model",
        side_effect=RuntimeError("no model"),
    ):
        assert cmd_generate_suite(args) == 1

    with patch(
        "insideLLMs.cli.commands.generate_suite.resolve_registered_model",
        return_value=MagicMock(),
    ):
        with patch(
            "insideLLMs.cli.commands.generate_suite.generate_test_dataset",
            side_effect=RuntimeError("gen fail"),
        ):
            assert cmd_generate_suite(args) == 1

        with patch(
            "insideLLMs.cli.commands.generate_suite.generate_test_dataset",
            return_value=[{"text": "a" * 120, "adversarial": True, "synthetic": True}],
        ):
            assert cmd_generate_suite(args) == 0
            assert out.exists()

            args2 = argparse.Namespace(
                **{**vars(args), "output": str(tmp_path / "s.json"), "format": "json"}
            )
            assert cmd_generate_suite(args2) == 0

            # write failure
            args3 = argparse.Namespace(
                **{**vars(args), "output": str(tmp_path / "blocked" / "x.jsonl")}
            )
            with patch("pathlib.Path.write_text", side_effect=OSError("nope")):
                # jsonl uses open(); force open failure
                with patch("builtins.open", side_effect=OSError("nope")):
                    assert cmd_generate_suite(args3) == 1


def test_structured_remaining_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    import insideLLMs.structured as st

    assert extract_json('{"k":1}') == '{"k":1}'
    with pytest.raises(ParsingError):
        extract_json("no json here at all")

    class V1Data:
        def dict(self):
            return {"v": 1}

    assert StructuredResult(
        data=V1Data(),
        raw_response="{}",
        schema={},
        prompt="p",
        model_name="m",
    ).to_dict() == {"v": 1}

    # pydantic v1 parse_obj branch: BaseModel subclass without model_validate
    class FakeBM:
        pass

    monkeypatch.setattr(st, "PYDANTIC_AVAILABLE", True)
    monkeypatch.setattr(st, "BaseModel", FakeBM)
    monkeypatch.setattr(st, "ValidationError", Exception)

    class V1(FakeBM):
        @classmethod
        def parse_obj(cls, data):
            return {"ok": data}

    assert st.parse_to_type({"a": 1}, V1) == {"ok": {"a": 1}}


def test_dataset_utils_jsonl_and_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from insideLLMs import dataset_utils as du

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        du.load_jsonl_dataset(str(bad))

    monkeypatch.setattr(du, "HF_DATASETS_AVAILABLE", False)
    monkeypatch.setattr(du, "load_dataset", None)
    with pytest.raises(ImportError, match="HuggingFace"):
        du.load_hf_dataset("x")

    monkeypatch.setattr(du, "HF_DATASETS_AVAILABLE", True)

    def fake_load(name, split="test", **kwargs):
        return [{"a": 1}, {"b": 2}]

    monkeypatch.setattr(du, "load_dataset", fake_load)
    assert du.load_hf_dataset("x") == [{"a": 1}, {"b": 2}]


def test_models_dummy_canned_and_getattr() -> None:
    import insideLLMs.models as models_pkg
    from insideLLMs.models import DummyModel

    m = DummyModel(name="d", canned_response="CAN")
    assert m.generate("hi") == "CAN"
    assert m.chat([{"role": "user", "content": "x"}]) == "CAN"

    with pytest.raises(AttributeError):
        models_pkg.__getattr__("TotallyMissingModel")
