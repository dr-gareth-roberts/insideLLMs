"""W7-0008 slice 7: runtime diffing/config_loader/result_utils + export gaps."""

from __future__ import annotations

import gzip
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs._serialization import StrictSerializationError
from insideLLMs.registry import NotFoundError
from insideLLMs.runtime import _config_loader as cfg
from insideLLMs.runtime import _result_utils as ru
from insideLLMs.runtime import diffing as diff


def _ts() -> datetime:
    return datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_diffing_helpers_and_judge_truncation() -> None:
    assert diff._trim_text("short") == "short"
    assert diff._trim_text("x" * 250).endswith("...")

    assert diff._output_text_fingerprint({"output": None}) is None
    assert diff._output_text_fingerprint({"output": "plain"}, ignore_keys=None) == "plain"
    assert diff._output_text_fingerprint({"output": "not-json"}, ignore_keys={"a"}) == "not-json"
    assert (
        diff._output_text_fingerprint({"output": json.dumps({"a": 1, "b": 2})}, ignore_keys={"a"})
        is not None
    )

    rec_events = {
        "custom": {
            "trace_events": [
                {
                    "kind": "tool_call_start",
                    "payload": {"tool_name": "t", "arguments": {"x": 1}},
                },
                {"kind": "tool_result", "payload": {"tool_name": "t", "result": "ok"}},
                {"kind": "generate_start"},
                {"kind": "custom"},
            ]
        }
    }
    steps = diff._trajectory_steps(rec_events)
    assert any(s["kind"] == "tool_call_start" for s in steps)
    assert any(s["kind"] == "tool_result" for s in steps)

    assert diff._trace_events({"output": {"trace_events": [{"kind": "error", "seq": 3}]}})
    assert diff._tool_calls({"output": {"tool_calls": [{"tool_name": "x", "arguments": {}}]}})
    assert diff._tool_calls({"custom": {"tool_calls": [{"name": "y"}]}})
    assert (
        len(
            diff._tool_calls(
                {"custom": {"trace": {"derived": {"tool_calls": {"sequence": ["a", None, "b"]}}}}}
            )
        )
        == 2
    )
    assert diff._trajectory_steps({"output": {"tool_calls": [{"tool_name": "z"}]}})

    assert diff._judge_section_rule("improvements")[0] == "acceptable"
    assert diff._judge_section_rule("changes")[0] == "review"
    assert diff._judge_section_rule("unknown")[0] == "review"
    assert diff._as_label({"label": "bad"}) == {}

    report = {
        "regressions": [{"kind": "score", "label": {"model": "m"}}],
        "improvements": [{"kind": "score"}],
        "changes": [{"kind": "out"}],
        "trace_drifts": "skip",
        "trajectory_drifts": [None, {"kind": "traj"}],
        "only_baseline": [],
        "only_candidate": [],
        "trace_violation_increases": [],
    }
    judged = diff.judge_diff_report(report, policy="balanced", limit=2)
    assert "truncated" in judged.judge_report["summary"]
    assert diff.judge_diff_report(report, policy="strict", limit=1).breaking is True


def test_config_loader_remaining_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "x.txt"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported config file format"):
        cfg.load_config(bad)

    snap = cfg._build_resolved_config_snapshot(
        {"dataset": {"format": "hf", "name": "n"}},
        tmp_path,
    )
    assert snap["dataset"]["format"] == "hf"

    assert cfg._create_middlewares_from_config(["passthrough"])

    monkeypatch.setattr(
        cfg.model_registry,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(NotFoundError("x")),
    )
    with pytest.raises(ValueError, match="Unknown model type"):
        cfg._create_model_from_config({"type": "nope"})

    monkeypatch.setattr(cfg.model_registry, "get", lambda *a, **k: MagicMock())
    piped = cfg._create_model_from_config(
        {
            "type": "dummy",
            "pipeline": {
                "middlewares": [{"type": "passthrough"}],
                "async": True,
                "name": "p",
            },
        },
        prefer_async_pipeline=True,
    )
    assert piped is not None
    piped2 = cfg._create_model_from_config(
        {
            "type": "dummy",
            "pipeline": {"middleware": [{"type": "passthrough"}], "async": False},
        }
    )
    assert piped2 is not None

    monkeypatch.setattr(
        cfg.probe_registry,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(NotFoundError("x")),
    )
    with pytest.raises(ValueError, match="Unknown probe type"):
        cfg._create_probe_from_config({"type": "nope"})

    monkeypatch.setattr(
        cfg.dataset_registry,
        "get_factory",
        lambda *a, **k: (_ for _ in ()).throw(NotFoundError("x")),
    )
    from insideLLMs import dataset_utils

    monkeypatch.setattr(dataset_utils, "load_hf_dataset", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Failed to load HuggingFace"):
        cfg._load_dataset_from_config({"format": "hf", "name": "n"}, tmp_path)

    with pytest.raises(ValueError, match="Unknown dataset format"):
        cfg._load_dataset_from_config({"format": "parquet"}, tmp_path)


def test_result_utils_info_and_strict_paths() -> None:
    class BoomDict:
        def dict(self):
            raise RuntimeError("no")

    class BoomDump:
        def model_dump(self):
            raise RuntimeError("no")

    assert ru._normalize_info_obj_to_dict(BoomDict()) == {}
    assert ru._normalize_info_obj_to_dict(BoomDump()) == {}

    @dataclass
    class Info:
        name: str

    assert ru._normalize_info_obj_to_dict(Info("x"))["name"] == "x"

    base = dict(
        schema_version="1.0.0",
        run_id="r",
        started_at=_ts(),
        completed_at=_ts(),
        model={"model_id": "m"},
        probe={"probe_id": "p"},
        dataset={"name": "d"},
        latency_ms=1.0,
        store_messages=True,
        index=0,
        status="ok",
        error=None,
        strict_serialization=True,
    )

    with patch.object(ru, "_stable_json_dumps", side_effect=StrictSerializationError("bad")):
        with pytest.raises(ValueError, match="JSON-stable message"):
            ru._build_result_record(
                **base,
                item={"messages": [{"role": "user", "content": object()}]},
                output="ok",
            )

    with patch.object(ru, "_fingerprint_value", side_effect=StrictSerializationError("bad")):
        with pytest.raises(ValueError, match="JSON-stable inputs"):
            ru._build_result_record(**base, item="hi", output="ok")

    calls = {"n": 0}
    real_fp = ru._fingerprint_value

    def fp_once(value, *a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise StrictSerializationError("bad")
        return real_fp(value, *a, **k)

    with patch.object(ru, "_fingerprint_value", side_effect=fp_once):
        with pytest.raises(ValueError, match="JSON-stable structured"):
            ru._build_result_record(**base, item="hi", output={"nested": {"x": 1}})

    rec = ru._build_result_record(
        schema_version="1.0.0",
        run_id="r",
        started_at=_ts(),
        completed_at=_ts(),
        model={"model_id": "m"},
        probe={"probe_id": "p"},
        dataset={"name": "d"},
        item="hi",
        output={"score": 0.5, "usage": {"tokens": 1}, "primary_metric": "score"},
        latency_ms=1.0,
        store_messages=False,
        index=0,
        status="error",
        error=ValueError("e"),
        strict_serialization=False,
    )
    assert rec.get("scores") or rec.get("error")


def test_export_bundle_validation_and_schema(tmp_path: Path) -> None:
    from insideLLMs.analysis import export as ex

    data = [{"a": "x", "b": 1, "c": True, "d": 1.5, "e": [1], "f": {"k": 1}}]
    bundle = ex.create_export_bundle(
        data,
        tmp_path / "out",
        formats=[ex.ExportFormat.JSON],
        include_schema=True,
        name="n",
        validate_output=False,
        compress=False,
    )
    assert (
        (Path(bundle) / "schema.json").exists()
        or (Path(bundle).parent / "n" / "schema.json").exists()
        or list(Path(tmp_path / "out" / "n").glob("schema.json"))
    )

    with (
        patch("insideLLMs.schemas.OutputValidator") as OV,
        patch("insideLLMs.schemas.SchemaRegistry") as SR,
    ):
        SR.return_value.EXPORT_METADATA = "ExportMetadata"
        SR.return_value.get_json_schema.return_value = {"type": "object"}
        OV.return_value.validate = MagicMock()
        bundle2 = ex.create_export_bundle(
            data,
            tmp_path / "out2",
            formats=[ex.ExportFormat.JSON],
            include_schema=True,
            validate_output=True,
            validate_schema_name="ResultRecord",
            schema_version="1.0.0",
            name="n2",
            compress=False,
        )
        meta = Path(tmp_path / "out2" / "n2" / "metadata.json")
        assert meta.exists() or Path(bundle2).exists()
        assert OV.return_value.validate.called

    arch = ex.DataArchiver(ex.CompressionType.GZIP)
    gz = tmp_path / "f.gz"
    gz.write_bytes(gzip.compress(b"hello"))
    out_path = arch.decompress_file(gz)
    assert Path(out_path).read_bytes() == b"hello"

    # unknown suffix → .decompressed default
    weird = tmp_path / "f.weird"
    weird.write_bytes(gzip.compress(b"z"))
    # compression GZIP still opens via gzip because compression type matches
    arch.decompress_file(weird, output_path=tmp_path / "out.bin")

    # zip branch
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("inner.txt", "hi")
    arch_zip = ex.DataArchiver(ex.CompressionType.ZIP)
    arch_zip.decompress_file(zpath, output_path=tmp_path / "unz")
