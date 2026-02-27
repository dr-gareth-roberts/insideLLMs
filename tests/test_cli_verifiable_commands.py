"""Coverage tests for attest/sign/verify-signatures CLI commands."""

import argparse
from pathlib import Path
from unittest.mock import patch

from insideLLMs.cli.commands.attest import cmd_attest
from insideLLMs.cli.commands.sign import cmd_sign
from insideLLMs.cli.commands.verify import cmd_verify_signatures


def _args(run_dir: Path, *, identity: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(run_dir=str(run_dir), identity=identity)


def test_cmd_attest_missing_run_dir(tmp_path: Path) -> None:
    rc = cmd_attest(_args(tmp_path / "missing"))
    assert rc == 1


def test_cmd_attest_missing_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rc = cmd_attest(_args(run_dir))
    assert rc == 1


def test_cmd_attest_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with patch("insideLLMs.cli.commands.attest.run_ultimate_post_artifact") as mock_attest:
        rc = cmd_attest(_args(run_dir))
    assert rc == 0
    mock_attest.assert_called_once_with(run_dir)


def test_cmd_sign_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attestations_dir = run_dir / "attestations"
    attestations_dir.mkdir(parents=True)
    (attestations_dir / "00.manifest.dsse.json").write_text("{}", encoding="utf-8")
    (attestations_dir / "01.records.dsse.json").write_text("{}", encoding="utf-8")

    with patch("insideLLMs.cli.commands.sign.sign_blob") as mock_sign:
        rc = cmd_sign(_args(run_dir))

    assert rc == 0
    assert mock_sign.call_count == 2
    assert (run_dir / "signing").exists()


def test_cmd_sign_no_attestations(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rc = cmd_sign(_args(run_dir))
    assert rc == 1


def test_cmd_verify_signatures_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attestations_dir = run_dir / "attestations"
    signing_dir = run_dir / "signing"
    attestations_dir.mkdir(parents=True)
    signing_dir.mkdir(parents=True)

    dsse_path = attestations_dir / "00.manifest.dsse.json"
    bundle_path = signing_dir / "00.manifest.dsse.sigstore.bundle.json"
    dsse_path.write_text("{}", encoding="utf-8")
    bundle_path.write_text("{}", encoding="utf-8")

    with patch("insideLLMs.cli.commands.verify.verify_bundle", return_value=True) as mock_verify:
        rc = cmd_verify_signatures(_args(run_dir, identity="issuer=example"))

    assert rc == 0
    mock_verify.assert_called_once_with(
        dsse_path, bundle_path, identity_constraints="issuer=example"
    )


def test_cmd_verify_signatures_missing_bundle_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    attestations_dir = run_dir / "attestations"
    signing_dir = run_dir / "signing"
    attestations_dir.mkdir(parents=True)
    signing_dir.mkdir(parents=True)
    (attestations_dir / "00.manifest.dsse.json").write_text("{}", encoding="utf-8")

    rc = cmd_verify_signatures(_args(run_dir))
    assert rc == 1
