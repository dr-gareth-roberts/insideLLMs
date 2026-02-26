"""Tests for ORAS push/pull and verification."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.publish.oras import (
    ORAS_AVAILABLE,
    PullResult,
    PushResult,
    pull_run_oci,
    push_run_oci,
)


@patch("insideLLMs.publish.oras.ORAS_AVAILABLE", True)
@patch("insideLLMs.publish.oras.oras_client")
def test_push_run_oci_returns_push_result(mock_client_module: MagicMock, tmp_path: Path) -> None:
    """Push returns PushResult with ref and digest."""
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client

    (tmp_path / "manifest.json").write_text("{}")

    result = push_run_oci(tmp_path, "ghcr.io/test/repo:latest")

    assert isinstance(result, PushResult)
    assert result.ref == "ghcr.io/test/repo:latest"
    mock_client.push.assert_called_once()


@patch("insideLLMs.publish.oras.ORAS_AVAILABLE", True)
@patch("insideLLMs.publish.oras.oras_client")
def test_pull_run_oci_returns_pull_result(mock_client_module: MagicMock, tmp_path: Path) -> None:
    """Pull returns PullResult with path."""
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client

    out_dir = tmp_path / "pulled"
    result = pull_run_oci("ghcr.io/test/repo:latest", out_dir)

    assert isinstance(result, PullResult)
    assert result.ref == "ghcr.io/test/repo:latest"
    assert result.path == out_dir
    mock_client.pull.assert_called_once()


@patch("insideLLMs.publish.oras.ORAS_AVAILABLE", True)
@patch("insideLLMs.publish.oras.oras_client")
def test_pull_run_oci_verify_succeeds_when_artifacts_present(
    mock_client_module: MagicMock, tmp_path: Path
) -> None:
    """verify=True passes when manifest.json and records.jsonl exist."""
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client

    out_dir = tmp_path / "pulled"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("{}")
    (out_dir / "records.jsonl").write_text("")

    result = pull_run_oci("ghcr.io/test/repo:latest", out_dir, verify=True)

    assert result.path == out_dir


@patch("insideLLMs.publish.oras.ORAS_AVAILABLE", True)
@patch("insideLLMs.publish.oras.oras_client")
def test_pull_run_oci_verify_fails_when_manifest_missing(
    mock_client_module: MagicMock, tmp_path: Path
) -> None:
    """verify=True raises when manifest.json is missing."""
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client

    out_dir = tmp_path / "pulled"
    out_dir.mkdir()
    (out_dir / "records.jsonl").write_text("")

    with pytest.raises(ValueError, match="manifest.json missing"):
        pull_run_oci("ghcr.io/test/repo:latest", out_dir, verify=True)


@patch("insideLLMs.publish.oras.ORAS_AVAILABLE", True)
@patch("insideLLMs.publish.oras.oras_client")
def test_pull_run_oci_verify_fails_when_records_missing(
    mock_client_module: MagicMock, tmp_path: Path
) -> None:
    """verify=True raises when records.jsonl is missing."""
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client

    out_dir = tmp_path / "pulled"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="records.jsonl missing"):
        pull_run_oci("ghcr.io/test/repo:latest", out_dir, verify=True)


@pytest.mark.skipif(ORAS_AVAILABLE, reason="ORAS installed; test for unavailable case")
def test_push_raises_when_oras_unavailable(tmp_path: Path) -> None:
    """push_run_oci raises RuntimeError when oras is not installed."""
    (tmp_path / "manifest.json").write_text("{}")

    with pytest.raises(RuntimeError, match="oras library is required"):
        push_run_oci(tmp_path, "ghcr.io/test/repo:latest")


@pytest.mark.skipif(ORAS_AVAILABLE, reason="ORAS installed; test for unavailable case")
def test_pull_raises_when_oras_unavailable(tmp_path: Path) -> None:
    """pull_run_oci raises RuntimeError when oras is not installed."""
    with pytest.raises(RuntimeError, match="oras library is required"):
        pull_run_oci("ghcr.io/test/repo:latest", tmp_path)
