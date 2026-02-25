import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from insideLLMs.publish.oras import push_run_oci, pull_run_oci

@patch("insideLLMs.publish.oras.oras_client")
def test_push_run_oci(mock_client_module, tmp_path):
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client
    
    # Create dummy files
    (tmp_path / "manifest.json").write_text("{}")
    
    # Should not raise
    push_run_oci(tmp_path, "ghcr.io/test/repo:latest")
    
    # Verify client was called
    mock_client.push.assert_called_once()
    
@patch("insideLLMs.publish.oras.oras_client")
def test_pull_run_oci(mock_client_module, tmp_path):
    mock_client = MagicMock()
    mock_client_module.OciClient.return_value = mock_client
    
    # Should not raise
    out_dir = pull_run_oci("ghcr.io/test/repo:latest", tmp_path)
    
    # Verify client was called
    mock_client.pull.assert_called_once()
    assert out_dir == tmp_path
