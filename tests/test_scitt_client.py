import pytest
from unittest.mock import patch, MagicMock
from insideLLMs.transparency.scitt_client import submit_statement, verify_receipt

@patch("insideLLMs.transparency.scitt_client.urllib.request.urlopen")
def test_submit_statement(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"receipt": "fake_receipt_data"}'
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response
    
    envelope = {"payload": "base64_encoded_payload", "signatures": []}
    
    receipt = submit_statement(envelope, service_url="https://scitt.example.com")
    
    assert receipt["status"] == "success"
    assert "receipt" in receipt
    assert receipt["receipt"]["receipt"] == "fake_receipt_data"
    mock_urlopen.assert_called_once()

def test_submit_statement_no_url():
    with pytest.raises(ValueError):
        submit_statement({"payload": "test"})

def test_verify_receipt():
    # In our minimal implementation, it just checks if it has a receipt and matches digest
    receipt = {
        "status": "success",
        "receipt": "fake_receipt_data",
        "statement_digest": "test_digest"
    }
    
    # Matching digest
    assert verify_receipt(receipt, "test_digest") is True
    
    # Mismatching digest
    assert verify_receipt(receipt, "wrong_digest") is False
    
    # Invalid receipt
    assert verify_receipt({"status": "error"}, "test_digest") is False
