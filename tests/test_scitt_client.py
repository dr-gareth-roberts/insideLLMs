"""Tests for SCITT client: submit, verify, timeout, retries."""

from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.transparency.scitt_client import (
    ScittSubmissionError,
    ScittTimeoutError,
    submit_statement,
    verify_receipt,
)


@patch("insideLLMs.transparency.scitt_client.urllib.request.urlopen")
def test_submit_statement_success(mock_urlopen: MagicMock) -> None:
    """Submit returns success with receipt when service responds."""
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"receipt": "fake_receipt_data"}'
    mock_response.__enter__ = lambda self: self
    mock_response.__exit__ = lambda *a: None
    mock_urlopen.return_value = mock_response

    envelope = {"payload": "base64_encoded_payload", "signatures": []}
    result = submit_statement(envelope, service_url="https://scitt.example.com")

    assert result["status"] == "success"
    assert "receipt" in result
    assert result["receipt"]["receipt"] == "fake_receipt_data"
    assert "statement_digest" in result
    mock_urlopen.assert_called_once()


def test_submit_statement_no_url() -> None:
    """Submit raises ValueError when service_url is empty."""
    with pytest.raises(ValueError, match="service_url is required"):
        submit_statement({"payload": "test"})


@patch("insideLLMs.transparency.scitt_client.urllib.request.urlopen")
def test_submit_statement_returns_error_on_http_failure(mock_urlopen: MagicMock) -> None:
    """Submit returns error dict on HTTP 500."""
    import urllib.error

    mock_urlopen.side_effect = urllib.error.HTTPError(
        "https://scitt.example.com/entries", 500, "Internal Error", {}, None
    )

    result = submit_statement(
        {"payload": "test"}, service_url="https://scitt.example.com", retries=0
    )

    assert result["status"] == "error"
    assert "message" in result
    assert "statement_digest" in result


@patch("insideLLMs.transparency.scitt_client.urllib.request.urlopen")
def test_submit_statement_timeout_returns_error(mock_urlopen: MagicMock) -> None:
    """Submit returns error dict on timeout."""
    mock_urlopen.side_effect = TimeoutError("timed out")

    result = submit_statement(
        {"payload": "test"},
        service_url="https://scitt.example.com",
        retries=0,
    )

    assert result["status"] == "error"
    assert "statement_digest" in result


def test_verify_receipt_success() -> None:
    """Verify passes when status=success, digest matches, receipt is non-empty dict."""
    receipt = {
        "status": "success",
        "statement_digest": "test_digest",
        "receipt": {"proof": "inclusion"},
    }
    assert verify_receipt(receipt, "test_digest") is True


def test_verify_receipt_fails_on_mismatched_digest() -> None:
    """Verify fails when statement_digest does not match."""
    receipt = {
        "status": "success",
        "statement_digest": "test_digest",
        "receipt": {"proof": "inclusion"},
    }
    assert verify_receipt(receipt, "wrong_digest") is False


def test_verify_receipt_fails_on_error_status() -> None:
    """Verify fails when status is not success."""
    assert verify_receipt({"status": "error"}, "test_digest") is False


def test_verify_receipt_fails_when_receipt_not_dict() -> None:
    """Verify fails when inner receipt is not a dict."""
    receipt = {
        "status": "success",
        "statement_digest": "test_digest",
        "receipt": "string_not_dict",
    }
    assert verify_receipt(receipt, "test_digest") is False


def test_verify_receipt_fails_when_receipt_empty() -> None:
    """Verify fails when inner receipt is empty dict."""
    receipt = {
        "status": "success",
        "statement_digest": "test_digest",
        "receipt": {},
    }
    assert verify_receipt(receipt, "test_digest") is False
