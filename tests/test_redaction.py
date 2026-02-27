import pytest
from insideLLMs.privacy.redaction import redact_pii

def test_redact_pii_string():
    text = "My email is john.doe@example.com and phone is 555-123-4567."
    redacted = redact_pii(text)
    assert "john.doe@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "[EMAIL]" in redacted
    assert "[PHONE]" in redacted

def test_redact_pii_dict():
    data = {
        "user": "alice@example.com",
        "profile": {
            "phone": "Call me at 123-456-7890",
            "age": 30
        },
        "tags": ["normal", "contact: bob@example.com"]
    }
    redacted = redact_pii(data)
    assert redacted["user"] == "[EMAIL]"
    assert redacted["profile"]["phone"] == "Call me at [PHONE]"
    assert redacted["profile"]["age"] == 30
    assert redacted["tags"][0] == "normal"
    assert redacted["tags"][1] == "contact: [EMAIL]"

def test_redact_pii_other_types():
    assert redact_pii(42) == 42
    assert redact_pii(True) is True
    assert redact_pii(None) is None
