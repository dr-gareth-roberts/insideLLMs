import pytest
from insideLLMs.model_identity import collect_declared_identity, run_fingerprint_suite, detect_drift

class MockModel:
    def __init__(self, model_name, provider):
        self.model_name = model_name
        self.provider = provider
        self.responses = {
            "What is 2+2?": "4",
            "Count to 3: 1, 2, ": "3"
        }
        
    def generate(self, prompt, **kwargs):
        return self.responses.get(prompt, "unknown")

def test_collect_declared_identity():
    model = MockModel("gpt-4", "openai")
    ident = collect_declared_identity(model)
    assert ident["model_id"] == "gpt-4"
    assert ident["provider"] == "openai"

def test_run_fingerprint_suite():
    model = MockModel("gpt-4", "openai")
    report = run_fingerprint_suite(model)
    
    assert report["status"] == "success"
    assert report["model_id"] == "gpt-4"
    assert report["provider"] == "openai"
    assert len(report["fingerprints"]) > 0
    
    # Check that responses are hashed
    fp = report["fingerprints"][0]
    assert "prompt" in fp
    assert "response_hash" in fp
    assert len(fp["response_hash"]) == 64 # sha256

def test_detect_drift():
    report_a = {
        "fingerprints": [
            {"prompt": "A", "response_hash": "hash1"},
            {"prompt": "B", "response_hash": "hash2"}
        ]
    }
    
    # Same hashes
    report_b = {
        "fingerprints": [
            {"prompt": "A", "response_hash": "hash1"},
            {"prompt": "B", "response_hash": "hash2"}
        ]
    }
    
    result = detect_drift(report_a, report_b)
    assert result["drift_detected"] is False
    assert len(result["differences"]) == 0
    
    # Different hashes
    report_c = {
        "fingerprints": [
            {"prompt": "A", "response_hash": "hash1"},
            {"prompt": "B", "response_hash": "hash3"} # Drifted
        ]
    }
    
    result2 = detect_drift(report_a, report_c)
    assert result2["drift_detected"] is True
    assert len(result2["differences"]) == 1
    assert result2["differences"][0]["prompt"] == "B"
