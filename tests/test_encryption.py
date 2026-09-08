import json
import os
import stat

import pytest

try:
    from cryptography.fernet import Fernet

    from insideLLMs.privacy.encryption import decrypt_jsonl, encrypt_jsonl
except BaseException:
    pytest.skip("cryptography not usable", allow_module_level=True)


def test_encrypt_decrypt_jsonl(tmp_path):
    # Create test data
    data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]

    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Generate a key
    key = Fernet.generate_key()

    # Encrypt
    encrypt_jsonl(file_path, key=key)

    # Verify it's encrypted (not valid JSON anymore)
    with open(file_path, "r") as f:
        content = f.read()
        assert "hello" not in content
        with pytest.raises(json.JSONDecodeError):
            json.loads(content.splitlines()[0])

    # Decrypt
    decrypt_jsonl(file_path, key=key)

    # Verify it's back to normal
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == data[0]
        assert json.loads(lines[1]) == data[1]


def test_encrypt_jsonl_no_key_raises(tmp_path):
    file_path = tmp_path / "test.jsonl"
    file_path.write_text('{"id": 1}\n')

    with pytest.raises(ValueError, match="Encryption key is required"):
        encrypt_jsonl(file_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not portable to Windows")
def test_encrypt_decrypt_jsonl_preserves_file_mode(tmp_path):
    file_path = tmp_path / "private.jsonl"
    file_path.write_text('{"secret":"value"}\n', encoding="utf-8")
    file_path.chmod(0o600)
    key = Fernet.generate_key()

    encrypt_jsonl(file_path, key=key)
    assert stat.S_IMODE(file_path.stat().st_mode) == 0o600

    decrypt_jsonl(file_path, key=key)
    assert stat.S_IMODE(file_path.stat().st_mode) == 0o600
