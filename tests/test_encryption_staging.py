import os
import stat
import threading
from pathlib import Path

import pytest

from insideLLMs.privacy import encryption
from insideLLMs.privacy.encryption import decrypt_jsonl, encrypt_jsonl

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    pytest.skip("cryptography not usable", allow_module_level=True)


def test_encrypt_does_not_touch_preexisting_staging_link(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b'{"value": 1}\n')
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"untouched")
    link = tmp_path / "records.jsonl.enc.tmp"
    link.symlink_to(sentinel)

    encrypt_jsonl(source, key=Fernet.generate_key())

    assert sentinel.read_bytes() == b"untouched"
    assert link.is_symlink()
    assert not source.is_symlink()


def test_decrypt_does_not_touch_preexisting_staging_link(tmp_path):
    key = Fernet.generate_key()
    source = tmp_path / "records.jsonl"
    source.write_bytes(Fernet(key).encrypt(b'{"value": 1}') + b"\n")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"untouched")
    link = tmp_path / "records.jsonl.dec.tmp"
    link.symlink_to(sentinel)

    decrypt_jsonl(source, key=key)

    assert source.read_bytes() == b'{"value": 1}\n'
    assert sentinel.read_bytes() == b"untouched"
    assert link.is_symlink()


def test_encrypt_preserves_preexisting_regular_staging_file(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b'{"value": 1}\n')
    old_stage = tmp_path / "records.jsonl.enc.tmp"
    old_stage.write_bytes(b"preexisting")

    encrypt_jsonl(source, key=Fernet.generate_key())

    assert old_stage.read_bytes() == b"preexisting"


def test_invalid_token_midway_preserves_source_and_removes_owned_stage(tmp_path):
    key = Fernet.generate_key()
    fernet = Fernet(key)
    source = tmp_path / "records.jsonl"
    original = fernet.encrypt(b'{"value": 1}') + b"\nnot-a-token\n"
    source.write_bytes(original)

    with pytest.raises(InvalidToken):
        decrypt_jsonl(source, key=key)

    assert source.read_bytes() == original
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def test_source_symlink_is_rejected_without_modifying_target(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_bytes(b'{"value": 1}\n')
    source = tmp_path / "records.jsonl"
    source.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        encrypt_jsonl(source, key=Fernet.generate_key())

    assert target.read_bytes() == b'{"value": 1}\n'


@pytest.mark.skipif(os.name != "posix", reason="FIFO is a POSIX special file")
def test_special_file_source_is_rejected(tmp_path):
    source = tmp_path / "records.jsonl"
    os.mkfifo(source)

    with pytest.raises(ValueError, match="regular file"):
        encrypt_jsonl(source, key=Fernet.generate_key())


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not portable to Windows")
def test_stage_stays_private_until_transform_succeeds_and_source_mode_is_preserved(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b'{"value": 1}\n')
    source.chmod(0o644)
    observed_modes: list[int] = []

    def transform(line: bytes) -> bytes:
        stages = list(tmp_path.glob(".records.jsonl.*.tmp"))
        assert len(stages) == 1
        observed_modes.append(stat.S_IMODE(stages[0].stat().st_mode))
        return line.upper()

    encryption._transform_jsonl(source, transform)

    assert observed_modes == [0o600]
    assert stat.S_IMODE(source.stat().st_mode) == 0o644


def test_replace_failure_preserves_source_and_removes_owned_stage(tmp_path, monkeypatch):
    source = tmp_path / "records.jsonl"
    original = b'{"value": 1}\n'
    source.write_bytes(original)

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(encryption.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        encrypt_jsonl(source, key=Fernet.generate_key())

    assert source.read_bytes() == original
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def test_cleanup_failure_keeps_transform_exception_primary(tmp_path, monkeypatch, caplog):
    source = tmp_path / "records.jsonl"
    original = b'{"value": 1}\n'
    source.write_bytes(original)
    real_unlink = Path.unlink

    def fail_owned_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(".records.jsonl."):
            raise OSError("cleanup path detail must be redacted")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_owned_unlink)

    def fail_transform(_line: bytes) -> bytes:
        raise RuntimeError("transform failed")

    with pytest.raises(RuntimeError, match="transform failed"):
        encryption._transform_jsonl(source, fail_transform)

    assert source.read_bytes() == original
    assert "cleanup path detail" not in caplog.text
    assert "OSError" in caplog.text


def test_concurrent_distinct_files_use_distinct_owned_stages(tmp_path):
    sources = [tmp_path / "one.jsonl", tmp_path / "two.jsonl"]
    for source in sources:
        source.write_bytes(b'{"value": 1}\n')
    barrier = threading.Barrier(2)
    observed_stages: list[set[str]] = []

    def transform_source(source: Path) -> None:
        def transform(line: bytes) -> bytes:
            barrier.wait(timeout=5)
            observed_stages.append({path.name for path in tmp_path.glob(".*.tmp")})
            return line

        encryption._transform_jsonl(source, transform)

    threads = [threading.Thread(target=transform_source, args=(source,)) for source in sources]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(observed_stages) == 2
    assert all(len(stages) == 2 for stages in observed_stages)
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("close_failure", [False, True])
def test_fdopen_failure_closes_raw_stage_descriptor_and_preserves_source(
    tmp_path, monkeypatch, close_failure
):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b"original\n")
    sentinel = tmp_path / ".records.jsonl.unowned.tmp"
    sentinel.write_bytes(b"unowned")
    descriptors = []
    real_close = os.close

    def fail_fdopen(descriptor, mode):
        descriptors.append(descriptor)
        raise RuntimeError("stream construction failed")

    def close(descriptor):
        real_close(descriptor)
        if close_failure:
            raise OSError("close failed")

    monkeypatch.setattr(encryption.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(encryption.os, "close", close)
    try:
        with pytest.raises(RuntimeError, match="stream construction failed"):
            encryption._transform_jsonl(source, lambda line: line)
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        assert source.read_bytes() == b"original\n"
        assert sentinel.read_bytes() == b"unowned"
        assert list(tmp_path.glob(".records.jsonl.*.tmp")) == [sentinel]
    finally:
        for descriptor in descriptors:
            try:
                real_close(descriptor)
            except OSError:
                pass
