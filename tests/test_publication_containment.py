"""Offline OCI publication containment at the filesystem boundary."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from insideLLMs.publish import oras


@pytest.fixture
def fake_client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(oras, "ORAS_AVAILABLE", True)
    monkeypatch.setattr(oras, "oras_client", SimpleNamespace(OciClient=lambda: client))
    return client


@pytest.mark.parametrize("kind", ["file", "directory", "fifo", "alias", "dangling"])
def test_public_push_rejects_links_and_special_files(tmp_path, fake_client, kind):
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_bytes(b"records")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"publication-sentinel")
    link = run / ("results.jsonl" if kind == "alias" else "extra")
    if kind == "fifo":
        os.mkfifo(link)
    else:
        target = {
            "file": sentinel,
            "directory": tmp_path,
            "alias": run / "records.jsonl",
            "dangling": tmp_path / "missing",
        }[kind]
        link.symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises((ValueError, OSError)):
        oras.push_run_oci(run, "example.invalid/test:local")
    assert fake_client.push.call_count == 0
    assert sentinel.read_bytes() == b"publication-sentinel"


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_swap_to_symlink_at_os_open_is_rejected(tmp_path, fake_client, monkeypatch, kind):
    run = tmp_path / "run"
    run.mkdir()
    item = run / "item"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"publication-sentinel")
    if kind == "directory":
        item.mkdir()
    else:
        item.write_bytes(b"initial")
    original_open = os.open
    swapped = False

    def swap_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "item" and not swapped:
            swapped = True
            item.rmdir() if kind == "directory" else item.unlink()
            item.symlink_to(tmp_path if kind == "directory" else sentinel)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {swap_open})
    with pytest.raises((OSError, ValueError)):
        oras.push_run_oci(run, "example.invalid/test:local")
    assert swapped
    fake_client.push.assert_not_called()
    assert sentinel.read_bytes() == b"publication-sentinel"


@pytest.mark.parametrize("fails", [False, True])
def test_publisher_reads_sorted_private_bytes_and_snapshot_is_removed(tmp_path, fake_client, fails):
    run = tmp_path / "run"
    run.mkdir()
    (run / "nested").mkdir()
    expected = {"a.txt": b"original", "nested/z.bin": b"\x00\xff"}
    for name, content in expected.items():
        (run / name).write_bytes(content)
    snapshots = []

    def push(*, target, files):
        (run / "a.txt").write_bytes(b"changed after snapshot")
        observed = {}
        for entry in files:
            filename, relative = entry.rsplit(":", 1)
            path = Path(filename)
            snapshot = path.parents[len(Path(relative).parts) - 1]
            assert snapshot != run
            assert snapshot.stat().st_mode & 0o077 == 0
            snapshots.append(snapshot)
            observed[relative] = path.read_bytes()
        assert list(observed) == sorted(expected)
        assert observed == expected
        assert target == "example.invalid/test:local"
        if fails:
            raise RuntimeError("fake publisher failed")

    fake_client.push.side_effect = push
    if fails:
        with pytest.raises(RuntimeError, match="fake publisher failed"):
            oras.push_run_oci(run, "example.invalid/test:local")
    else:
        assert oras.push_run_oci(run, "example.invalid/test:local").digest is None
    assert snapshots and all(not path.parent.exists() for path in snapshots)


def test_unsupported_no_follow_operations_fail_closed(tmp_path, fake_client, monkeypatch):
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(OSError, match="no-follow"):
        oras.push_run_oci(tmp_path, "example.invalid/test:local")
    fake_client.push.assert_not_called()


def test_replacement_with_regular_file_at_open_is_rejected(tmp_path, fake_client, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    artifact = run / "records.jsonl"
    artifact.write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replaced")
    original_open = os.open

    def replace_open(path, flags, *args, **kwargs):
        if path == "records.jsonl":
            replacement.replace(artifact)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_open)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {replace_open})
    with pytest.raises(ValueError, match="changed before copying"):
        oras.push_run_oci(run, "example.invalid/test:local")
    fake_client.push.assert_not_called()


def test_mutation_during_chunked_copy_is_rejected_and_cleaned_up(tmp_path, monkeypatch):
    from insideLLMs import _artifact_snapshot as snapshots

    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "records.jsonl"
    artifact.write_bytes(b"original" * 100)
    monkeypatch.setattr(snapshots, "_CHUNK_SIZE", 7)
    original_open = Path.open
    destinations = []

    def changing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if args == ("xb",):
            destinations.append(path.parent)
            artifact.write_bytes(b"changed while copying")
        return stream

    monkeypatch.setattr(Path, "open", changing_open)
    with pytest.raises(ValueError, match="changed while copying"):
        with snapshots.regular_tree_snapshot(source):
            pytest.fail("mutated source was admitted")
    assert destinations and all(not path.parent.exists() for path in destinations)
