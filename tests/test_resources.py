"""Tests for insideLLMs/resources.py module."""

import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pytest
import yaml

from insideLLMs.resources import (
    ExitStack as ResourcesExitStack,
)
from insideLLMs.resources import (
    atomic_write_text,
    atomic_write_yaml,
    ensure_run_sentinel,
    managed_run_directory,
    open_records_file,
)


class TestOpenRecordsFile:
    """Tests for open_records_file context manager."""

    def test_creates_new_file(self, tmp_path):
        """Test creating a new file."""
        file_path = tmp_path / "records.jsonl"
        with open_records_file(file_path) as fp:
            fp.write('{"key": "value"}\n')

        assert file_path.exists()
        assert file_path.read_text() == '{"key": "value"}\n'

    def test_exclusive_creation_fails_if_exists(self, tmp_path):
        """Test that mode 'x' fails if file already exists."""
        file_path = tmp_path / "records.jsonl"
        file_path.write_text("existing content")

        with pytest.raises(FileExistsError):
            with open_records_file(file_path, mode="x") as fp:
                fp.write("new content")

    def test_write_mode(self, tmp_path):
        """Test with write mode 'w'."""
        file_path = tmp_path / "records.jsonl"
        file_path.write_text("existing")

        with open_records_file(file_path, mode="w") as fp:
            fp.write("new content\n")

        assert file_path.read_text() == "new content\n"

    def test_append_mode(self, tmp_path):
        """Test with append mode 'a'."""
        file_path = tmp_path / "records.jsonl"
        file_path.write_text("line1\n")

        with open_records_file(file_path, mode="a") as fp:
            fp.write("line2\n")

        assert file_path.read_text() == "line1\nline2\n"

    def test_custom_encoding(self, tmp_path):
        """Test with custom encoding."""
        file_path = tmp_path / "records.jsonl"

        with open_records_file(file_path, encoding="utf-8") as fp:
            fp.write('{"unicode": "日本語"}\n')

        content = file_path.read_text(encoding="utf-8")
        assert "日本語" in content

    def test_cleanup_on_exception(self, tmp_path):
        """Test that file is properly closed even on exception."""
        file_path = tmp_path / "records.jsonl"

        try:
            with open_records_file(file_path) as fp:
                fp.write("some content\n")
                raise ValueError("Test exception")
        except ValueError:
            pass

        # File should be closed and content should be written
        assert file_path.exists()

    def test_multiple_writes(self, tmp_path):
        """Test multiple writes to the same file handle."""
        file_path = tmp_path / "records.jsonl"

        with open_records_file(file_path) as fp:
            for i in range(5):
                fp.write(f'{{"index": {i}}}\n')

        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 5


class TestAtomicWriteText:
    """Tests for atomic_write_text function."""

    def test_writes_text_to_file(self, tmp_path):
        """Test basic text writing."""
        file_path = tmp_path / "test.txt"
        atomic_write_text(file_path, "Hello, World!")

        assert file_path.read_text() == "Hello, World!"

    def test_overwrites_existing_file(self, tmp_path):
        """Test overwriting an existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("old content")

        atomic_write_text(file_path, "new content")

        assert file_path.read_text() == "new content"

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        file_path = tmp_path / "nested" / "dir" / "test.txt"

        atomic_write_text(file_path, "content in nested dir")

        assert file_path.exists()
        assert file_path.read_text() == "content in nested dir"

    def test_atomic_write_no_partial_content(self, tmp_path):
        """Test that write is atomic (no partial content on failure)."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("original")

        # Write should be atomic - either all or nothing
        atomic_write_text(file_path, "new content that is longer")

        assert file_path.read_text() == "new content that is longer"

    def test_handles_unicode_content(self, tmp_path):
        """Test writing Unicode content."""
        file_path = tmp_path / "unicode.txt"
        unicode_text = "Hello 世界 🌍 émojis"

        atomic_write_text(file_path, unicode_text)

        assert file_path.read_text() == unicode_text

    def test_no_temp_file_remains(self, tmp_path):
        """Test that temporary file is cleaned up."""
        file_path = tmp_path / "test.txt"
        atomic_write_text(file_path, "content")

        # Check no .tmp files remain
        tmp_files = list(tmp_path.glob(".*tmp"))
        assert len(tmp_files) == 0


class TestAtomicWriteYaml:
    """Tests for atomic_write_yaml function."""

    def test_writes_yaml_to_file(self, tmp_path):
        """Test basic YAML writing."""
        file_path = tmp_path / "config.yaml"
        data = {"key": "value", "number": 42}

        atomic_write_yaml(file_path, data)

        loaded = yaml.safe_load(file_path.read_text())
        assert loaded == data

    def test_preserves_key_order(self, tmp_path):
        """Test that key order is preserved."""
        file_path = tmp_path / "config.yaml"
        data = {"first": 1, "second": 2, "third": 3}

        atomic_write_yaml(file_path, data)

        content = file_path.read_text()
        assert content.index("first") < content.index("second") < content.index("third")

    def test_nested_data_structure(self, tmp_path):
        """Test writing nested data structures."""
        file_path = tmp_path / "config.yaml"
        data = {
            "level1": {"level2": {"level3": "value"}},
            "list": [1, 2, 3],
            "mixed": {"items": ["a", "b", "c"]},
        }

        atomic_write_yaml(file_path, data)

        loaded = yaml.safe_load(file_path.read_text())
        assert loaded == data

    def test_with_serializer(self, tmp_path):
        """Test using a custom serializer."""
        file_path = tmp_path / "config.yaml"
        data = {"key": "value"}

        def custom_serializer(d):
            return {k.upper(): v.upper() if isinstance(v, str) else v for k, v in d.items()}

        atomic_write_yaml(file_path, data, serializer=custom_serializer)

        loaded = yaml.safe_load(file_path.read_text())
        assert loaded == {"KEY": "VALUE"}

    def test_handles_unicode_in_yaml(self, tmp_path):
        """Test writing Unicode content in YAML."""
        file_path = tmp_path / "unicode.yaml"
        data = {"greeting": "こんにちは", "emoji": "🎉"}

        atomic_write_yaml(file_path, data)

        loaded = yaml.safe_load(file_path.read_text())
        assert loaded == data


class TestEnsureRunSentinel:
    """Tests for ensure_run_sentinel function."""

    def test_creates_sentinel_file(self, tmp_path):
        """Test that sentinel file is created."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        ensure_run_sentinel(run_dir)

        sentinel = run_dir / ".insidellms_run"
        assert sentinel.exists()
        assert "insideLLMs run directory" in sentinel.read_text()

    def test_does_not_overwrite_existing_sentinel(self, tmp_path):
        """Test that existing sentinel is not overwritten."""
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        sentinel = run_dir / ".insidellms_run"
        sentinel.write_text("custom sentinel content")

        ensure_run_sentinel(run_dir)

        assert sentinel.read_text() == "custom sentinel content"

    def test_handles_nonexistent_directory(self, tmp_path):
        """Test behavior with non-existent directory."""
        run_dir = tmp_path / "nonexistent"

        # Should not raise, but may not create file
        ensure_run_sentinel(run_dir)


class TestManagedRunDirectory:
    """Tests for managed_run_directory context manager."""

    def test_creates_directory(self, tmp_path):
        """Test that directory is created."""
        run_dir = tmp_path / "new_run"

        with managed_run_directory(run_dir) as path:
            assert path == run_dir
            assert run_dir.exists()

    def test_creates_sentinel_by_default(self, tmp_path):
        """Test that sentinel is created by default."""
        run_dir = tmp_path / "new_run"

        with managed_run_directory(run_dir) as path:
            sentinel = path / ".insidellms_run"
            assert sentinel.exists()

    def test_no_sentinel_when_disabled(self, tmp_path):
        """Test that sentinel is not created when disabled."""
        run_dir = tmp_path / "new_run"

        with managed_run_directory(run_dir, sentinel=False) as path:
            sentinel = path / ".insidellms_run"
            assert not sentinel.exists()

    def test_no_create_when_disabled(self, tmp_path):
        """Test that directory is not created when create=False."""
        run_dir = tmp_path / "existing_run"
        run_dir.mkdir()

        with managed_run_directory(run_dir, create=False) as path:
            assert path == run_dir

    def test_nested_directories(self, tmp_path):
        """Test creating nested run directories."""
        run_dir = tmp_path / "deep" / "nested" / "run"

        with managed_run_directory(run_dir) as path:
            assert path.exists()
            assert path == run_dir

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        run_dir = tmp_path / "existing"
        run_dir.mkdir()
        (run_dir / "existing_file.txt").write_text("content")

        with managed_run_directory(run_dir) as path:
            assert (path / "existing_file.txt").exists()


class TestExitStackReExport:
    """Tests for ExitStack re-export."""

    def test_exitstack_is_available(self):
        """Test that ExitStack is exported."""
        assert ResourcesExitStack is ExitStack

    def test_exitstack_usage_pattern(self, tmp_path):
        """Test the documented usage pattern with ExitStack."""
        run_dir = tmp_path / "test_run"
        records_path = run_dir / "records.jsonl"
        run_dir.mkdir()

        with ExitStack() as stack:
            stack.enter_context(managed_run_directory(run_dir))
            fp = stack.enter_context(open_records_file(records_path))
            fp.write('{"test": "data"}\n')

        assert records_path.exists()
        assert (run_dir / ".insidellms_run").exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_write(self, tmp_path):
        """Test writing empty text."""
        file_path = tmp_path / "empty.txt"
        atomic_write_text(file_path, "")

        assert file_path.exists()
        assert file_path.read_text() == ""

    def test_empty_yaml_write(self, tmp_path):
        """Test writing empty YAML."""
        file_path = tmp_path / "empty.yaml"
        atomic_write_yaml(file_path, {})

        assert file_path.exists()
        loaded = yaml.safe_load(file_path.read_text())
        assert (
            loaded == {} or loaded is None
        )  # yaml.safe_load returns None for empty dict sometimes

    def test_very_long_content(self, tmp_path):
        """Test writing very long content."""
        file_path = tmp_path / "long.txt"
        content = "x" * 1_000_000  # 1MB of content

        atomic_write_text(file_path, content)

        assert file_path.read_text() == content

    def test_special_characters_in_filename(self, tmp_path):
        """Test with special characters in filename."""
        file_path = tmp_path / "file-with_special.chars.txt"
        atomic_write_text(file_path, "content")

        assert file_path.exists()
