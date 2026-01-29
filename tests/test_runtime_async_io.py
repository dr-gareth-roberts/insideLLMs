"""Tests for insideLLMs/runtime/async_io.py module.

This module tests the async file I/O utilities used for non-blocking
record writing in the runner.
"""

import asyncio
from pathlib import Path

import pytest


class TestAsyncWriteText:
    """Tests for async_write_text function."""

    @pytest.mark.asyncio
    async def test_write_text_creates_file(self, tmp_path: Path):
        """Test that async_write_text creates a new file."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "test.txt"
        await async_write_text(filepath, "Hello, World!")

        assert filepath.exists()
        assert filepath.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_text_append_mode(self, tmp_path: Path):
        """Test appending to existing file."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "append.txt"
        filepath.write_text("First line\n")

        await async_write_text(filepath, "Second line\n", mode="a")

        content = filepath.read_text()
        assert "First line" in content
        assert "Second line" in content

    @pytest.mark.asyncio
    async def test_write_text_overwrite_mode(self, tmp_path: Path):
        """Test overwriting existing file."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "overwrite.txt"
        filepath.write_text("Original content")

        await async_write_text(filepath, "New content", mode="w")

        assert filepath.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_text_exclusive_mode(self, tmp_path: Path):
        """Test exclusive create mode fails if file exists."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "exclusive.txt"
        filepath.write_text("Existing content")

        with pytest.raises(FileExistsError):
            await async_write_text(filepath, "New content", mode="x")

    @pytest.mark.asyncio
    async def test_write_text_exclusive_mode_creates_new(self, tmp_path: Path):
        """Test exclusive create mode works for new files."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "new_exclusive.txt"

        await async_write_text(filepath, "Content", mode="x")

        assert filepath.exists()
        assert filepath.read_text() == "Content"

    @pytest.mark.asyncio
    async def test_write_text_custom_encoding(self, tmp_path: Path):
        """Test writing with custom encoding."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "encoded.txt"
        content = "Hello, 世界! 🌍"

        await async_write_text(filepath, content, encoding="utf-8")

        assert filepath.read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_write_text_invalid_path_raises(self, tmp_path: Path):
        """Test that writing to invalid path raises OSError."""
        from insideLLMs.runtime.async_io import async_write_text

        # Try to write to a non-existent directory
        invalid_path = tmp_path / "nonexistent" / "file.txt"

        with pytest.raises((IOError, OSError)):
            await async_write_text(invalid_path, "content")

    @pytest.mark.asyncio
    async def test_write_text_concurrent_writes(self, tmp_path: Path):
        """Test multiple concurrent writes to different files."""
        from insideLLMs.runtime.async_io import async_write_text

        files = [tmp_path / f"file_{i}.txt" for i in range(10)]

        async def write_file(filepath: Path, content: str):
            await async_write_text(filepath, content, mode="w")

        await asyncio.gather(*[write_file(f, f"Content for {f.name}") for f in files])

        for f in files:
            assert f.exists()
            assert f"Content for {f.name}" in f.read_text()

    @pytest.mark.asyncio
    async def test_write_text_empty_content(self, tmp_path: Path):
        """Test writing empty content creates empty file."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "empty.txt"
        await async_write_text(filepath, "", mode="w")

        assert filepath.exists()
        assert filepath.read_text() == ""


class TestAsyncWriteLines:
    """Tests for async_write_lines function."""

    @pytest.mark.asyncio
    async def test_write_lines_basic(self, tmp_path: Path):
        """Test writing multiple lines."""
        from insideLLMs.runtime.async_io import async_write_lines

        filepath = tmp_path / "lines.txt"
        lines = ["Line 1", "Line 2", "Line 3"]

        await async_write_lines(filepath, lines, mode="w")

        content = filepath.read_text()
        assert "Line 1\n" in content
        assert "Line 2\n" in content
        assert "Line 3\n" in content

    @pytest.mark.asyncio
    async def test_write_lines_adds_newlines(self, tmp_path: Path):
        """Test that newlines are added if missing."""
        from insideLLMs.runtime.async_io import async_write_lines

        filepath = tmp_path / "newlines.txt"
        lines = ["No newline", "Also no newline"]

        await async_write_lines(filepath, lines, mode="w")

        content = filepath.read_text()
        lines_in_file = content.strip().split("\n")
        assert len(lines_in_file) == 2

    @pytest.mark.asyncio
    async def test_write_lines_preserves_existing_newlines(self, tmp_path: Path):
        """Test that existing newlines are not doubled."""
        from insideLLMs.runtime.async_io import async_write_lines

        filepath = tmp_path / "preserved.txt"
        lines = ["With newline\n", "Without newline"]

        await async_write_lines(filepath, lines, mode="w")

        content = filepath.read_text()
        # Should not have double newlines
        assert "\n\n" not in content

    @pytest.mark.asyncio
    async def test_write_lines_append_mode(self, tmp_path: Path):
        """Test appending lines to existing file."""
        from insideLLMs.runtime.async_io import async_write_lines

        filepath = tmp_path / "append_lines.txt"
        filepath.write_text("Existing line\n")

        await async_write_lines(filepath, ["New line 1", "New line 2"], mode="a")

        content = filepath.read_text()
        assert "Existing line" in content
        assert "New line 1" in content
        assert "New line 2" in content

    @pytest.mark.asyncio
    async def test_write_lines_empty_list(self, tmp_path: Path):
        """Test writing empty list creates file but no content."""
        from insideLLMs.runtime.async_io import async_write_lines

        filepath = tmp_path / "empty_lines.txt"
        await async_write_lines(filepath, [], mode="w")

        assert filepath.exists()
        assert filepath.read_text() == ""

    @pytest.mark.asyncio
    async def test_write_lines_jsonl_format(self, tmp_path: Path):
        """Test writing JSONL-style content."""
        import json

        from insideLLMs.runtime.async_io import async_write_lines

        filepath = tmp_path / "records.jsonl"
        records = [
            json.dumps({"id": 1, "data": "first"}),
            json.dumps({"id": 2, "data": "second"}),
        ]

        await async_write_lines(filepath, records, mode="w")

        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[1])["id"] == 2

    @pytest.mark.asyncio
    async def test_write_lines_invalid_path_raises(self, tmp_path: Path):
        """Test that writing to invalid path raises OSError."""
        from insideLLMs.runtime.async_io import async_write_lines

        invalid_path = tmp_path / "nonexistent" / "file.txt"

        with pytest.raises((IOError, OSError)):
            await async_write_lines(invalid_path, ["content"])

    @pytest.mark.asyncio
    async def test_write_lines_concurrent(self, tmp_path: Path):
        """Test concurrent line writes to different files."""
        from insideLLMs.runtime.async_io import async_write_lines

        files = [tmp_path / f"lines_{i}.txt" for i in range(5)]

        async def write_to_file(filepath: Path, idx: int):
            lines = [f"File {idx} line {j}" for j in range(3)]
            await async_write_lines(filepath, lines, mode="w")

        await asyncio.gather(*[write_to_file(f, i) for i, f in enumerate(files)])

        for i, f in enumerate(files):
            content = f.read_text()
            assert f"File {i} line 0" in content
            assert f"File {i} line 2" in content


class TestAsyncIOIntegration:
    """Integration tests for async I/O utilities."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_high_volume_writes(self, tmp_path: Path):
        """Test handling many concurrent write operations."""
        from insideLLMs.runtime.async_io import async_write_text

        num_files = 50

        async def write_numbered_file(i: int):
            filepath = tmp_path / f"file_{i:04d}.txt"
            await async_write_text(filepath, f"Content {i}", mode="w")
            return filepath

        paths = await asyncio.gather(*[write_numbered_file(i) for i in range(num_files)])

        assert len(paths) == num_files
        for i, p in enumerate(paths):
            assert p.exists()
            assert p.read_text() == f"Content {i}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_alternating_append_and_read(self, tmp_path: Path):
        """Test file consistency with alternating appends."""
        from insideLLMs.runtime.async_io import async_write_text

        filepath = tmp_path / "alternating.txt"
        await async_write_text(filepath, "", mode="w")  # Create empty

        for i in range(20):
            await async_write_text(filepath, f"Line {i}\n", mode="a")

        content = filepath.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 20
        assert lines[0] == "Line 0"
        assert lines[19] == "Line 19"
