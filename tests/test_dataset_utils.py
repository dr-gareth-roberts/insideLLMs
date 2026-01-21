"""Tests for insideLLMs/dataset_utils.py module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from insideLLMs.dataset_utils import (
    HF_DATASETS_AVAILABLE,
    load_csv_dataset,
    load_hf_dataset,
    load_jsonl_dataset,
)


class TestLoadCsvDataset:
    """Tests for load_csv_dataset function."""

    def test_load_csv_basic(self):
        """Test loading a basic CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,city\n")
            f.write("Alice,30,NYC\n")
            f.write("Bob,25,LA\n")
            path = f.name

        try:
            result = load_csv_dataset(path)
            assert len(result) == 2
            assert result[0]["name"] == "Alice"
            assert result[0]["age"] == "30"
            assert result[1]["name"] == "Bob"
        finally:
            Path(path).unlink()

    def test_load_csv_empty(self):
        """Test loading an empty CSV (headers only)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2,col3\n")
            path = f.name

        try:
            result = load_csv_dataset(path)
            assert result == []
        finally:
            Path(path).unlink()

    def test_load_csv_single_row(self):
        """Test loading a CSV with a single row."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            f.write("What is 2+2?,4\n")
            path = f.name

        try:
            result = load_csv_dataset(path)
            assert len(result) == 1
            assert result[0]["question"] == "What is 2+2?"
            assert result[0]["answer"] == "4"
        finally:
            Path(path).unlink()


class TestLoadJsonlDataset:
    """Tests for load_jsonl_dataset function."""

    def test_load_jsonl_basic(self):
        """Test loading a basic JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": 1, "text": "Hello"}\n')
            f.write('{"id": 2, "text": "World"}\n')
            path = f.name

        try:
            result = load_jsonl_dataset(path)
            assert len(result) == 2
            assert result[0]["id"] == 1
            assert result[0]["text"] == "Hello"
            assert result[1]["id"] == 2
        finally:
            Path(path).unlink()

    def test_load_jsonl_empty(self):
        """Test loading an empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            path = f.name

        try:
            result = load_jsonl_dataset(path)
            assert result == []
        finally:
            Path(path).unlink()

    def test_load_jsonl_nested(self):
        """Test loading JSONL with nested data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"user": {"name": "Alice"}, "scores": [1, 2, 3]}\n')
            path = f.name

        try:
            result = load_jsonl_dataset(path)
            assert len(result) == 1
            assert result[0]["user"]["name"] == "Alice"
            assert result[0]["scores"] == [1, 2, 3]
        finally:
            Path(path).unlink()


class TestLoadHfDataset:
    """Tests for load_hf_dataset function."""

    def test_hf_not_available(self):
        """Test that ImportError is raised when HF Datasets not installed."""
        with patch("insideLLMs.dataset_utils.HF_DATASETS_AVAILABLE", False):
            # Need to reimport to pick up the patched value
            from importlib import reload
            import insideLLMs.dataset_utils as du
            reload(du)

            if not du.HF_DATASETS_AVAILABLE:
                with pytest.raises(ImportError, match="HuggingFace Datasets"):
                    du.load_hf_dataset("test_dataset")

    @pytest.mark.skipif(not HF_DATASETS_AVAILABLE, reason="HF Datasets not installed")
    def test_hf_dataset_mocked(self):
        """Test loading HF dataset with mocked library."""
        with patch("insideLLMs.dataset_utils.load_dataset") as mock_load:
            mock_dataset = [
                {"text": "sample 1"},
                {"text": "sample 2"},
            ]
            mock_load.return_value = mock_dataset

            result = load_hf_dataset("test_dataset", split="train")

            mock_load.assert_called_once_with("test_dataset", split="train")
            assert len(result) == 2
            assert result[0]["text"] == "sample 1"


class TestHfDatasetsAvailable:
    """Tests for HF_DATASETS_AVAILABLE flag."""

    def test_flag_is_boolean(self):
        """Test that the flag is a boolean."""
        assert isinstance(HF_DATASETS_AVAILABLE, bool)
