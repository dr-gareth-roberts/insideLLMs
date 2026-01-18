"""Dataset loader utility for CSV, JSONL, and HuggingFace Datasets."""

import csv
import json
from typing import Any, Optional

try:
    from datasets import load_dataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def load_csv_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def load_jsonl_dataset(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_hf_dataset(
    dataset_name: str, split: str = "test", **kwargs
) -> Optional[list[dict[str, Any]]]:
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace Datasets not installed.")
    ds = load_dataset(dataset_name, split=split, **kwargs)
    return [dict(x) for x in ds]
