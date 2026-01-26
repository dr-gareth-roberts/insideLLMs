"""Dataset loader utility for CSV, JSONL, and HuggingFace Datasets.

This module provides unified interfaces for loading datasets from various sources,
including local CSV files, JSONL (JSON Lines) files, and HuggingFace Datasets Hub.
All loaders return data in a consistent format: a list of dictionaries where each
dictionary represents a single data record.

Module Overview
---------------
The module offers three primary loading functions:

- :func:`load_csv_dataset`: Load tabular data from CSV files
- :func:`load_jsonl_dataset`: Load structured data from JSONL files
- :func:`load_hf_dataset`: Load datasets from HuggingFace Datasets Hub

Examples
--------
Loading a local CSV file containing prompts:

>>> from insideLLMs.dataset_utils import load_csv_dataset
>>> data = load_csv_dataset("prompts.csv")
>>> print(data[0])
{'prompt': 'What is the capital of France?', 'category': 'geography'}

Loading a JSONL file with evaluation data:

>>> from insideLLMs.dataset_utils import load_jsonl_dataset
>>> data = load_jsonl_dataset("eval_data.jsonl")
>>> for record in data[:2]:
...     print(record['question'])
What is 2 + 2?
Who wrote Hamlet?

Loading a HuggingFace dataset for benchmarking:

>>> from insideLLMs.dataset_utils import load_hf_dataset
>>> data = load_hf_dataset("gsm8k", split="test", name="main")
>>> print(len(data))
1319

Combining multiple data sources:

>>> csv_data = load_csv_dataset("custom_prompts.csv")
>>> jsonl_data = load_jsonl_dataset("additional_prompts.jsonl")
>>> combined = csv_data + jsonl_data
>>> print(f"Total records: {len(combined)}")
Total records: 150

Notes
-----
- All functions return data as ``list[dict[str, Any]]`` for consistent handling
- HuggingFace Datasets support requires the ``datasets`` package to be installed
- CSV files are expected to have a header row defining column names
- JSONL files should contain one valid JSON object per line

See Also
--------
datasets.load_dataset : HuggingFace's native dataset loading function
csv.DictReader : Python's built-in CSV dictionary reader
"""

import csv
import json
from typing import Any, Optional

try:
    from datasets import load_dataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


def load_csv_dataset(path: str) -> list[dict[str, Any]]:
    """Load a dataset from a CSV file into a list of dictionaries.

    Reads a CSV file where the first row contains column headers and each
    subsequent row represents a data record. Each record is converted to
    a dictionary with column names as keys.

    Args:
        path: Path to the CSV file. Can be absolute or relative to the
            current working directory.

    Returns:
        A list of dictionaries where each dictionary represents a row
        from the CSV file. Keys are the column headers from the first row,
        and values are the corresponding cell values as strings.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be read due to permissions.

    Examples:
        Loading a simple prompts dataset:

        >>> data = load_csv_dataset("prompts.csv")
        >>> print(data[0])
        {'id': '1', 'prompt': 'Explain quantum computing', 'difficulty': 'hard'}

        Processing loaded data for model evaluation:

        >>> data = load_csv_dataset("evaluation_set.csv")
        >>> prompts = [record['prompt'] for record in data]
        >>> print(f"Loaded {len(prompts)} prompts for evaluation")
        Loaded 100 prompts for evaluation

        Loading a dataset with multiple columns:

        >>> data = load_csv_dataset("benchmark.csv")
        >>> for record in data[:3]:
        ...     print(f"Category: {record['category']}, Prompt: {record['prompt'][:30]}...")
        Category: math, Prompt: What is the integral of x^2...
        Category: coding, Prompt: Write a Python function that...
        Category: reasoning, Prompt: If all cats are mammals and...

        Handling datasets with special characters (quotes, commas):

        >>> data = load_csv_dataset("complex_prompts.csv")
        >>> # CSV quoting handles embedded commas and quotes automatically
        >>> print(data[0]['prompt'])
        Please explain "machine learning" in simple terms

    Notes:
        - All values are returned as strings; numeric conversion must be
          done manually if needed
        - The CSV file must have a header row; files without headers will
          use the first data row as column names
        - Empty cells are returned as empty strings, not None
        - Uses Python's csv.DictReader which handles quoted fields correctly

    See Also:
        load_jsonl_dataset: For loading JSON Lines formatted files
        load_hf_dataset: For loading HuggingFace Datasets
    """
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def load_jsonl_dataset(path: str) -> list[dict[str, Any]]:
    """Load a dataset from a JSONL (JSON Lines) file into a list of dictionaries.

    Reads a JSONL file where each line contains a valid JSON object. This format
    is commonly used for large datasets as it allows for streaming and line-by-line
    processing without loading the entire file into memory for parsing.

    Args:
        path: Path to the JSONL file. Can be absolute or relative to the
            current working directory.

    Returns:
        A list of dictionaries where each dictionary represents a parsed
        JSON object from a single line of the file. Values retain their
        JSON types (strings, numbers, booleans, nested objects, arrays).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the file cannot be read due to permissions.
        json.JSONDecodeError: If any line contains invalid JSON.

    Examples:
        Loading a basic JSONL dataset:

        >>> data = load_jsonl_dataset("prompts.jsonl")
        >>> print(data[0])
        {'id': 1, 'prompt': 'What is machine learning?', 'expected_topics': ['AI', 'ML']}

        Processing a dataset with nested structures:

        >>> data = load_jsonl_dataset("conversations.jsonl")
        >>> for record in data[:2]:
        ...     messages = record['messages']
        ...     print(f"Conversation with {len(messages)} messages")
        Conversation with 3 messages
        Conversation with 5 messages

        Loading evaluation data with ground truth:

        >>> data = load_jsonl_dataset("eval_qa.jsonl")
        >>> for item in data[:3]:
        ...     print(f"Q: {item['question'][:40]}... A: {item['answer'][:20]}...")
        Q: What is the capital of France?... A: Paris is the capital...
        Q: Explain the theory of relativity... A: Einstein's theory of...
        Q: How do neural networks learn?... A: Neural networks learn...

        Working with typed data (numbers, booleans preserved):

        >>> data = load_jsonl_dataset("benchmark_results.jsonl")
        >>> scores = [record['score'] for record in data]
        >>> print(f"Average score: {sum(scores) / len(scores):.2f}")
        Average score: 0.85

    Notes:
        - Unlike CSV, JSONL preserves data types (integers, floats, booleans,
          nested objects, and arrays)
        - Each line must be a complete, valid JSON object
        - Empty lines in the file will cause a JSONDecodeError
        - The entire file is loaded into memory; for very large files, consider
          using a streaming approach
        - Commonly used formats like OpenAI's fine-tuning format use JSONL

    See Also:
        load_csv_dataset: For loading CSV formatted files
        load_hf_dataset: For loading HuggingFace Datasets
    """
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{e.msg} (line {line_no} in {path})",
                    e.doc,
                    e.pos,
                ) from e
    return items


def load_hf_dataset(
    dataset_name: str, split: str = "test", **kwargs
) -> Optional[list[dict[str, Any]]]:
    """Load a dataset from the HuggingFace Datasets Hub into a list of dictionaries.

    Downloads and loads a dataset from the HuggingFace Hub, converting it to the
    same list-of-dictionaries format used by the other loaders in this module.
    This provides access to thousands of publicly available datasets for NLP,
    computer vision, and other machine learning tasks.

    Args:
        dataset_name: The name of the dataset on HuggingFace Hub. Can be in
            the format "dataset_name" for official datasets or "user/dataset"
            for community datasets.
        split: The dataset split to load. Common values are "train", "test",
            "validation". Defaults to "test".
        **kwargs: Additional keyword arguments passed to ``datasets.load_dataset``.
            Common options include:
            - name (str): Configuration/subset name for datasets with multiple configs
            - data_dir (str): Path to local data files
            - data_files (str or list): Specific files to load
            - cache_dir (str): Custom cache directory
            - token (str or bool): HuggingFace token for private datasets
            - trust_remote_code (bool): Allow running dataset scripts

    Returns:
        A list of dictionaries where each dictionary represents a single
        example from the dataset. Returns None only if the dataset is empty.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
        ValueError: If the dataset or split does not exist.
        ConnectionError: If unable to connect to HuggingFace Hub.

    Examples:
        Loading a popular benchmark dataset:

        >>> data = load_hf_dataset("gsm8k", split="test", name="main")
        >>> print(f"Loaded {len(data)} examples")
        Loaded 1319 examples
        >>> print(data[0].keys())
        dict_keys(['question', 'answer'])

        Loading a specific configuration of a dataset:

        >>> data = load_hf_dataset("glue", split="validation", name="mrpc")
        >>> print(f"MRPC validation set: {len(data)} examples")
        MRPC validation set: 408 examples
        >>> print(data[0])
        {'sentence1': '...', 'sentence2': '...', 'label': 1, 'idx': 0}

        Loading a subset of a large dataset:

        >>> data = load_hf_dataset(
        ...     "openai/gsm8k",
        ...     split="train[:100]",  # First 100 examples only
        ...     name="main"
        ... )
        >>> print(f"Loaded {len(data)} training examples")
        Loaded 100 training examples

        Loading a community dataset with authentication:

        >>> data = load_hf_dataset(
        ...     "username/private-dataset",
        ...     split="test",
        ...     token="hf_..."  # Your HuggingFace token
        ... )
        >>> print(f"Private dataset: {len(data)} examples")
        Private dataset: 500 examples

    Notes:
        - Requires the ``datasets`` package: ``pip install datasets``
        - Datasets are cached locally after first download (~/.cache/huggingface/)
        - Split syntax supports slicing: "train[:1000]", "test[50%:]", etc.
        - Some datasets require accepting terms on the HuggingFace website first
        - The ``name`` parameter is required for datasets with multiple configurations
          (e.g., GLUE, SuperGLUE, many benchmark suites)
        - For very large datasets, consider using split slicing to load subsets

    See Also:
        load_csv_dataset: For loading local CSV files
        load_jsonl_dataset: For loading local JSONL files
        datasets.load_dataset: The underlying HuggingFace function with full options
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("HuggingFace Datasets not installed.")
    ds = load_dataset(dataset_name, split=split, **kwargs)
    return [dict(x) for x in ds]
