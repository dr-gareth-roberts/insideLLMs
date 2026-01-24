"""Resource management utilities for insideLLMs.

This module provides context managers and utilities for deterministic
resource cleanup, particularly for file handles and run directories.
It implements the RAII (Resource Acquisition Is Initialization) pattern
for Python, ensuring that resources are properly released even when
exceptions occur.

Overview
--------
The module centers around three main concerns:

1. **File Handle Management**: Safe opening and closing of files with
   guaranteed cleanup via context managers (`open_records_file`).

2. **Atomic File Operations**: Write operations that prevent partial
   writes from corrupting files (`atomic_write_text`, `atomic_write_yaml`).

3. **Run Directory Management**: Setup and tracking of experiment run
   directories with sentinel files for safety (`managed_run_directory`,
   `ensure_run_sentinel`).

The primary pattern uses `contextlib.ExitStack` for managing multiple
resources with guaranteed cleanup even when exceptions occur. This is
especially useful in LLM evaluation pipelines where you may have multiple
output files, log files, and temporary directories that all need proper
cleanup.

Key Patterns
------------
**Single Resource Management**::

    from insideLLMs.resources import open_records_file
    from pathlib import Path

    with open_records_file(Path("output/records.jsonl")) as fp:
        fp.write('{"prompt": "Hello", "response": "Hi"}\\n')
        fp.write('{"prompt": "How are you?", "response": "Good"}\\n')
    # File is automatically flushed and closed here

**Multiple Resource Management with ExitStack**::

    from contextlib import ExitStack
    from insideLLMs.resources import (
        open_records_file,
        managed_run_directory,
        atomic_write_yaml,
    )
    from pathlib import Path

    with ExitStack() as stack:
        # Setup run directory
        run_dir = stack.enter_context(
            managed_run_directory(Path("runs/experiment_001"))
        )

        # Open multiple output files
        records_fp = stack.enter_context(
            open_records_file(run_dir / "records.jsonl")
        )
        errors_fp = stack.enter_context(
            open_records_file(run_dir / "errors.jsonl")
        )

        # Process data - cleanup happens automatically on exit
        for item in dataset:
            try:
                result = process(item)
                records_fp.write(json.dumps(result) + "\\n")
            except Exception as e:
                errors_fp.write(json.dumps({"error": str(e)}) + "\\n")

    # All files closed, even if an exception occurred

**Atomic File Writes**::

    from insideLLMs.resources import atomic_write_yaml
    from pathlib import Path

    config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    # This write is atomic - either fully succeeds or leaves original intact
    atomic_write_yaml(Path("config.yaml"), config)

Thread Safety
-------------
The context managers in this module are NOT thread-safe. Each thread should
use its own file handles and run directories. For concurrent workloads,
create separate resources per worker.

Error Handling
--------------
All context managers guarantee cleanup even when exceptions occur:

- `open_records_file`: Always flushes and closes the file
- `managed_run_directory`: Directory remains in valid state
- Atomic writes: Either succeed completely or leave target unchanged

Notes
-----
- The sentinel file (`.insidellms_run`) is used by the `--overwrite` CLI
  flag to safely identify legitimate run directories before deletion.
- Atomic writes use a temporary file with `os.replace()` for atomicity.
- File handles use explicit flush before close for data integrity.

See Also
--------
contextlib.ExitStack : For managing multiple context managers together.
insideLLMs.experiment_tracking : For higher-level experiment management.
insideLLMs.runtime.pipeline : For LLM pipeline execution with resources.

Examples
--------
Basic file writing with guaranteed cleanup:

>>> from insideLLMs.resources import open_records_file
>>> from pathlib import Path
>>> import tempfile
>>> import os
>>>
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir) / "records.jsonl"
...     with open_records_file(path) as fp:
...         fp.write('{"id": 1, "result": "success"}\\n')
...         fp.write('{"id": 2, "result": "success"}\\n')
...     # Verify file was written
...     print(path.read_text())
{"id": 1, "result": "success"}
{"id": 2, "result": "success"}

Setting up a complete experiment run:

>>> from insideLLMs.resources import managed_run_directory, atomic_write_yaml
>>> from pathlib import Path
>>> import tempfile
>>>
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     run_dir = Path(tmpdir) / "run_001"
...     with managed_run_directory(run_dir) as rd:
...         # Directory is created with sentinel
...         assert rd.exists()
...         assert (rd / ".insidellms_run").exists()
...
...         # Save configuration atomically
...         config = {"model": "gpt-4", "probe": "factuality"}
...         atomic_write_yaml(rd / "config.yaml", config)
...
...         # Config file exists
...         assert (rd / "config.yaml").exists()

Using ExitStack for complex resource management:

>>> from contextlib import ExitStack
>>> from insideLLMs.resources import open_records_file, managed_run_directory
>>> from pathlib import Path
>>> import tempfile
>>> import json
>>>
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     with ExitStack() as stack:
...         # Setup run directory
...         run_dir = stack.enter_context(
...             managed_run_directory(Path(tmpdir) / "run")
...         )
...
...         # Open output files
...         results_fp = stack.enter_context(
...             open_records_file(run_dir / "results.jsonl")
...         )
...         metrics_fp = stack.enter_context(
...             open_records_file(run_dir / "metrics.jsonl")
...         )
...
...         # Write some data
...         results_fp.write(json.dumps({"status": "ok"}) + "\\n")
...         metrics_fp.write(json.dumps({"latency_ms": 150}) + "\\n")
...
...     # All resources cleaned up
...     assert (Path(tmpdir) / "run" / "results.jsonl").exists()
"""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, Any, Iterator

import yaml


@contextmanager
def open_records_file(
    path: Path,
    mode: str = "x",
    encoding: str = "utf-8",
) -> Iterator[IO[str]]:
    """Open a records file with guaranteed cleanup.

    This context manager provides safe file handling for JSONL (JSON Lines)
    record files commonly used in LLM evaluation pipelines. It ensures that
    files are always properly flushed and closed, even when exceptions occur.

    The default mode "x" (exclusive creation) prevents accidentally overwriting
    existing files, which is important for preserving evaluation results.

    Parameters
    ----------
    path : Path
        Path to the records file. Parent directories must exist.
        Typically uses `.jsonl` extension for JSON Lines format.
    mode : str, default="x"
        File open mode. Common values:

        - "x": Exclusive creation (fails if file exists) - recommended for
          new runs to prevent accidental overwrites.
        - "w": Write mode (truncates existing file) - use with caution.
        - "a": Append mode - useful for resuming interrupted runs.

    encoding : str, default="utf-8"
        Character encoding for the file. UTF-8 is strongly recommended
        for compatibility with JSON and international text.

    Yields
    ------
    IO[str]
        A text file handle for writing records. Supports standard file
        operations like `write()`, `writelines()`, and `flush()`.

    Raises
    ------
    FileExistsError
        If mode is "x" (exclusive creation) and the file already exists.
        This is a safety feature to prevent overwriting existing results.
    FileNotFoundError
        If the parent directory does not exist.
    PermissionError
        If the user lacks write permissions to the directory.
    OSError
        For other I/O errors (disk full, invalid path, etc.).

    See Also
    --------
    atomic_write_text : For atomic single-write operations.
    managed_run_directory : For managing the parent run directory.
    contextlib.ExitStack : For managing multiple file handles together.

    Notes
    -----
    - The file is automatically flushed before closing to ensure all
      data is written to disk.
    - Close errors are silently suppressed to ensure cleanup completes.
    - For atomic writes (write-or-fail-completely), use `atomic_write_text`
      instead.
    - This function is NOT thread-safe. Use separate file handles per thread.

    Examples
    --------
    Basic usage - writing evaluation records:

    >>> from insideLLMs.resources import open_records_file
    >>> from pathlib import Path
    >>> import tempfile
    >>> import json
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "results.jsonl"
    ...     with open_records_file(path) as fp:
    ...         record = {"prompt": "What is 2+2?", "response": "4", "correct": True}
    ...         fp.write(json.dumps(record) + "\\n")
    ...         record = {"prompt": "Capital of France?", "response": "Paris", "correct": True}
    ...         fp.write(json.dumps(record) + "\\n")
    ...
    ...     # Verify records were written
    ...     lines = path.read_text().strip().split("\\n")
    ...     print(f"Wrote {len(lines)} records")
    Wrote 2 records

    Handling the FileExistsError for safe reruns:

    >>> from insideLLMs.resources import open_records_file
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "results.jsonl"
    ...
    ...     # First run succeeds
    ...     with open_records_file(path) as fp:
    ...         fp.write('{"run": 1}\\n')
    ...
    ...     # Second run fails safely (file exists)
    ...     try:
    ...         with open_records_file(path) as fp:
    ...             fp.write('{"run": 2}\\n')
    ...     except FileExistsError:
    ...         print("File already exists - use mode='a' to append or delete first")
    File already exists - use mode='a' to append or delete first

    Appending to an existing file for resumed runs:

    >>> from insideLLMs.resources import open_records_file
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "results.jsonl"
    ...
    ...     # Initial records
    ...     with open_records_file(path) as fp:
    ...         fp.write('{"batch": 1}\\n')
    ...
    ...     # Resume with append mode
    ...     with open_records_file(path, mode="a") as fp:
    ...         fp.write('{"batch": 2}\\n')
    ...
    ...     lines = path.read_text().strip().split("\\n")
    ...     print(f"Total records: {len(lines)}")
    Total records: 2

    Using with ExitStack for multiple files:

    >>> from contextlib import ExitStack
    >>> from insideLLMs.resources import open_records_file
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     with ExitStack() as stack:
    ...         results = stack.enter_context(
    ...             open_records_file(Path(tmpdir) / "results.jsonl")
    ...         )
    ...         errors = stack.enter_context(
    ...             open_records_file(Path(tmpdir) / "errors.jsonl")
    ...         )
    ...
    ...         # Write to both files
    ...         results.write('{"status": "success"}\\n')
    ...         errors.write('{"error": "timeout", "retry": True}\\n')
    ...
    ...     # Both files properly closed
    ...     print("Results:", (Path(tmpdir) / "results.jsonl").exists())
    ...     print("Errors:", (Path(tmpdir) / "errors.jsonl").exists())
    Results: True
    Errors: True

    Exception handling - file still closed on error:

    >>> from insideLLMs.resources import open_records_file
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir) / "results.jsonl"
    ...     try:
    ...         with open_records_file(path) as fp:
    ...             fp.write('{"before_error": true}\\n')
    ...             raise ValueError("Simulated processing error")
    ...     except ValueError:
    ...         pass  # Handle error
    ...
    ...     # File was still properly closed and data flushed
    ...     content = path.read_text()
    ...     print("Data preserved:", "before_error" in content)
    Data preserved: True
    """
    fp = open(path, mode, encoding=encoding)
    try:
        yield fp
    finally:
        try:
            fp.flush()
            fp.close()
        except Exception:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically.

    This function implements atomic file writes using the write-to-temp-then-rename
    pattern. It ensures that the target file is either fully written with the new
    content or left unchanged - there is no intermediate state where the file
    contains partial data.

    This is critical for configuration files, state files, and any data that
    must maintain consistency even during crashes or power failures.

    Parameters
    ----------
    path : Path
        Target file path. The parent directory will be created if it doesn't
        exist. The file will be created if it doesn't exist, or replaced if
        it does.
    text : str
        Text content to write to the file. Will be encoded as UTF-8.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the write operation fails (disk full, permissions, etc.).
        In this case, the original file (if any) is left unchanged.
    TypeError
        If `text` is not a string.

    See Also
    --------
    atomic_write_yaml : Atomic write with YAML serialization.
    open_records_file : For streaming writes to record files.

    Notes
    -----
    The atomicity is achieved through the following steps:

    1. Create a temporary file in the same directory as the target
       (using a `.filename.tmp` naming pattern).
    2. Write all content to the temporary file.
    3. Flush the file buffer to the OS.
    4. Call `fsync()` to ensure data is written to disk (best-effort).
    5. Atomically replace the target with the temporary file using
       `os.replace()`.

    The `os.replace()` operation is atomic on POSIX systems and on
    Windows (NTFS). This means the target file will never be in a
    partially-written state.

    If the write fails at any step before the replace, the original
    file is left unchanged and the temporary file is abandoned.

    Thread Safety: This function is thread-safe as long as different
    threads write to different files. Concurrent writes to the same
    file will result in a race condition where the last writer wins.

    Examples
    --------
    Basic usage - writing a configuration file:

    >>> from insideLLMs.resources import atomic_write_text
    >>> from pathlib import Path
    >>> import tempfile
    >>> import json
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     config_path = Path(tmpdir) / "config.json"
    ...
    ...     # Write configuration atomically
    ...     config = {"model": "gpt-4", "temperature": 0.7}
    ...     atomic_write_text(config_path, json.dumps(config, indent=2))
    ...
    ...     # Read it back
    ...     loaded = json.loads(config_path.read_text())
    ...     print(f"Model: {loaded['model']}")
    Model: gpt-4

    Creating parent directories automatically:

    >>> from insideLLMs.resources import atomic_write_text
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Nested path - directories created automatically
    ...     nested_path = Path(tmpdir) / "deep" / "nested" / "file.txt"
    ...     atomic_write_text(nested_path, "Hello from nested file!")
    ...
    ...     print("File exists:", nested_path.exists())
    ...     print("Content:", nested_path.read_text())
    File exists: True
    Content: Hello from nested file!

    Updating a file safely (original preserved on failure):

    >>> from insideLLMs.resources import atomic_write_text
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     state_path = Path(tmpdir) / "state.json"
    ...
    ...     # Initial state
    ...     atomic_write_text(state_path, '{"count": 0}')
    ...
    ...     # Update state
    ...     atomic_write_text(state_path, '{"count": 1}')
    ...
    ...     # Update again
    ...     atomic_write_text(state_path, '{"count": 2}')
    ...
    ...     # Only the final state exists
    ...     print("Final state:", state_path.read_text())
    Final state: {"count": 2}

    Saving experiment results:

    >>> from insideLLMs.resources import atomic_write_text
    >>> from pathlib import Path
    >>> import tempfile
    >>> import json
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     results_path = Path(tmpdir) / "experiment_results.json"
    ...
    ...     results = {
    ...         "experiment_id": "exp_001",
    ...         "model": "gpt-4",
    ...         "metrics": {
    ...             "accuracy": 0.95,
    ...             "latency_ms": 150.3,
    ...             "tokens": 1234
    ...         },
    ...         "status": "completed"
    ...     }
    ...
    ...     atomic_write_text(results_path, json.dumps(results, indent=2))
    ...     print("Results saved successfully")
    Results saved successfully

    Using in a try/except for robust error handling:

    >>> from insideLLMs.resources import atomic_write_text
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> def save_checkpoint(path: Path, data: dict) -> bool:
    ...     '''Save checkpoint with error handling.'''
    ...     import json
    ...     try:
    ...         atomic_write_text(path, json.dumps(data))
    ...         return True
    ...     except OSError as e:
    ...         print(f"Failed to save checkpoint: {e}")
    ...         return False
    ...
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     checkpoint_path = Path(tmpdir) / "checkpoint.json"
    ...     success = save_checkpoint(checkpoint_path, {"epoch": 10, "loss": 0.05})
    ...     print(f"Checkpoint saved: {success}")
    Checkpoint saved: True
    """
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def atomic_write_yaml(path: Path, data: Any, serializer: Any = None) -> None:
    """Write data to a YAML file atomically.

    This function combines YAML serialization with atomic file writing to
    safely persist configuration data, experiment parameters, and other
    structured data. The file is either fully written or left unchanged.

    YAML format is preferred for human-readable configuration files because
    it supports comments, multi-line strings, and is more readable than JSON
    for complex nested structures.

    Parameters
    ----------
    path : Path
        Target file path for the YAML output. The parent directory will be
        created if it doesn't exist. Typically uses `.yaml` or `.yml` extension.
    data : Any
        Data to serialize as YAML. Can be any YAML-serializable type:

        - dict: Key-value mappings
        - list: Sequences
        - str, int, float, bool: Scalar values
        - None: Null values
        - Nested combinations of the above

    serializer : Callable[[Any], Any], optional
        Optional function to transform data before YAML serialization.
        Useful for converting dataclasses, Pydantic models, or other
        custom objects to dictionaries. If provided, the function is
        called as `serializer(data)` and its return value is serialized.

    Returns
    -------
    None

    Raises
    ------
    yaml.YAMLError
        If the data cannot be serialized to YAML.
    OSError
        If the file write operation fails.
    TypeError
        If `serializer` is provided but is not callable.

    See Also
    --------
    atomic_write_text : For writing raw text atomically.
    yaml.safe_dump : The underlying YAML serialization function.

    Notes
    -----
    - Uses `yaml.safe_dump` which only serializes standard Python types.
      For custom objects, provide a `serializer` function.
    - Output is formatted with `sort_keys=False` to preserve insertion order,
      `default_flow_style=False` for readable block style, and
      `allow_unicode=True` for international text support.
    - The atomicity guarantees are the same as `atomic_write_text`.

    Examples
    --------
    Basic usage - writing a configuration file:

    >>> from insideLLMs.resources import atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     config_path = Path(tmpdir) / "config.yaml"
    ...
    ...     config = {
    ...         "model": "gpt-4",
    ...         "parameters": {
    ...             "temperature": 0.7,
    ...             "max_tokens": 1000,
    ...             "top_p": 0.95
    ...         },
    ...         "probes": ["factuality", "bias", "toxicity"]
    ...     }
    ...
    ...     atomic_write_yaml(config_path, config)
    ...
    ...     # Read and display the YAML
    ...     content = config_path.read_text()
    ...     print(content.strip().split("\\n")[0])  # First line
    model: gpt-4

    Saving experiment metadata:

    >>> from insideLLMs.resources import atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>> from datetime import datetime
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     meta_path = Path(tmpdir) / "metadata.yaml"
    ...
    ...     metadata = {
    ...         "experiment_id": "exp_20240115_001",
    ...         "created_at": "2024-01-15T10:30:00",
    ...         "model_info": {
    ...             "name": "gpt-4",
    ...             "provider": "openai",
    ...             "version": "0613"
    ...         },
    ...         "dataset": {
    ...             "name": "factuality_bench",
    ...             "size": 1000,
    ...             "split": "test"
    ...         },
    ...         "status": "completed"
    ...     }
    ...
    ...     atomic_write_yaml(meta_path, metadata)
    ...     print("Metadata saved:", meta_path.exists())
    Metadata saved: True

    Using a custom serializer for dataclasses:

    >>> from insideLLMs.resources import atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>> from dataclasses import dataclass, asdict
    >>>
    >>> @dataclass
    ... class ModelConfig:
    ...     name: str
    ...     temperature: float
    ...     max_tokens: int
    ...
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     config_path = Path(tmpdir) / "model_config.yaml"
    ...
    ...     config = ModelConfig(name="gpt-4", temperature=0.7, max_tokens=1000)
    ...
    ...     # Use asdict as serializer to convert dataclass to dict
    ...     atomic_write_yaml(config_path, config, serializer=asdict)
    ...
    ...     content = config_path.read_text()
    ...     print("name:" in content)
    True

    Saving probe results with nested structures:

    >>> from insideLLMs.resources import atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     results_path = Path(tmpdir) / "probe_results.yaml"
    ...
    ...     results = {
    ...         "probe": "factuality",
    ...         "model": "gpt-4",
    ...         "metrics": {
    ...             "accuracy": 0.92,
    ...             "precision": 0.89,
    ...             "recall": 0.94,
    ...             "f1_score": 0.915
    ...         },
    ...         "per_category": {
    ...             "science": {"accuracy": 0.95, "count": 200},
    ...             "history": {"accuracy": 0.88, "count": 150},
    ...             "geography": {"accuracy": 0.93, "count": 175}
    ...         },
    ...         "errors": [
    ...             {"id": 42, "expected": "Paris", "got": "London"},
    ...             {"id": 87, "expected": "1969", "got": "1968"}
    ...         ]
    ...     }
    ...
    ...     atomic_write_yaml(results_path, results)
    ...     print("Results saved:", results_path.exists())
    Results saved: True

    Using with managed_run_directory for complete experiment setup:

    >>> from insideLLMs.resources import atomic_write_yaml, managed_run_directory
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     with managed_run_directory(Path(tmpdir) / "run_001") as run_dir:
    ...         # Save configuration
    ...         config = {"model": "gpt-4", "probe": "factuality"}
    ...         atomic_write_yaml(run_dir / "config.yaml", config)
    ...
    ...         # Save parameters
    ...         params = {"batch_size": 32, "temperature": 0.7}
    ...         atomic_write_yaml(run_dir / "params.yaml", params)
    ...
    ...         # Both files exist
    ...         print("Config:", (run_dir / "config.yaml").exists())
    ...         print("Params:", (run_dir / "params.yaml").exists())
    Config: True
    Params: True
    """
    if serializer is not None:
        data = serializer(data)
    content = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    atomic_write_text(path, content)


def ensure_run_sentinel(run_dir: Path) -> None:
    """Create a sentinel file marking a directory as an insideLLMs run.

    This function creates a hidden marker file (`.insidellms_run`) in the
    specified directory to identify it as an insideLLMs experiment run
    directory. This marker serves as a safety mechanism for destructive
    operations like the `--overwrite` CLI flag.

    The sentinel file pattern is commonly used in build systems and
    experiment tracking to distinguish managed directories from user
    directories, preventing accidental data loss.

    Parameters
    ----------
    run_dir : Path
        The run directory path where the sentinel file should be created.
        The directory must exist before calling this function.

    Returns
    -------
    None

    Raises
    ------
    No exceptions are raised. All errors are silently suppressed to ensure
    this function never disrupts the main workflow. This is intentional
    because the sentinel is a safety feature, not a critical requirement.

    See Also
    --------
    managed_run_directory : Context manager that calls this function.

    Notes
    -----
    - The sentinel file is named `.insidellms_run` (hidden on Unix systems).
    - The file contains the text "insideLLMs run directory\\n".
    - If the sentinel already exists, this function does nothing.
    - All write errors are silently suppressed (read-only filesystem, etc.).
    - The `--overwrite` CLI flag checks for this sentinel before deleting
      a directory, providing protection against accidentally deleting
      unrelated directories.

    Warning
    -------
    This function does NOT create the directory. The directory must exist
    before calling this function. Use `managed_run_directory` for complete
    directory setup with sentinel creation.

    Examples
    --------
    Basic usage - marking an existing directory:

    >>> from insideLLMs.resources import ensure_run_sentinel
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     run_dir = Path(tmpdir) / "my_run"
    ...     run_dir.mkdir()  # Directory must exist first
    ...
    ...     # Create sentinel
    ...     ensure_run_sentinel(run_dir)
    ...
    ...     # Verify sentinel exists
    ...     sentinel = run_dir / ".insidellms_run"
    ...     print("Sentinel exists:", sentinel.exists())
    ...     print("Content:", sentinel.read_text().strip())
    Sentinel exists: True
    Content: insideLLMs run directory

    Idempotent - safe to call multiple times:

    >>> from insideLLMs.resources import ensure_run_sentinel
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     run_dir = Path(tmpdir) / "my_run"
    ...     run_dir.mkdir()
    ...
    ...     # Call multiple times - no error
    ...     ensure_run_sentinel(run_dir)
    ...     ensure_run_sentinel(run_dir)
    ...     ensure_run_sentinel(run_dir)
    ...
    ...     # Still only one sentinel file
    ...     sentinel = run_dir / ".insidellms_run"
    ...     print("Sentinel exists:", sentinel.exists())
    Sentinel exists: True

    Checking for sentinel before destructive operations:

    >>> from insideLLMs.resources import ensure_run_sentinel
    >>> from pathlib import Path
    >>> import tempfile
    >>> import shutil
    >>>
    >>> def safe_delete_run(run_dir: Path) -> bool:
    ...     '''Only delete if it looks like a run directory.'''
    ...     sentinel = run_dir / ".insidellms_run"
    ...     if not sentinel.exists():
    ...         print(f"Warning: {run_dir} is not a run directory, skipping")
    ...         return False
    ...     shutil.rmtree(run_dir)
    ...     return True
    ...
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Create a run directory with sentinel
    ...     run_dir = Path(tmpdir) / "run_001"
    ...     run_dir.mkdir()
    ...     ensure_run_sentinel(run_dir)
    ...
    ...     # Create a regular directory without sentinel
    ...     other_dir = Path(tmpdir) / "my_data"
    ...     other_dir.mkdir()
    ...
    ...     # Safe delete protects the regular directory
    ...     print("Delete run_001:", safe_delete_run(run_dir))
    ...     print("Delete my_data:", safe_delete_run(other_dir))
    Delete run_001: True
    Delete my_data: Warning: ... is not a run directory, skipping
    False

    Integration with experiment setup:

    >>> from insideLLMs.resources import ensure_run_sentinel, atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> def setup_experiment_run(base_dir: Path, run_id: str, config: dict) -> Path:
    ...     '''Set up a new experiment run directory.'''
    ...     run_dir = base_dir / run_id
    ...     run_dir.mkdir(parents=True, exist_ok=True)
    ...
    ...     # Mark as run directory
    ...     ensure_run_sentinel(run_dir)
    ...
    ...     # Save configuration
    ...     atomic_write_yaml(run_dir / "config.yaml", config)
    ...
    ...     return run_dir
    ...
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     config = {"model": "gpt-4", "probe": "factuality"}
    ...     run_dir = setup_experiment_run(Path(tmpdir), "exp_001", config)
    ...
    ...     # Verify setup
    ...     print("Run directory:", run_dir.name)
    ...     print("Has sentinel:", (run_dir / ".insidellms_run").exists())
    ...     print("Has config:", (run_dir / "config.yaml").exists())
    Run directory: exp_001
    Has sentinel: True
    Has config: True
    """
    marker = run_dir / ".insidellms_run"
    if not marker.exists():
        try:
            marker.write_text("insideLLMs run directory\n", encoding="utf-8")
        except Exception:
            pass


@contextmanager
def managed_run_directory(
    run_dir: Path,
    *,
    create: bool = True,
    sentinel: bool = True,
) -> Iterator[Path]:
    """Context manager for a run directory with optional setup.

    This context manager provides a convenient way to set up and manage
    experiment run directories with proper initialization. It handles
    directory creation, sentinel file placement, and provides a clean
    interface for working with run directories.

    The context manager pattern ensures consistent setup regardless of
    how the code exits (normal completion, exception, or early return).

    Parameters
    ----------
    run_dir : Path
        Path to the run directory. This can be an absolute or relative path.
        If the directory doesn't exist and `create=True`, it will be created
        along with any necessary parent directories.
    create : bool, default=True
        If True, create the directory (and parent directories) if they don't
        exist. If False, the directory must already exist or a FileNotFoundError
        will be raised when attempting to use it.
    sentinel : bool, default=True
        If True, create the `.insidellms_run` sentinel file to mark this
        directory as an insideLLMs run directory. This is recommended for
        safety with the `--overwrite` CLI flag.

    Yields
    ------
    Path
        The run directory path, same as the input `run_dir` parameter.
        This allows using the path directly in the `with` statement.

    Raises
    ------
    OSError
        If directory creation fails (permissions, disk full, etc.) when
        `create=True`.
    FileNotFoundError
        If the directory doesn't exist and `create=False` (raised when
        attempting to use the directory, not immediately).

    See Also
    --------
    ensure_run_sentinel : Creates the sentinel file.
    open_records_file : For opening output files within the run directory.
    atomic_write_yaml : For saving configuration to the run directory.
    contextlib.ExitStack : For combining with other context managers.

    Notes
    -----
    - Directory creation uses `parents=True` and `exist_ok=True`, so it's
      safe to call even if the directory already exists.
    - The sentinel file is created after the directory, so partial setup
      is avoided.
    - Unlike some context managers, this one doesn't clean up the directory
      on exit. The run directory and its contents persist after the context
      manager exits.
    - For temporary directories that should be cleaned up, use
      `tempfile.TemporaryDirectory` instead.

    Examples
    --------
    Basic usage - create a run directory:

    >>> from insideLLMs.resources import managed_run_directory
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     with managed_run_directory(Path(tmpdir) / "run_001") as run_dir:
    ...         print("Run directory:", run_dir.name)
    ...         print("Exists:", run_dir.exists())
    ...         print("Has sentinel:", (run_dir / ".insidellms_run").exists())
    Run directory: run_001
    Exists: True
    Has sentinel: True

    Complete experiment setup with output files:

    >>> from contextlib import ExitStack
    >>> from insideLLMs.resources import managed_run_directory, open_records_file
    >>> from insideLLMs.resources import atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>> import json
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     with ExitStack() as stack:
    ...         # Setup run directory
    ...         run_dir = stack.enter_context(
    ...             managed_run_directory(Path(tmpdir) / "experiment_001")
    ...         )
    ...
    ...         # Save configuration
    ...         config = {
    ...             "model": "gpt-4",
    ...             "probe": "factuality",
    ...             "dataset_size": 1000
    ...         }
    ...         atomic_write_yaml(run_dir / "config.yaml", config)
    ...
    ...         # Open output files
    ...         results_fp = stack.enter_context(
    ...             open_records_file(run_dir / "results.jsonl")
    ...         )
    ...
    ...         # Write results
    ...         for i in range(3):
    ...             record = {"id": i, "correct": True}
    ...             results_fp.write(json.dumps(record) + "\\n")
    ...
    ...     # After context - verify files exist
    ...     print("Config exists:", (Path(tmpdir) / "experiment_001" / "config.yaml").exists())
    ...     print("Results exist:", (Path(tmpdir) / "experiment_001" / "results.jsonl").exists())
    Config exists: True
    Results exist: True

    Nested run directories for experiment variations:

    >>> from insideLLMs.resources import managed_run_directory, atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     base_dir = Path(tmpdir) / "experiments" / "factuality"
    ...
    ...     # Create multiple runs
    ...     for temp in [0.5, 0.7, 0.9]:
    ...         run_name = f"temp_{temp}"
    ...         with managed_run_directory(base_dir / run_name) as run_dir:
    ...             config = {"temperature": temp, "model": "gpt-4"}
    ...             atomic_write_yaml(run_dir / "config.yaml", config)
    ...
    ...     # List all runs
    ...     runs = list(base_dir.iterdir())
    ...     print(f"Created {len(runs)} runs")
    ...     print("Run names:", sorted(r.name for r in runs))
    Created 3 runs
    Run names: ['temp_0.5', 'temp_0.7', 'temp_0.9']

    Using with existing directory (no creation):

    >>> from insideLLMs.resources import managed_run_directory
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Pre-create the directory
    ...     run_dir = Path(tmpdir) / "existing_run"
    ...     run_dir.mkdir()
    ...
    ...     # Use without creating
    ...     with managed_run_directory(run_dir, create=False) as rd:
    ...         print("Same directory:", rd == run_dir)
    ...         print("Has sentinel:", (rd / ".insidellms_run").exists())
    Same directory: True
    Has sentinel: True

    Skipping sentinel for temporary or non-standard runs:

    >>> from insideLLMs.resources import managed_run_directory
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Debug run - no sentinel needed
    ...     with managed_run_directory(
    ...         Path(tmpdir) / "debug_run",
    ...         sentinel=False
    ...     ) as run_dir:
    ...         print("Directory exists:", run_dir.exists())
    ...         print("Has sentinel:", (run_dir / ".insidellms_run").exists())
    Directory exists: True
    Has sentinel: False

    Error handling for experiment setup:

    >>> from insideLLMs.resources import managed_run_directory, atomic_write_yaml
    >>> from pathlib import Path
    >>> import tempfile
    >>>
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     try:
    ...         with managed_run_directory(Path(tmpdir) / "run_001") as run_dir:
    ...             # Save initial config
    ...             atomic_write_yaml(run_dir / "config.yaml", {"status": "started"})
    ...
    ...             # Simulate error during experiment
    ...             raise ValueError("Simulated experiment error")
    ...
    ...     except ValueError as e:
    ...         print(f"Error occurred: {e}")
    ...
    ...     # Directory and files still exist after error
    ...     config_path = Path(tmpdir) / "run_001" / "config.yaml"
    ...     print("Config preserved:", config_path.exists())
    Error occurred: Simulated experiment error
    Config preserved: True

    Integration with benchmark workflow:

    >>> from insideLLMs.resources import managed_run_directory, atomic_write_yaml
    >>> from insideLLMs.resources import open_records_file
    >>> from pathlib import Path
    >>> import tempfile
    >>> import json
    >>> from datetime import datetime
    >>>
    >>> def run_benchmark(base_dir: Path, model: str, probe: str) -> Path:
    ...     '''Run a benchmark and save results.'''
    ...     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ...     run_id = f"{model}_{probe}_{timestamp}"
    ...
    ...     with managed_run_directory(base_dir / run_id) as run_dir:
    ...         # Save config
    ...         config = {"model": model, "probe": probe, "timestamp": timestamp}
    ...         atomic_write_yaml(run_dir / "config.yaml", config)
    ...
    ...         # Simulate benchmark results
    ...         with open_records_file(run_dir / "results.jsonl") as fp:
    ...             for i in range(5):
    ...                 result = {"item": i, "score": 0.9 + i * 0.02}
    ...                 fp.write(json.dumps(result) + "\\n")
    ...
    ...         # Save summary
    ...         summary = {"total_items": 5, "mean_score": 0.94}
    ...         atomic_write_yaml(run_dir / "summary.yaml", summary)
    ...
    ...         return run_dir
    ...
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     run_dir = run_benchmark(Path(tmpdir), "gpt-4", "factuality")
    ...     print("Run completed:", run_dir.name[:5])  # First 5 chars
    ...     print("Files:", sorted(f.name for f in run_dir.iterdir() if not f.name.startswith(".")))
    Run completed: gpt-4
    Files: ['config.yaml', 'results.jsonl', 'summary.yaml']
    """
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    if sentinel:
        ensure_run_sentinel(run_dir)
    yield run_dir


__all__ = [
    "open_records_file",
    "atomic_write_text",
    "atomic_write_yaml",
    "ensure_run_sentinel",
    "managed_run_directory",
    "ExitStack",
]
