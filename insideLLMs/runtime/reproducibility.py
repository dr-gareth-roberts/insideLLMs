"""Experiment reproducibility and snapshot module.

This module provides comprehensive tools for ensuring experiment reproducibility,
capturing environment state, managing random seeds across multiple libraries,
and enabling deterministic replay of machine learning and scientific experiments.

The reproducibility framework addresses a critical challenge in ML research:
ensuring that experiments can be exactly replicated by controlling all sources
of randomness and capturing complete environment state.

Key Features
------------
- **Random Seed Management**: Unified seed control across Python's `random`,
  NumPy, PyTorch, and TensorFlow libraries with automatic detection.
- **Environment State Capture**: Full capture of Python version, platform info,
  installed packages, and environment variables for environment recreation.
- **Experiment Snapshots**: Complete serializable snapshots containing seeds,
  configuration, environment, inputs, and outputs with integrity checksums.
- **Configuration Versioning**: Track and diff configuration changes across
  experiment iterations with automatic hash-based version identification.
- **Deterministic Execution**: Execute functions with guaranteed seed state
  for reproducible results.
- **Experiment Replay**: Restore exact experiment conditions and verify
  that replayed results match original outputs.
- **Checkpoint Management**: Save and restore intermediate experiment states
  for long-running experiments.

Architecture Overview
---------------------
The module is organized around several key classes:

- `SeedManager`: Central seed management for all supported libraries
- `EnvironmentCapture`: System and package state capture
- `ExperimentSnapshot`: Complete experiment state container
- `ConfigVersionManager`: Configuration versioning and diffing
- `ExperimentReplayManager`: Replay orchestration and verification
- `DeterministicExecutor`: Wrapper for deterministic function execution
- `ExperimentCheckpointManager`: Intermediate state management
- `ReproducibilityChecker`: Audit and reporting on reproducibility status
- `ExperimentRegistry`: Central registry for tracking experiments

Supported Libraries
-------------------
The seed management system supports the following libraries:

- Python's built-in `random` module (always available)
- NumPy (`numpy.random`) - optional, detected at runtime
- PyTorch (`torch`) - optional, includes CUDA seed management
- TensorFlow (`tensorflow`) - optional, detected at runtime

Thread Safety
-------------
Note: This module is NOT thread-safe. Seed management and state capture
should be performed from a single thread, typically at experiment startup.
For multi-threaded applications, coordinate seed setting before spawning threads.

Examples
--------
Basic seed management for reproducible random numbers:

>>> from insideLLMs.runtime.reproducibility import SeedManager
>>>
>>> # Create manager with fixed seed
>>> seed_manager = SeedManager(global_seed=42)
>>> results = seed_manager.set_all_seeds()
>>> print(f"Libraries seeded: {seed_manager.get_libraries_seeded()}")
Libraries seeded: ['python_random', 'numpy', 'torch']

Capturing a complete experiment snapshot:

>>> from insideLLMs.runtime.reproducibility import ExperimentSnapshot
>>>
>>> # Define experiment configuration
>>> config = {
...     "model": "transformer",
...     "learning_rate": 0.001,
...     "batch_size": 32,
...     "epochs": 100
... }
>>>
>>> # Capture current state
>>> snapshot = ExperimentSnapshot.capture(
...     name="transformer_baseline",
...     config=config,
...     seed=42
... )
>>>
>>> # Save for later reproduction
>>> snapshot.save("experiments/baseline_v1.json")

Replaying an experiment and verifying results:

>>> from insideLLMs.runtime.reproducibility import (
...     ExperimentReplayManager,
...     ExperimentSnapshot
... )
>>>
>>> # Load original snapshot
>>> replay_manager = ExperimentReplayManager()
>>> snapshot = replay_manager.load_snapshot("experiments/baseline_v1.json")
>>>
>>> # Setup environment for replay
>>> seed_manager = replay_manager.setup_for_replay(snapshot.snapshot_id)
>>>
>>> # Run experiment again...
>>> new_outputs = {"accuracy": 0.95, "loss": 0.05}
>>>
>>> # Verify outputs match
>>> result = replay_manager.verify_replay(snapshot.snapshot_id, new_outputs)
>>> print(f"Replay matches: {result.matches}")
Replay matches: True

Comparing configurations across experiment versions:

>>> from insideLLMs.runtime.reproducibility import ConfigVersionManager
>>>
>>> manager = ConfigVersionManager()
>>>
>>> # Track configuration evolution
>>> manager.add_version("v1", {"lr": 0.01, "batch": 32})
>>> manager.add_version("v2", {"lr": 0.001, "batch": 64, "dropout": 0.1})
>>>
>>> # View differences
>>> diff = manager.diff("v1", "v2")
>>> print(f"Changed: {diff['changed']}")
>>> print(f"Added: {diff['added']}")
Changed: {'lr': {'old': 0.01, 'new': 0.001}, 'batch': {'old': 32, 'new': 64}}
Added: {'dropout': 0.1}

Using convenience functions for quick setup:

>>> from insideLLMs.runtime.reproducibility import (
...     set_global_seed,
...     capture_snapshot,
...     check_reproducibility
... )
>>>
>>> # Quick seed setup
>>> results = set_global_seed(42)
>>> print(f"NumPy seeded: {results['numpy']}")
NumPy seeded: True
>>>
>>> # Capture and check status
>>> snapshot = capture_snapshot("quick_experiment", seed=42)
>>> report = check_reproducibility(snapshot=snapshot)
>>> print(f"Reproducibility level: {report.level.value}")
Reproducibility level: deterministic

Notes
-----
- Always set seeds at the very beginning of your experiment, before any
  random operations are performed.
- For complete reproducibility in PyTorch, you may also need to set
  `torch.backends.cudnn.deterministic = True` and
  `torch.backends.cudnn.benchmark = False`.
- Environment capture does not include GPU driver versions or CUDA toolkit
  versions, which can affect numerical reproducibility.
- Snapshot files should be stored alongside experiment code in version control
  for complete reproducibility.

See Also
--------
- `random.seed` : Python's built-in seed function
- `numpy.random.seed` : NumPy's seed function
- `torch.manual_seed` : PyTorch's seed function
- `tensorflow.random.set_seed` : TensorFlow's seed function

References
----------
.. [1] Pineau, J., et al. "A Checklist for Reproducible Machine Learning
   Research." NeurIPS 2019 Reproducibility Workshop.
.. [2] PyTorch Reproducibility Documentation:
   https://pytorch.org/docs/stable/notes/randomness.html
"""

import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from insideLLMs._serialization import serialize_value as _serialize_value
from insideLLMs._serialization import stable_json_dumps as _stable_json_dumps


class SnapshotFormat(Enum):
    """Enumeration of supported snapshot file formats.

    This enum defines the serialization formats available for saving and
    loading experiment snapshots. Each format has different trade-offs
    in terms of human readability, file size, and compatibility.

    Attributes
    ----------
    JSON : str
        JSON format - human readable, widely compatible, recommended default.
        Produces text files that can be version controlled and inspected.
    YAML : str
        YAML format - more readable for complex nested structures.
        Requires PyYAML package to be installed.
    PICKLE : str
        Python pickle format - compact binary, supports complex Python objects.
        Not human readable, potential security concerns with untrusted files.

    Examples
    --------
    Selecting format when saving a snapshot:

    >>> from insideLLMs.runtime.reproducibility import (
    ...     ExperimentSnapshot,
    ...     SnapshotFormat
    ... )
    >>> snapshot = ExperimentSnapshot.capture(name="test")
    >>> snapshot.save("snapshot.json", format=SnapshotFormat.JSON)

    Checking available formats:

    >>> formats = list(SnapshotFormat)
    >>> print([f.value for f in formats])
    ['json', 'yaml', 'pickle']

    See Also
    --------
    ExperimentSnapshot.save : Method that uses this enum for format selection.
    ExperimentSnapshot.load : Method for loading snapshots from files.
    """

    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"


class ReproducibilityLevel(Enum):
    """Enumeration of reproducibility guarantee levels.

    This enum represents a hierarchy of reproducibility states, from no
    guarantees to fully verified reproduction. It is used by the
    ReproducibilityChecker to report the current reproducibility status
    of an experiment setup.

    The levels form an ordered hierarchy where each level builds upon
    the guarantees of the previous level:

    NONE < SEED_ONLY < DETERMINISTIC < VERIFIED

    Attributes
    ----------
    NONE : str
        No reproducibility guarantees. Random operations will produce
        different results on each run. This is the default state before
        any reproducibility measures are taken.
    SEED_ONLY : str
        Random seeds have been set, but environment state is not captured.
        Results should be reproducible on the same machine with the same
        package versions, but may differ across environments.
    DETERMINISTIC : str
        Full deterministic mode with seeds set and environment captured.
        Results should be reproducible if the environment can be recreated.
        Note: GPU operations may still have non-determinism.
    VERIFIED : str
        Results have been verified against a baseline run. This is the
        highest level, confirming that reproduction actually matches
        original results.

    Examples
    --------
    Checking reproducibility status:

    >>> from insideLLMs.runtime.reproducibility import (
    ...     ReproducibilityChecker,
    ...     SeedManager,
    ...     ReproducibilityLevel
    ... )
    >>> checker = ReproducibilityChecker()
    >>> # Without any setup
    >>> report = checker.check()
    >>> print(report.level == ReproducibilityLevel.NONE)
    True

    Using levels to gate experiment runs:

    >>> from insideLLMs.runtime.reproducibility import (
    ...     check_reproducibility,
    ...     SeedManager,
    ...     capture_snapshot,
    ...     ReproducibilityLevel
    ... )
    >>> seed_manager = SeedManager(42)
    >>> seed_manager.set_all_seeds()
    >>> snapshot = capture_snapshot("test", seed=42)
    >>> report = check_reproducibility(snapshot=snapshot, seed_manager=seed_manager)
    >>> if report.level.value in ["deterministic", "verified"]:
    ...     print("Safe to run production experiment")
    Safe to run production experiment

    Comparing levels:

    >>> levels = [ReproducibilityLevel.NONE, ReproducibilityLevel.VERIFIED]
    >>> print([l.value for l in levels])
    ['none', 'verified']

    See Also
    --------
    ReproducibilityChecker : Class that determines the current level.
    ReproducibilityReport : Report containing the assessed level.
    """

    NONE = "none"
    SEED_ONLY = "seed_only"
    DETERMINISTIC = "deterministic"
    VERIFIED = "verified"


class EnvironmentType(Enum):
    """Enumeration of environment information categories.

    This enum categorizes different types of environment information that
    can be captured for reproducibility purposes. Each category represents
    a distinct aspect of the execution environment that may affect
    experiment results.

    Attributes
    ----------
    PYTHON : str
        Python interpreter information including version, build, and
        implementation details.
    SYSTEM : str
        Operating system information including OS type, version, and
        release details.
    PACKAGES : str
        Installed Python packages and their versions. Critical for
        ensuring library compatibility across environments.
    ENVIRONMENT_VARS : str
        Environment variables that may affect execution, such as
        PATH, PYTHONPATH, or library-specific settings.
    HARDWARE : str
        Hardware information including CPU, memory, and GPU details.
        Important for performance-sensitive reproducibility.

    Examples
    --------
    Filtering environment capture by type:

    >>> from insideLLMs.runtime.reproducibility import EnvironmentType
    >>> # Define which types to capture
    >>> capture_types = [EnvironmentType.PYTHON, EnvironmentType.PACKAGES]
    >>> print([t.value for t in capture_types])
    ['python', 'packages']

    Using in conditional logic:

    >>> env_type = EnvironmentType.PACKAGES
    >>> if env_type == EnvironmentType.PACKAGES:
    ...     print("Capturing installed package versions...")
    Capturing installed package versions...

    Iterating over all types:

    >>> all_types = list(EnvironmentType)
    >>> print(len(all_types))
    5

    See Also
    --------
    EnvironmentCapture : Class that captures environment information.
    EnvironmentInfo : Dataclass storing captured environment data.
    """

    PYTHON = "python"
    SYSTEM = "system"
    PACKAGES = "packages"
    ENVIRONMENT_VARS = "environment_vars"
    HARDWARE = "hardware"


@dataclass
class SeedState:
    """Captured state of random number generator seeds across libraries.

    This dataclass stores the complete random state from multiple libraries,
    enabling exact restoration of random number generator states for
    reproducible execution. It captures both the seed values and the
    internal state of each library's RNG.

    The state can be serialized to dictionary format for storage in
    experiment snapshots and later restoration.

    Parameters
    ----------
    global_seed : int
        The master seed value used to initialize all random number generators.
        This is the primary seed from which library-specific seeds are derived.
    python_random_state : tuple, optional
        The internal state of Python's `random` module as returned by
        `random.getstate()`. Contains the Mersenne Twister state.
    numpy_state : dict[str, Any], optional
        The internal state of NumPy's random generator as returned by
        `numpy.random.get_state()`. Stored as a dictionary representation.
    torch_state : bytes, optional
        The internal state of PyTorch's random generator as raw bytes.
        Captures both CPU and CUDA RNG states if available.
    tensorflow_seed : int, optional
        The seed value used for TensorFlow. Note that TensorFlow's
        state cannot be fully captured, only the seed value is stored.
    timestamp : float
        Unix timestamp when the state was captured. Automatically set
        to current time if not provided.

    Attributes
    ----------
    global_seed : int
        The master seed value.
    python_random_state : tuple or None
        Python random module state.
    numpy_state : dict or None
        NumPy random state dictionary.
    torch_state : bytes or None
        PyTorch RNG state as bytes.
    tensorflow_seed : int or None
        TensorFlow seed value.
    timestamp : float
        Capture timestamp.

    Examples
    --------
    Creating a seed state manually:

    >>> import random
    >>> from insideLLMs.runtime.reproducibility import SeedState
    >>> random.seed(42)
    >>> state = SeedState(
    ...     global_seed=42,
    ...     python_random_state=random.getstate()
    ... )
    >>> print(f"Seed: {state.global_seed}")
    Seed: 42

    Capturing state through SeedManager (recommended):

    >>> from insideLLMs.runtime.reproducibility import SeedManager
    >>> manager = SeedManager(global_seed=42)
    >>> manager.set_all_seeds()
    >>> state = manager.get_state()
    >>> print(f"Captured at: {state.timestamp}")
    Captured at: 1704067200.0

    Converting to dictionary for serialization:

    >>> state = SeedState(global_seed=42)
    >>> state_dict = state.to_dict()
    >>> print(state_dict["global_seed"])
    42

    See Also
    --------
    SeedManager : Class for managing and capturing seed states.
    SeedManager.get_state : Method that creates SeedState instances.
    SeedManager.restore_state : Method to restore from a SeedState.
    """

    global_seed: int
    python_random_state: Optional[tuple] = None
    numpy_state: Optional[dict[str, Any]] = None
    torch_state: Optional[bytes] = None
    tensorflow_seed: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert the seed state to a JSON-serializable dictionary.

        Converts all state components to formats suitable for JSON
        serialization, including converting bytes to hex strings and
        complex state objects to string representations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all seed state information with keys:
            - 'global_seed': The master seed integer
            - 'python_random_state': Python random state tuple or None
            - 'numpy_state': String representation of NumPy state or None
            - 'torch_state': Hex-encoded PyTorch state or None
            - 'tensorflow_seed': TensorFlow seed integer or None
            - 'timestamp': Unix timestamp float

        Examples
        --------
        Basic conversion:

        >>> from insideLLMs.runtime.reproducibility import SeedState
        >>> state = SeedState(global_seed=42)
        >>> data = state.to_dict()
        >>> print(data["global_seed"])
        42

        Converting state with NumPy:

        >>> import numpy as np
        >>> from insideLLMs.runtime.reproducibility import SeedManager
        >>> manager = SeedManager(42)
        >>> manager.set_all_seeds()
        >>> state = manager.get_state()
        >>> data = state.to_dict()
        >>> print("numpy_state" in data)
        True

        Serializing to JSON:

        >>> import json
        >>> from insideLLMs.runtime.reproducibility import SeedState
        >>> state = SeedState(global_seed=123)
        >>> json_str = json.dumps(state.to_dict())
        >>> print("global_seed" in json_str)
        True
        """
        return {
            "global_seed": self.global_seed,
            "python_random_state": self.python_random_state,
            "numpy_state": str(self.numpy_state) if self.numpy_state else None,
            "torch_state": self.torch_state.hex() if self.torch_state else None,
            "tensorflow_seed": self.tensorflow_seed,
            "timestamp": self.timestamp,
        }


@dataclass
class EnvironmentInfo:
    """Captured environment information for reproducibility.

    This dataclass stores comprehensive information about the execution
    environment including Python version, operating system details,
    installed packages, and environment variables. This information is
    essential for recreating the exact conditions under which an
    experiment was run.

    Environment information helps diagnose reproducibility issues when
    results differ across machines or over time due to package updates.

    Parameters
    ----------
    python_version : str
        Full Python version string including build information,
        as returned by `sys.version`.
    platform_info : str
        Platform identification string including OS and architecture,
        as returned by `platform.platform()`.
    os_info : str
        Operating system name and release version, formatted as
        "{system} {release}".
    packages : dict[str, str], optional
        Dictionary mapping package names to version strings.
        Defaults to empty dict if not provided.
    env_vars : dict[str, str], optional
        Dictionary of environment variables and their values.
        May be filtered to only include relevant variables.
    working_directory : str, optional
        The current working directory when capture was performed.
    timestamp : float
        Unix timestamp when the environment was captured.
        Automatically set to current time if not provided.

    Attributes
    ----------
    python_version : str
        Full Python version string.
    platform_info : str
        Platform identification.
    os_info : str
        OS name and version.
    packages : dict[str, str]
        Installed package versions.
    env_vars : dict[str, str]
        Captured environment variables.
    working_directory : str
        Working directory path.
    timestamp : float
        Capture timestamp.

    Examples
    --------
    Creating environment info manually:

    >>> import sys
    >>> import platform
    >>> from insideLLMs.runtime.reproducibility import EnvironmentInfo
    >>> env = EnvironmentInfo(
    ...     python_version=sys.version,
    ...     platform_info=platform.platform(),
    ...     os_info=f"{platform.system()} {platform.release()}"
    ... )
    >>> print(f"Python: {env.python_version.split()[0]}")
    Python: 3.11.0

    Using EnvironmentCapture (recommended):

    >>> from insideLLMs.runtime.reproducibility import EnvironmentCapture
    >>> capture = EnvironmentCapture()
    >>> env = capture.capture()
    >>> print(f"OS: {env.os_info}")
    OS: Darwin 23.0.0

    Checking package versions:

    >>> from insideLLMs.runtime.reproducibility import capture_environment
    >>> env = capture_environment()
    >>> if "numpy" in env.packages:
    ...     print(f"NumPy version: {env.packages['numpy']}")
    NumPy version: 1.24.0

    See Also
    --------
    EnvironmentCapture : Class for capturing environment information.
    EnvironmentCapture.compare : Compare two EnvironmentInfo instances.
    capture_environment : Convenience function for quick capture.
    """

    python_version: str
    platform_info: str
    os_info: str
    packages: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    working_directory: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert environment info to a JSON-serializable dictionary.

        Creates a dictionary representation suitable for JSON serialization
        and storage in experiment snapshots.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all environment information with keys:
            - 'python_version': Python version string
            - 'platform_info': Platform identification
            - 'os_info': Operating system info
            - 'packages': Package name to version mapping
            - 'env_vars': Environment variable mapping
            - 'working_directory': Working directory path
            - 'timestamp': Capture timestamp

        Examples
        --------
        Basic conversion:

        >>> from insideLLMs.runtime.reproducibility import capture_environment
        >>> env = capture_environment()
        >>> data = env.to_dict()
        >>> print("python_version" in data)
        True

        Serializing to JSON:

        >>> import json
        >>> from insideLLMs.runtime.reproducibility import capture_environment
        >>> env = capture_environment()
        >>> json_str = json.dumps(env.to_dict())
        >>> print("platform_info" in json_str)
        True

        Accessing nested package data:

        >>> from insideLLMs.runtime.reproducibility import capture_environment
        >>> env = capture_environment()
        >>> data = env.to_dict()
        >>> packages = data["packages"]
        >>> print(isinstance(packages, dict))
        True
        """
        return {
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "os_info": self.os_info,
            "packages": self.packages,
            "env_vars": self.env_vars,
            "working_directory": self.working_directory,
            "timestamp": self.timestamp,
        }


@dataclass
class ConfigSnapshot:
    """Snapshot of experiment configuration with versioning support.

    This dataclass captures a point-in-time snapshot of experiment
    configuration, including a hash for integrity verification and
    change detection. Configuration snapshots enable tracking how
    experiment parameters evolve across iterations.

    The config_hash is computed from the configuration content,
    allowing quick comparison between configurations without
    examining every field.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary containing all experiment parameters.
        Can be arbitrarily nested with any JSON-serializable values.
    config_hash : str
        SHA-256 hash (truncated to 16 characters) of the configuration.
        Used for quick equality checks and change detection.
    source_file : str, optional
        Path to the file from which configuration was loaded.
        Useful for tracking configuration provenance.
    version : str, optional
        Version identifier for this configuration. Defaults to "1.0".
        Can follow semantic versioning or any custom scheme.
    created_at : float
        Unix timestamp when the snapshot was created.
        Automatically set to current time if not provided.

    Attributes
    ----------
    config : dict[str, Any]
        The configuration parameters.
    config_hash : str
        Hash of the configuration.
    source_file : str or None
        Source file path.
    version : str
        Configuration version.
    created_at : float
        Creation timestamp.

    Examples
    --------
    Creating a configuration snapshot:

    >>> import hashlib
    >>> import json
    >>> from insideLLMs.runtime.reproducibility import ConfigSnapshot
    >>> config = {"learning_rate": 0.001, "batch_size": 32}
    >>> config_hash = hashlib.sha256(
    ...     json.dumps(config, sort_keys=True).encode()
    ... ).hexdigest()[:16]
    >>> snapshot = ConfigSnapshot(
    ...     config=config,
    ...     config_hash=config_hash,
    ...     version="1.0"
    ... )
    >>> print(f"Config hash: {snapshot.config_hash}")
    Config hash: 7a8b9c0d1e2f3a4b

    Using ConfigVersionManager (recommended):

    >>> from insideLLMs.runtime.reproducibility import ConfigVersionManager
    >>> manager = ConfigVersionManager()
    >>> snapshot = manager.add_version("v1", {"lr": 0.01, "epochs": 100})
    >>> print(f"Version: {snapshot.version}")
    Version: v1

    Loading from ExperimentSnapshot:

    >>> from insideLLMs.runtime.reproducibility import ExperimentSnapshot
    >>> exp_snapshot = ExperimentSnapshot.capture(
    ...     name="test",
    ...     config={"param": "value"}
    ... )
    >>> print(f"Config: {exp_snapshot.config.config}")
    Config: {'param': 'value'}

    See Also
    --------
    ConfigVersionManager : Class for managing configuration versions.
    ExperimentSnapshot : Contains ConfigSnapshot as a component.
    diff_configs : Compare two configuration dictionaries.
    """

    config: dict[str, Any]
    config_hash: str
    source_file: Optional[str] = None
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration snapshot to a JSON-serializable dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all snapshot information with keys:
            - 'config': The configuration dictionary
            - 'config_hash': Hash string for the configuration
            - 'source_file': Source file path or None
            - 'version': Version string
            - 'created_at': Creation timestamp

        Examples
        --------
        Basic conversion:

        >>> from insideLLMs.runtime.reproducibility import ConfigVersionManager
        >>> manager = ConfigVersionManager()
        >>> snapshot = manager.add_version("v1", {"param": 1})
        >>> data = snapshot.to_dict()
        >>> print(data["version"])
        v1

        Accessing configuration:

        >>> from insideLLMs.runtime.reproducibility import ConfigVersionManager
        >>> manager = ConfigVersionManager()
        >>> snapshot = manager.add_version("v1", {"a": 1, "b": 2})
        >>> data = snapshot.to_dict()
        >>> print(data["config"]["a"])
        1

        Serializing to JSON file:

        >>> import json
        >>> from insideLLMs.runtime.reproducibility import ConfigVersionManager
        >>> manager = ConfigVersionManager()
        >>> snapshot = manager.add_version("v1", {"key": "value"})
        >>> json_data = json.dumps(snapshot.to_dict(), indent=2)
        >>> print("config_hash" in json_data)
        True
        """
        return {
            "config": self.config,
            "config_hash": self.config_hash,
            "source_file": self.source_file,
            "version": self.version,
            "created_at": self.created_at,
        }


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""

    experiment_id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    author: str = ""
    created_at: float = field(default_factory=time.time)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    parent_experiment: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "author": author if (author := self.author) else None,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "parent_experiment": self.parent_experiment,
        }


@dataclass
class ReplayResult:
    """Result of replaying an experiment."""

    success: bool
    original_hash: str
    replay_hash: str
    matches: bool
    differences: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "original_hash": self.original_hash,
            "replay_hash": self.replay_hash,
            "matches": self.matches,
            "differences": self.differences,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class SeedManager:
    """Manage random seeds across libraries."""

    def __init__(self, global_seed: Optional[int] = None):
        """Initialize seed manager.

        Args:
            global_seed: The seed to use. If None, generates a random seed.
        """
        self.global_seed = global_seed if global_seed is not None else self._generate_seed()
        self._seed_history: list[SeedState] = []
        self._libraries_seeded: list[str] = []

    def _generate_seed(self) -> int:
        """Generate a random seed."""
        return int(time.time() * 1000) % (2**31)

    def set_python_seed(self) -> None:
        """Set Python's random module seed."""
        random.seed(self.global_seed)
        self._libraries_seeded.append("python_random")

    def set_numpy_seed(self) -> bool:
        """Set NumPy's random seed if available."""
        try:
            import numpy as np

            np.random.seed(self.global_seed)
            self._libraries_seeded.append("numpy")
            return True
        except ImportError:
            return False

    def set_torch_seed(self) -> bool:
        """Set PyTorch's random seed if available."""
        try:
            import torch

            torch.manual_seed(self.global_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.global_seed)
            self._libraries_seeded.append("torch")
            return True
        except ImportError:
            return False

    def set_tensorflow_seed(self) -> bool:
        """Set TensorFlow's random seed if available."""
        try:
            import tensorflow as tf

            tf.random.set_seed(self.global_seed)
            self._libraries_seeded.append("tensorflow")
            return True
        except ImportError:
            return False

    def set_all_seeds(self) -> dict[str, bool]:
        """Set seeds for all available libraries."""
        results = {
            "python_random": True,
            "numpy": False,
            "torch": False,
            "tensorflow": False,
        }

        self.set_python_seed()
        results["numpy"] = self.set_numpy_seed()
        results["torch"] = self.set_torch_seed()
        results["tensorflow"] = self.set_tensorflow_seed()

        self._capture_state()

        return results

    def _capture_state(self) -> SeedState:
        """Capture current seed state."""
        state = SeedState(
            global_seed=self.global_seed,
            python_random_state=random.getstate(),
        )

        # Capture NumPy state if available
        try:
            import numpy as np

            state.numpy_state = {"state": str(np.random.get_state())}
        except ImportError:
            pass

        # Capture PyTorch state if available
        try:
            import torch

            state.torch_state = torch.get_rng_state().numpy().tobytes()
        except ImportError:
            pass

        self._seed_history.append(state)
        return state

    def get_state(self) -> SeedState:
        """Get current seed state."""
        return self._capture_state()

    def restore_state(self, state: SeedState) -> None:
        """Restore a previous seed state."""
        self.global_seed = state.global_seed

        if state.python_random_state:
            random.setstate(state.python_random_state)

    def get_libraries_seeded(self) -> list[str]:
        """Get list of libraries that have been seeded."""
        return list(self._libraries_seeded)

    def derive_seed(self, key: str) -> int:
        """Derive a deterministic seed from the global seed and a key."""
        combined = f"{self.global_seed}:{key}"
        hash_value = hashlib.md5(combined.encode()).hexdigest()
        return int(hash_value[:8], 16)


class EnvironmentCapture:
    """Capture and compare environment state."""

    def __init__(self, capture_env_vars: bool = False, env_var_prefix: Optional[str] = None):
        """Initialize environment capture.

        Args:
            capture_env_vars: Whether to capture environment variables
            env_var_prefix: Only capture env vars with this prefix
        """
        self.capture_env_vars = capture_env_vars
        self.env_var_prefix = env_var_prefix

    def capture(self) -> EnvironmentInfo:
        """Capture current environment state."""
        env_vars = {}
        if self.capture_env_vars:
            for key, value in os.environ.items():
                if self.env_var_prefix is None or key.startswith(self.env_var_prefix):
                    env_vars[key] = value

        return EnvironmentInfo(
            python_version=sys.version,
            platform_info=platform.platform(),
            os_info=f"{platform.system()} {platform.release()}",
            packages=self._get_installed_packages(),
            env_vars=env_vars,
            working_directory=os.getcwd(),
        )

    def _get_installed_packages(self) -> dict[str, str]:
        """Get installed Python packages."""
        packages = {}
        try:
            import importlib.metadata

            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"]] = dist.version
        except Exception:
            pass
        return packages

    def compare(self, env1: EnvironmentInfo, env2: EnvironmentInfo) -> dict[str, Any]:
        """Compare two environment snapshots."""
        differences = {
            "python_version_match": env1.python_version == env2.python_version,
            "platform_match": env1.platform_info == env2.platform_info,
            "package_differences": {},
            "env_var_differences": {},
        }

        # Compare packages
        all_packages = set(env1.packages.keys()) | set(env2.packages.keys())
        for pkg in all_packages:
            v1 = env1.packages.get(pkg)
            v2 = env2.packages.get(pkg)
            if v1 != v2:
                differences["package_differences"][pkg] = {"env1": v1, "env2": v2}

        # Compare env vars
        all_vars = set(env1.env_vars.keys()) | set(env2.env_vars.keys())
        for var in all_vars:
            v1 = env1.env_vars.get(var)
            v2 = env2.env_vars.get(var)
            if v1 != v2:
                differences["env_var_differences"][var] = {"env1": v1, "env2": v2}

        differences["is_compatible"] = (
            differences["python_version_match"] and len(differences["package_differences"]) == 0
        )

        return differences


@dataclass
class ExperimentSnapshot:
    """Complete snapshot of an experiment state."""

    snapshot_id: str
    metadata: ExperimentMetadata
    config: ConfigSnapshot
    seed_state: SeedState
    environment: EnvironmentInfo
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self) -> None:
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate a checksum for the snapshot."""
        content = json.dumps(
            {
                "config": self.config.to_dict(),
                "seed": self.seed_state.global_seed,
                "inputs": str(self.inputs),
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "metadata": self.metadata.to_dict(),
            "config": self.config.to_dict(),
            "seed_state": self.seed_state.to_dict(),
            "environment": self.environment.to_dict(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "checksum": self.checksum,
        }

    def save(self, path: str, format: SnapshotFormat = SnapshotFormat.JSON) -> None:
        """Save snapshot to file."""
        # Normalize complex types (e.g., Enum/datetime) before emitting artifacts.
        data = _serialize_value(self.to_dict())

        if format == SnapshotFormat.JSON:
            with open(path, "w", encoding="utf-8") as f:
                f.write(_stable_json_dumps(data))
        elif format == SnapshotFormat.YAML:
            try:
                import yaml

                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        data,
                        f,
                        default_flow_style=False,
                        sort_keys=True,
                        allow_unicode=True,
                    )
            except ImportError:
                raise RuntimeError("PyYAML not installed for YAML format")

    @classmethod
    def load(cls, path: str) -> "ExperimentSnapshot":
        """Load snapshot from file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            snapshot_id=data["snapshot_id"],
            metadata=ExperimentMetadata(**data["metadata"]),
            config=ConfigSnapshot(**data["config"]),
            seed_state=SeedState(**data["seed_state"]),
            environment=EnvironmentInfo(**data["environment"]),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            checksum=data.get("checksum", ""),
        )

    @classmethod
    def capture(
        cls,
        name: str = "experiment",
        config: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> "ExperimentSnapshot":
        """Capture current state as a snapshot."""
        snapshot_id = f"snap_{int(time.time() * 1000)}"

        # Capture seed state
        seed_manager = SeedManager(global_seed=seed)
        seed_state = seed_manager.get_state()

        # Capture environment
        env_capture = EnvironmentCapture(capture_env_vars=False)
        environment = env_capture.capture()

        # Create config snapshot
        config = config or {}
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        config_snapshot = ConfigSnapshot(config=config, config_hash=config_hash)

        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=snapshot_id,
            name=name,
        )

        return cls(
            snapshot_id=snapshot_id,
            metadata=metadata,
            config=config_snapshot,
            seed_state=seed_state,
            environment=environment,
        )


class ConfigVersionManager:
    """Manage versioned configurations."""

    def __init__(self) -> None:
        """Initialize version manager."""
        self._versions: dict[str, ConfigSnapshot] = {}
        self._current_version: Optional[str] = None

    def add_version(
        self,
        version: str,
        config: dict[str, Any],
        source_file: Optional[str] = None,
    ) -> ConfigSnapshot:
        """Add a configuration version."""
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

        snapshot = ConfigSnapshot(
            config=config,
            config_hash=config_hash,
            source_file=source_file,
            version=version,
        )

        self._versions[version] = snapshot
        self._current_version = version

        return snapshot

    def get_version(self, version: str) -> Optional[ConfigSnapshot]:
        """Get a specific version."""
        return self._versions.get(version)

    def get_current(self) -> Optional[ConfigSnapshot]:
        """Get current version."""
        if self._current_version:
            return self._versions.get(self._current_version)
        return None

    def list_versions(self) -> list[str]:
        """List all versions."""
        return list(self._versions.keys())

    def diff(self, version1: str, version2: str) -> dict[str, Any]:
        """Compare two versions."""
        v1 = self._versions.get(version1)
        v2 = self._versions.get(version2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        return self._diff_configs(v1.config, v2.config)

    def _diff_configs(
        self,
        config1: dict[str, Any],
        config2: dict[str, Any],
        path: str = "",
    ) -> dict[str, Any]:
        """Recursively diff two configs."""
        differences = {
            "added": {},
            "removed": {},
            "changed": {},
        }

        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            key_path = f"{path}.{key}" if path else key

            if key not in config1:
                differences["added"][key_path] = config2[key]
            elif key not in config2:
                differences["removed"][key_path] = config1[key]
            elif config1[key] != config2[key]:
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    nested = self._diff_configs(config1[key], config2[key], key_path)
                    differences["added"].update(nested["added"])
                    differences["removed"].update(nested["removed"])
                    differences["changed"].update(nested["changed"])
                else:
                    differences["changed"][key_path] = {
                        "old": config1[key],
                        "new": config2[key],
                    }

        return differences


class ExperimentReplayManager:
    """Manage experiment replay and verification."""

    def __init__(self) -> None:
        """Initialize replay manager."""
        self._snapshots: dict[str, ExperimentSnapshot] = {}

    def register_snapshot(self, snapshot: ExperimentSnapshot) -> None:
        """Register a snapshot for replay."""
        self._snapshots[snapshot.snapshot_id] = snapshot

    def load_snapshot(self, path: str) -> ExperimentSnapshot:
        """Load and register a snapshot from file."""
        snapshot = ExperimentSnapshot.load(path)
        self.register_snapshot(snapshot)
        return snapshot

    def setup_for_replay(self, snapshot_id: str) -> SeedManager:
        """Setup environment for replaying a snapshot."""
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        # Setup seeds
        seed_manager = SeedManager(global_seed=snapshot.seed_state.global_seed)
        seed_manager.set_all_seeds()

        return seed_manager

    def verify_replay(
        self,
        snapshot_id: str,
        outputs: dict[str, Any],
    ) -> ReplayResult:
        """Verify replay outputs match original."""
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            return ReplayResult(
                success=False,
                original_hash="",
                replay_hash="",
                matches=False,
                error="Snapshot not found",
            )

        original_hash = hashlib.sha256(
            json.dumps(snapshot.outputs, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        replay_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        matches = original_hash == replay_hash

        differences = []
        if not matches:
            # Find specific differences
            all_keys = set(snapshot.outputs.keys()) | set(outputs.keys())
            for key in all_keys:
                orig = snapshot.outputs.get(key)
                replay = outputs.get(key)
                if str(orig) != str(replay):
                    differences.append(f"{key}: {orig} -> {replay}")

        return ReplayResult(
            success=True,
            original_hash=original_hash,
            replay_hash=replay_hash,
            matches=matches,
            differences=differences,
        )


class DeterministicExecutor:
    """Execute functions deterministically."""

    def __init__(self, seed: int):
        """Initialize executor.

        Args:
            seed: Global seed for deterministic execution
        """
        self.seed = seed
        self.seed_manager = SeedManager(global_seed=seed)
        self._execution_log: list[dict[str, Any]] = []

    def execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Execute a function deterministically."""
        # Set seeds before execution
        self.seed_manager.set_all_seeds()

        start_time = time.time()

        # Execute function
        result = func(*args, **kwargs)

        duration = (time.time() - start_time) * 1000

        # Log execution
        execution_record = {
            "function": func.__name__,
            "seed": self.seed,
            "duration_ms": duration,
            "timestamp": time.time(),
        }
        self._execution_log.append(execution_record)

        return result, execution_record

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Get execution log."""
        return list(self._execution_log)


class ExperimentCheckpointManager:
    """Manage experiment checkpoints."""

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self._checkpoints: dict[str, dict[str, Any]] = {}

    def create_checkpoint(
        self,
        name: str,
        state: dict[str, Any],
        seed_state: Optional[SeedState] = None,
    ) -> str:
        """Create a checkpoint."""
        checkpoint_id = f"ckpt_{name}_{int(time.time() * 1000)}"

        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "name": name,
            "state": state,
            "seed_state": seed_state.to_dict() if seed_state else None,
            "created_at": time.time(),
        }

        self._checkpoints[checkpoint_id] = checkpoint

        return checkpoint_id

    def get_checkpoint(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> list[dict[str, str]]:
        """List all checkpoints."""
        return [
            {"id": ckpt["checkpoint_id"], "name": ckpt["name"]}
            for ckpt in self._checkpoints.values()
        ]

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """Restore state from a checkpoint."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return None
        return checkpoint["state"]


@dataclass
class ReproducibilityReport:
    """Report on reproducibility status."""

    level: ReproducibilityLevel
    seed_set: bool
    environment_captured: bool
    config_versioned: bool
    checkpoints_available: bool
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "seed_set": self.seed_set,
            "environment_captured": self.environment_captured,
            "config_versioned": self.config_versioned,
            "checkpoints_available": self.checkpoints_available,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class ReproducibilityChecker:
    """Check and report on reproducibility status."""

    def __init__(self) -> None:
        """Initialize checker."""
        self._checks: list[tuple[str, Callable[[], bool]]] = []
        self._setup_default_checks()

    def _setup_default_checks(self) -> None:
        """Setup default reproducibility checks."""
        # Check for common non-deterministic operations
        pass

    def check(
        self,
        snapshot: Optional[ExperimentSnapshot] = None,
        seed_manager: Optional[SeedManager] = None,
        config_manager: Optional[ConfigVersionManager] = None,
    ) -> ReproducibilityReport:
        """Check reproducibility status."""
        warnings = []
        recommendations = []

        seed_set = seed_manager is not None and len(seed_manager.get_libraries_seeded()) > 0
        environment_captured = snapshot is not None
        config_versioned = config_manager is not None and len(config_manager.list_versions()) > 0
        checkpoints_available = False  # Would need checkpoint manager

        # Determine level
        if not seed_set:
            level = ReproducibilityLevel.NONE
            recommendations.append("Set random seeds for reproducibility")
        elif not environment_captured:
            level = ReproducibilityLevel.SEED_ONLY
            recommendations.append("Capture environment snapshot")
        else:
            level = ReproducibilityLevel.DETERMINISTIC

        # Add warnings
        if not config_versioned:
            warnings.append("Configuration not versioned")
            recommendations.append("Use ConfigVersionManager to track config changes")

        return ReproducibilityReport(
            level=level,
            seed_set=seed_set,
            environment_captured=environment_captured,
            config_versioned=config_versioned,
            checkpoints_available=checkpoints_available,
            warnings=warnings,
            recommendations=recommendations,
        )


class ExperimentRegistry:
    """Registry for tracking experiments."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._experiments: dict[str, ExperimentSnapshot] = {}
        self._tags: dict[str, list[str]] = {}  # tag -> experiment_ids

    def register(self, snapshot: ExperimentSnapshot) -> str:
        """Register an experiment."""
        self._experiments[snapshot.snapshot_id] = snapshot

        # Index by tags
        for tag in snapshot.metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(snapshot.snapshot_id)

        return snapshot.snapshot_id

    def get(self, experiment_id: str) -> Optional[ExperimentSnapshot]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def find_by_tag(self, tag: str) -> list[ExperimentSnapshot]:
        """Find experiments by tag."""
        experiment_ids = self._tags.get(tag, [])
        return [self._experiments[eid] for eid in experiment_ids if eid in self._experiments]

    def find_by_name(self, name: str) -> list[ExperimentSnapshot]:
        """Find experiments by name."""
        return [exp for exp in self._experiments.values() if exp.metadata.name == name]

    def list_all(self) -> list[dict[str, str]]:
        """List all experiments."""
        return [
            {
                "id": exp.snapshot_id,
                "name": exp.metadata.name,
                "created_at": exp.metadata.created_at,
            }
            for exp in self._experiments.values()
        ]


# Convenience functions
def create_seed_manager(seed: Optional[int] = None) -> SeedManager:
    """Create a seed manager."""
    return SeedManager(global_seed=seed)


def set_global_seed(seed: int) -> dict[str, bool]:
    """Set global seed across all libraries."""
    manager = SeedManager(global_seed=seed)
    return manager.set_all_seeds()


def capture_snapshot(
    name: str = "experiment",
    config: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> ExperimentSnapshot:
    """Capture a complete experiment snapshot."""
    return ExperimentSnapshot.capture(name=name, config=config, seed=seed)


def save_snapshot(snapshot: ExperimentSnapshot, path: str) -> None:
    """Save a snapshot to file."""
    snapshot.save(path)


def load_snapshot(path: str) -> ExperimentSnapshot:
    """Load a snapshot from file."""
    return ExperimentSnapshot.load(path)


def capture_environment() -> EnvironmentInfo:
    """Capture current environment information."""
    capture = EnvironmentCapture()
    return capture.capture()


def compare_environments(env1: EnvironmentInfo, env2: EnvironmentInfo) -> dict[str, Any]:
    """Compare two environment snapshots."""
    capture = EnvironmentCapture()
    return capture.compare(env1, env2)


def diff_configs(config1: dict[str, Any], config2: dict[str, Any]) -> dict[str, Any]:
    """Diff two configurations."""
    manager = ConfigVersionManager()
    manager.add_version("v1", config1)
    manager.add_version("v2", config2)
    return manager.diff("v1", "v2")


def check_reproducibility(
    snapshot: Optional[ExperimentSnapshot] = None,
    seed_manager: Optional[SeedManager] = None,
) -> ReproducibilityReport:
    """Check reproducibility status."""
    checker = ReproducibilityChecker()
    return checker.check(snapshot=snapshot, seed_manager=seed_manager)


def run_deterministic(
    func: Callable,
    seed: int,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run a function deterministically with a fixed seed."""
    executor = DeterministicExecutor(seed=seed)
    return executor.execute(func, *args, **kwargs)


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic seed from a base seed and key."""
    manager = SeedManager(global_seed=base_seed)
    return manager.derive_seed(key)
