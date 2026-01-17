"""Experiment reproducibility and snapshot module.

This module provides tools for ensuring experiment reproducibility,
capturing environment state, managing seeds, and enabling deterministic replay.

Key Features:
- Random seed management across multiple libraries
- Environment state capture and restoration
- Experiment snapshot creation and loading
- Configuration versioning and diff
- Deterministic execution helpers
- Experiment replay and verification

Example:
    >>> from insideLLMs.reproducibility import ExperimentSnapshot, SeedManager
    >>>
    >>> seed_manager = SeedManager(global_seed=42)
    >>> seed_manager.set_all_seeds()
    >>>
    >>> snapshot = ExperimentSnapshot.capture()
    >>> snapshot.save("experiment_v1.json")
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import json
import os
import platform
import random
import sys
import time


class SnapshotFormat(Enum):
    """Snapshot file formats."""
    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"


class ReproducibilityLevel(Enum):
    """Levels of reproducibility guarantees."""
    NONE = "none"  # No guarantees
    SEED_ONLY = "seed_only"  # Random seeds set
    DETERMINISTIC = "deterministic"  # Full deterministic mode
    VERIFIED = "verified"  # Verified against baseline


class EnvironmentType(Enum):
    """Types of environment information."""
    PYTHON = "python"
    SYSTEM = "system"
    PACKAGES = "packages"
    ENVIRONMENT_VARS = "environment_vars"
    HARDWARE = "hardware"


@dataclass
class SeedState:
    """Captured state of random seeds."""
    global_seed: int
    python_random_state: Optional[tuple] = None
    numpy_state: Optional[Dict[str, Any]] = None
    torch_state: Optional[bytes] = None
    tensorflow_seed: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Captured environment information."""
    python_version: str
    platform_info: str
    os_info: str
    packages: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_directory: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Snapshot of experiment configuration."""
    config: Dict[str, Any]
    config_hash: str
    source_file: Optional[str] = None
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: float = field(default_factory=time.time)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    parent_experiment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    differences: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
        self._seed_history: List[SeedState] = []
        self._libraries_seeded: List[str] = []

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

    def set_all_seeds(self) -> Dict[str, bool]:
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

    def get_libraries_seeded(self) -> List[str]:
        """Get list of libraries that have been seeded."""
        return list(self._libraries_seeded)

    def derive_seed(self, key: str) -> int:
        """Derive a deterministic seed from the global seed and a key."""
        combined = f"{self.global_seed}:{key}"
        hash_value = hashlib.md5(combined.encode()).hexdigest()
        return int(hash_value[:8], 16)


class EnvironmentCapture:
    """Capture and compare environment state."""

    def __init__(self, capture_env_vars: bool = True, env_var_prefix: Optional[str] = None):
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

    def _get_installed_packages(self) -> Dict[str, str]:
        """Get installed Python packages."""
        packages = {}
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"]] = dist.version
        except Exception:
            pass
        return packages

    def compare(self, env1: EnvironmentInfo, env2: EnvironmentInfo) -> Dict[str, Any]:
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
            differences["python_version_match"]
            and len(differences["package_differences"]) == 0
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
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate a checksum for the snapshot."""
        content = json.dumps({
            "config": self.config.to_dict(),
            "seed": self.seed_state.global_seed,
            "inputs": str(self.inputs),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
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
        data = self.to_dict()

        if format == SnapshotFormat.JSON:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif format == SnapshotFormat.YAML:
            try:
                import yaml
                with open(path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                raise RuntimeError("PyYAML not installed for YAML format")

    @classmethod
    def load(cls, path: str) -> "ExperimentSnapshot":
        """Load snapshot from file."""
        with open(path, "r") as f:
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
        config: Optional[Dict[str, Any]] = None,
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

    def __init__(self):
        """Initialize version manager."""
        self._versions: Dict[str, ConfigSnapshot] = {}
        self._current_version: Optional[str] = None

    def add_version(
        self,
        version: str,
        config: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> ConfigSnapshot:
        """Add a configuration version."""
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]

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

    def list_versions(self) -> List[str]:
        """List all versions."""
        return list(self._versions.keys())

    def diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions."""
        v1 = self._versions.get(version1)
        v2 = self._versions.get(version2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        return self._diff_configs(v1.config, v2.config)

    def _diff_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        path: str = "",
    ) -> Dict[str, Any]:
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

    def __init__(self):
        """Initialize replay manager."""
        self._snapshots: Dict[str, ExperimentSnapshot] = {}

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
        outputs: Dict[str, Any],
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
        self._execution_log: List[Dict[str, Any]] = []

    def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
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

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution log."""
        return list(self._execution_log)


class CheckpointManager:
    """Manage experiment checkpoints."""

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    def create_checkpoint(
        self,
        name: str,
        state: Dict[str, Any],
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

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> List[Dict[str, str]]:
        """List all checkpoints."""
        return [
            {"id": ckpt["checkpoint_id"], "name": ckpt["name"]}
            for ckpt in self._checkpoints.values()
        ]

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
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
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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

    def __init__(self):
        """Initialize checker."""
        self._checks: List[Tuple[str, Callable[[], bool]]] = []
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

    def __init__(self):
        """Initialize registry."""
        self._experiments: Dict[str, ExperimentSnapshot] = {}
        self._tags: Dict[str, List[str]] = {}  # tag -> experiment_ids

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

    def find_by_tag(self, tag: str) -> List[ExperimentSnapshot]:
        """Find experiments by tag."""
        experiment_ids = self._tags.get(tag, [])
        return [self._experiments[eid] for eid in experiment_ids if eid in self._experiments]

    def find_by_name(self, name: str) -> List[ExperimentSnapshot]:
        """Find experiments by name."""
        return [
            exp for exp in self._experiments.values()
            if exp.metadata.name == name
        ]

    def list_all(self) -> List[Dict[str, str]]:
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


def set_global_seed(seed: int) -> Dict[str, bool]:
    """Set global seed across all libraries."""
    manager = SeedManager(global_seed=seed)
    return manager.set_all_seeds()


def capture_snapshot(
    name: str = "experiment",
    config: Optional[Dict[str, Any]] = None,
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


def compare_environments(env1: EnvironmentInfo, env2: EnvironmentInfo) -> Dict[str, Any]:
    """Compare two environment snapshots."""
    capture = EnvironmentCapture()
    return capture.compare(env1, env2)


def diff_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
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
    *args,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """Run a function deterministically with a fixed seed."""
    executor = DeterministicExecutor(seed=seed)
    return executor.execute(func, *args, **kwargs)


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a deterministic seed from a base seed and key."""
    manager = SeedManager(global_seed=base_seed)
    return manager.derive_seed(key)
