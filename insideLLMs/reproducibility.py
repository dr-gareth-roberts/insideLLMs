"""
Experiment Reproducibility and Snapshot Module.

This module provides a compatibility shim that re-exports all components from
:mod:`insideLLMs.runtime.reproducibility`. It offers comprehensive tools for
ensuring experiment reproducibility, capturing environment state, managing
random seeds across multiple libraries, and enabling deterministic replay
of experiments.

Overview
--------
Reproducibility is critical for scientific machine learning research and
production ML systems. This module addresses reproducibility challenges by
providing:

1. **Seed Management**: Unified control over random number generators in Python,
   NumPy, PyTorch, and TensorFlow.

2. **Environment Capture**: Snapshot the complete execution environment including
   Python version, installed packages, and relevant environment variables.

3. **Experiment Snapshots**: Create comprehensive snapshots that combine seeds,
   environment, configuration, and I/O for complete experiment state capture.

4. **Configuration Versioning**: Track configuration changes over time with
   diff capabilities to understand experiment evolution.

5. **Deterministic Execution**: Execute functions with guaranteed seed state
   for reproducible results.

6. **Replay and Verification**: Replay experiments from snapshots and verify
   that outputs match the original run.

Classes
-------
.. autosummary::
    :toctree: generated

    SnapshotFormat
    ReproducibilityLevel
    EnvironmentType
    SeedState
    EnvironmentInfo
    ConfigSnapshot
    ExperimentMetadata
    ReplayResult
    SeedManager
    EnvironmentCapture
    ExperimentSnapshot
    ConfigVersionManager
    ExperimentReplayManager
    DeterministicExecutor
    ExperimentCheckpointManager
    ReproducibilityReport
    ReproducibilityChecker
    ExperimentRegistry

Functions
---------
.. autosummary::
    :toctree: generated

    create_seed_manager
    set_global_seed
    capture_snapshot
    save_snapshot
    load_snapshot
    capture_environment
    compare_environments
    diff_configs
    check_reproducibility
    run_deterministic
    derive_seed

Examples
--------
**Basic Seed Management**

Set seeds across all available ML libraries for reproducibility:

>>> from insideLLMs.reproducibility import SeedManager
>>>
>>> # Create manager with a fixed seed
>>> seed_manager = SeedManager(global_seed=42)
>>> results = seed_manager.set_all_seeds()
>>> print(results)
{'python_random': True, 'numpy': True, 'torch': True, 'tensorflow': False}
>>>
>>> # Check which libraries were seeded
>>> print(seed_manager.get_libraries_seeded())
['python_random', 'numpy', 'torch']

**Quick Seed Setting**

For simple cases, use the convenience function:

>>> from insideLLMs.reproducibility import set_global_seed
>>>
>>> results = set_global_seed(12345)
>>> # Now all random operations are deterministic

**Capturing Experiment Snapshots**

Capture the complete state of an experiment:

>>> from insideLLMs.reproducibility import ExperimentSnapshot
>>>
>>> # Define your experiment configuration
>>> config = {
...     "model": "transformer",
...     "hidden_size": 256,
...     "learning_rate": 0.001,
...     "batch_size": 32,
... }
>>>
>>> # Capture current state
>>> snapshot = ExperimentSnapshot.capture(
...     name="attention_experiment_v1",
...     config=config,
...     seed=42,
... )
>>>
>>> # Save for later use
>>> snapshot.save("experiment_snapshot.json")
>>>
>>> # Verify snapshot integrity
>>> print(f"Snapshot ID: {snapshot.snapshot_id}")
>>> print(f"Checksum: {snapshot.checksum}")

**Loading and Replaying Experiments**

Restore an experiment from a saved snapshot:

>>> from insideLLMs.reproducibility import (
...     ExperimentSnapshot,
...     ExperimentReplayManager,
... )
>>>
>>> # Load a previous snapshot
>>> snapshot = ExperimentSnapshot.load("experiment_snapshot.json")
>>>
>>> # Setup replay manager
>>> replay_manager = ExperimentReplayManager()
>>> replay_manager.register_snapshot(snapshot)
>>>
>>> # Prepare environment for replay
>>> seed_manager = replay_manager.setup_for_replay(snapshot.snapshot_id)
>>>
>>> # Run your experiment...
>>> # results = run_my_experiment()
>>>
>>> # Verify results match original
>>> # verification = replay_manager.verify_replay(
>>> #     snapshot.snapshot_id,
>>> #     outputs={"accuracy": results["accuracy"]}
>>> # )

**Environment Comparison**

Compare execution environments between runs:

>>> from insideLLMs.reproducibility import (
...     capture_environment,
...     compare_environments,
... )
>>>
>>> # Capture current environment
>>> env_now = capture_environment()
>>> print(f"Python: {env_now.python_version}")
>>> print(f"Platform: {env_now.platform_info}")
>>>
>>> # Later, compare with a saved environment
>>> # env_then = ...  # from a loaded snapshot
>>> # diff = compare_environments(env_then, env_now)
>>> # if not diff["is_compatible"]:
>>> #     print("Warning: Environment has changed!")
>>> #     print(f"Package differences: {diff['package_differences']}")

**Configuration Version Management**

Track configuration changes across experiment iterations:

>>> from insideLLMs.reproducibility import ConfigVersionManager, diff_configs
>>>
>>> # Create version manager
>>> config_manager = ConfigVersionManager()
>>>
>>> # Add initial configuration
>>> config_v1 = {
...     "model": "lstm",
...     "hidden_size": 128,
...     "dropout": 0.1,
... }
>>> config_manager.add_version("v1.0", config_v1)
>>>
>>> # Add updated configuration
>>> config_v2 = {
...     "model": "transformer",
...     "hidden_size": 256,
...     "dropout": 0.1,
...     "attention_heads": 4,
... }
>>> config_manager.add_version("v2.0", config_v2)
>>>
>>> # Compare versions
>>> changes = config_manager.diff("v1.0", "v2.0")
>>> print(f"Changed: {changes['changed']}")
>>> print(f"Added: {changes['added']}")
>>>
>>> # Quick diff without manager
>>> diff = diff_configs(config_v1, config_v2)

**Deterministic Function Execution**

Run functions with guaranteed reproducibility:

>>> from insideLLMs.reproducibility import run_deterministic, DeterministicExecutor
>>> import random
>>>
>>> def stochastic_init():
...     return [random.random() for _ in range(5)]
>>>
>>> # Run deterministically
>>> result1, log1 = run_deterministic(stochastic_init, seed=42)
>>> result2, log2 = run_deterministic(stochastic_init, seed=42)
>>>
>>> # Results are identical
>>> assert result1 == result2
>>>
>>> # Using the executor class for multiple runs
>>> executor = DeterministicExecutor(seed=42)
>>> r1, _ = executor.execute(stochastic_init)
>>> r2, _ = executor.execute(stochastic_init)
>>> # Note: Each execute() resets seeds, so r1 == r2

**Seed Derivation for Components**

Derive deterministic seeds for different experiment components:

>>> from insideLLMs.reproducibility import derive_seed, SeedManager
>>>
>>> base_seed = 42
>>>
>>> # Derive seeds for different components
>>> model_seed = derive_seed(base_seed, "model_init")
>>> data_seed = derive_seed(base_seed, "data_shuffle")
>>> augment_seed = derive_seed(base_seed, "augmentation")
>>>
>>> print(f"Model seed: {model_seed}")
>>> print(f"Data seed: {data_seed}")
>>>
>>> # Same inputs always produce same derived seeds
>>> assert derive_seed(42, "model_init") == derive_seed(42, "model_init")
>>>
>>> # Different keys produce different seeds
>>> assert derive_seed(42, "model") != derive_seed(42, "data")

**Experiment Checkpointing**

Save and restore experiment state at various points:

>>> from insideLLMs.reproducibility import ExperimentCheckpointManager, SeedManager
>>>
>>> # Initialize checkpoint manager
>>> ckpt_manager = ExperimentCheckpointManager(checkpoint_dir=".checkpoints")
>>>
>>> # During training, create checkpoints
>>> seed_manager = SeedManager(global_seed=42)
>>> seed_manager.set_all_seeds()
>>>
>>> for epoch in range(10):
...     # ... training logic ...
...     model_state = {"epoch": epoch, "loss": 0.5 - epoch * 0.04}
...
...     if epoch % 5 == 0:
...         ckpt_id = ckpt_manager.create_checkpoint(
...             name=f"epoch_{epoch}",
...             state=model_state,
...             seed_state=seed_manager.get_state(),
...         )
...         print(f"Checkpoint created: {ckpt_id}")
>>>
>>> # List all checkpoints
>>> checkpoints = ckpt_manager.list_checkpoints()
>>> print(f"Available: {checkpoints}")
>>>
>>> # Restore from checkpoint
>>> # state = ckpt_manager.restore_checkpoint(ckpt_id)

**Experiment Registry**

Track and search experiments:

>>> from insideLLMs.reproducibility import ExperimentRegistry, ExperimentSnapshot
>>>
>>> # Create registry
>>> registry = ExperimentRegistry()
>>>
>>> # Create and register experiments
>>> snap1 = ExperimentSnapshot.capture(name="baseline", config={"lr": 0.01})
>>> snap1.metadata.tags = ["baseline", "v1"]
>>> registry.register(snap1)
>>>
>>> snap2 = ExperimentSnapshot.capture(name="improved", config={"lr": 0.001})
>>> snap2.metadata.tags = ["improved", "v1"]
>>> registry.register(snap2)
>>>
>>> # Search experiments
>>> v1_experiments = registry.find_by_tag("v1")
>>> print(f"Found {len(v1_experiments)} v1 experiments")
>>>
>>> baseline = registry.find_by_name("baseline")
>>> print(f"Baseline experiments: {len(baseline)}")
>>>
>>> # List all experiments
>>> all_experiments = registry.list_all()

**Reproducibility Checking**

Assess the reproducibility level of your setup:

>>> from insideLLMs.reproducibility import (
...     check_reproducibility,
...     SeedManager,
...     ExperimentSnapshot,
... )
>>>
>>> # Without any setup
>>> report = check_reproducibility()
>>> print(f"Level: {report.level.value}")  # 'none'
>>>
>>> # With seeds set
>>> seed_manager = SeedManager(global_seed=42)
>>> seed_manager.set_all_seeds()
>>> report = check_reproducibility(seed_manager=seed_manager)
>>> print(f"Level: {report.level.value}")  # 'seed_only'
>>>
>>> # With full snapshot
>>> snapshot = ExperimentSnapshot.capture("test", seed=42)
>>> report = check_reproducibility(snapshot=snapshot, seed_manager=seed_manager)
>>> print(f"Level: {report.level.value}")  # 'deterministic'
>>> print(f"Warnings: {report.warnings}")
>>> print(f"Recommendations: {report.recommendations}")

Notes
-----
**Canonical Import Location**

This module serves as a compatibility shim. The canonical implementation is
located at :mod:`insideLLMs.runtime.reproducibility`. Both import paths are
supported and functionally equivalent:

>>> # Both are valid and equivalent
>>> from insideLLMs.reproducibility import SeedManager
>>> from insideLLMs.runtime.reproducibility import SeedManager

**Library Support**

Seed management supports these libraries when available:

- **Python random**: Always available via stdlib
- **NumPy**: Supported if installed (numpy.random.seed)
- **PyTorch**: Supported if installed (torch.manual_seed, torch.cuda.manual_seed_all)
- **TensorFlow**: Supported if installed (tf.random.set_seed)

If a library is not installed, its seed-setting function returns False and
the library is skipped gracefully.

**Thread Safety**

The seed management functions modify global state in the underlying random
number generators. In multi-threaded applications, ensure seeds are set
before spawning threads, and consider using separate seed managers per thread
with derived seeds.

**Serialization**

Experiment snapshots can be saved in JSON format (default) or YAML format
(requires PyYAML). Pickle format is defined but should be used with caution
due to security implications.

**Reproducibility Limitations**

True reproducibility may require additional steps beyond seed setting:

1. Set PYTHONHASHSEED environment variable before Python starts
2. Use torch.backends.cudnn.deterministic = True for PyTorch
3. Avoid operations with non-deterministic implementations
4. Pin all package versions in requirements.txt
5. Consider hardware differences between machines

See Also
--------
insideLLMs.runtime.reproducibility : Canonical implementation module.
random : Python's built-in random module.
numpy.random : NumPy's random module.
torch.manual_seed : PyTorch seed setting.
tensorflow.random.set_seed : TensorFlow seed setting.

References
----------
.. [1] PyTorch Reproducibility Documentation
   https://pytorch.org/docs/stable/notes/randomness.html

.. [2] TensorFlow Determinism
   https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism

.. [3] NumPy Random Sampling
   https://numpy.org/doc/stable/reference/random/index.html
"""

from insideLLMs.runtime.reproducibility import *  # noqa: F401,F403
from insideLLMs.runtime.reproducibility import ExperimentCheckpointManager

# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

# Older code and tests may import CheckpointManager. The canonical name is
# ExperimentCheckpointManager.
CheckpointManager = ExperimentCheckpointManager
