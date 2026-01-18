"""Tests for the experiment reproducibility and snapshot module."""

import os
import random
import tempfile

import pytest

from insideLLMs.reproducibility import (
    CheckpointManager,
    ConfigSnapshot,
    ConfigVersionManager,
    DeterministicExecutor,
    EnvironmentCapture,
    EnvironmentInfo,
    EnvironmentType,
    ExperimentMetadata,
    ExperimentRegistry,
    ExperimentReplayManager,
    ExperimentSnapshot,
    ReplayResult,
    ReproducibilityChecker,
    ReproducibilityLevel,
    ReproducibilityReport,
    # Classes
    SeedManager,
    # Dataclasses
    SeedState,
    # Enums
    SnapshotFormat,
    capture_environment,
    capture_snapshot,
    check_reproducibility,
    compare_environments,
    # Functions
    create_seed_manager,
    derive_seed,
    diff_configs,
    load_snapshot,
    run_deterministic,
    save_snapshot,
    set_global_seed,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestSnapshotFormat:
    """Tests for SnapshotFormat enum."""

    def test_all_formats_exist(self):
        """Test all formats exist."""
        assert SnapshotFormat.JSON.value == "json"
        assert SnapshotFormat.YAML.value == "yaml"
        assert SnapshotFormat.PICKLE.value == "pickle"


class TestReproducibilityLevel:
    """Tests for ReproducibilityLevel enum."""

    def test_all_levels_exist(self):
        """Test all levels exist."""
        assert ReproducibilityLevel.NONE.value == "none"
        assert ReproducibilityLevel.SEED_ONLY.value == "seed_only"
        assert ReproducibilityLevel.DETERMINISTIC.value == "deterministic"
        assert ReproducibilityLevel.VERIFIED.value == "verified"


class TestEnvironmentType:
    """Tests for EnvironmentType enum."""

    def test_all_types_exist(self):
        """Test all types exist."""
        assert EnvironmentType.PYTHON.value == "python"
        assert EnvironmentType.SYSTEM.value == "system"
        assert EnvironmentType.PACKAGES.value == "packages"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestSeedState:
    """Tests for SeedState dataclass."""

    def test_create_seed_state(self):
        """Test creating a seed state."""
        state = SeedState(global_seed=42)

        assert state.global_seed == 42
        assert state.python_random_state is None

    def test_seed_state_to_dict(self):
        """Test seed state serialization."""
        state = SeedState(global_seed=42)

        d = state.to_dict()

        assert d["global_seed"] == 42
        assert "timestamp" in d


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo dataclass."""

    def test_create_environment_info(self):
        """Test creating environment info."""
        info = EnvironmentInfo(
            python_version="3.12.0",
            platform_info="Linux-5.4.0",
            os_info="Linux 5.4.0",
        )

        assert info.python_version == "3.12.0"
        assert info.platform_info == "Linux-5.4.0"

    def test_environment_info_to_dict(self):
        """Test environment info serialization."""
        info = EnvironmentInfo(
            python_version="3.12.0",
            platform_info="Linux",
            os_info="Linux 5.4.0",
            packages={"numpy": "1.24.0"},
        )

        d = info.to_dict()

        assert d["python_version"] == "3.12.0"
        assert d["packages"]["numpy"] == "1.24.0"


class TestConfigSnapshot:
    """Tests for ConfigSnapshot dataclass."""

    def test_create_config_snapshot(self):
        """Test creating config snapshot."""
        config = {"model": "gpt-4", "temperature": 0.7}
        snapshot = ConfigSnapshot(
            config=config,
            config_hash="abc123",
        )

        assert snapshot.config["model"] == "gpt-4"
        assert snapshot.config_hash == "abc123"

    def test_config_snapshot_to_dict(self):
        """Test config snapshot serialization."""
        snapshot = ConfigSnapshot(
            config={"key": "value"},
            config_hash="hash123",
            version="2.0",
        )

        d = snapshot.to_dict()

        assert d["config"]["key"] == "value"
        assert d["version"] == "2.0"


class TestExperimentMetadata:
    """Tests for ExperimentMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = ExperimentMetadata(
            experiment_id="exp_001",
            name="test_experiment",
            description="A test experiment",
            tags=["test", "unit"],
        )

        assert metadata.experiment_id == "exp_001"
        assert metadata.name == "test_experiment"
        assert "test" in metadata.tags

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = ExperimentMetadata(
            experiment_id="exp_001",
            name="test",
        )

        d = metadata.to_dict()

        assert d["experiment_id"] == "exp_001"
        assert "created_at" in d


class TestReplayResult:
    """Tests for ReplayResult dataclass."""

    def test_create_replay_result(self):
        """Test creating replay result."""
        result = ReplayResult(
            success=True,
            original_hash="abc",
            replay_hash="abc",
            matches=True,
        )

        assert result.success is True
        assert result.matches is True

    def test_replay_result_with_differences(self):
        """Test replay result with differences."""
        result = ReplayResult(
            success=True,
            original_hash="abc",
            replay_hash="def",
            matches=False,
            differences=["output changed"],
        )

        assert result.matches is False
        assert len(result.differences) == 1


class TestReproducibilityReport:
    """Tests for ReproducibilityReport dataclass."""

    def test_create_report(self):
        """Test creating reproducibility report."""
        report = ReproducibilityReport(
            level=ReproducibilityLevel.DETERMINISTIC,
            seed_set=True,
            environment_captured=True,
            config_versioned=True,
            checkpoints_available=False,
        )

        assert report.level == ReproducibilityLevel.DETERMINISTIC
        assert report.seed_set is True

    def test_report_to_dict(self):
        """Test report serialization."""
        report = ReproducibilityReport(
            level=ReproducibilityLevel.SEED_ONLY,
            seed_set=True,
            environment_captured=False,
            config_versioned=False,
            checkpoints_available=False,
            warnings=["No environment captured"],
        )

        d = report.to_dict()

        assert d["level"] == "seed_only"
        assert len(d["warnings"]) == 1


# =============================================================================
# SeedManager Tests
# =============================================================================


class TestSeedManager:
    """Tests for SeedManager class."""

    def test_create_with_seed(self):
        """Test creating seed manager with specific seed."""
        manager = SeedManager(global_seed=42)

        assert manager.global_seed == 42

    def test_create_without_seed(self):
        """Test creating seed manager without seed generates one."""
        manager = SeedManager()

        assert manager.global_seed is not None
        assert isinstance(manager.global_seed, int)

    def test_set_python_seed(self):
        """Test setting Python random seed."""
        manager = SeedManager(global_seed=42)
        manager.set_python_seed()

        # Verify deterministic behavior
        val1 = random.random()

        manager.set_python_seed()
        val2 = random.random()

        assert val1 == val2

    def test_set_all_seeds(self):
        """Test setting all available seeds."""
        manager = SeedManager(global_seed=42)

        results = manager.set_all_seeds()

        assert results["python_random"] is True
        assert "numpy" in results
        assert "torch" in results

    def test_get_state(self):
        """Test getting seed state."""
        manager = SeedManager(global_seed=42)
        manager.set_all_seeds()

        state = manager.get_state()

        assert state.global_seed == 42
        assert state.python_random_state is not None

    def test_restore_state(self):
        """Test restoring seed state."""
        manager = SeedManager(global_seed=42)
        manager.set_python_seed()

        # Generate some random numbers
        random.random()
        random.random()

        # Capture state
        state = manager.get_state()

        # Generate more numbers
        val1 = random.random()

        # Restore state
        manager.restore_state(state)
        val2 = random.random()

        assert val1 == val2

    def test_get_libraries_seeded(self):
        """Test getting list of seeded libraries."""
        manager = SeedManager(global_seed=42)
        manager.set_python_seed()

        libs = manager.get_libraries_seeded()

        assert "python_random" in libs

    def test_derive_seed(self):
        """Test deriving deterministic seed from key."""
        manager = SeedManager(global_seed=42)

        seed1 = manager.derive_seed("task_1")
        seed2 = manager.derive_seed("task_2")
        seed3 = manager.derive_seed("task_1")

        assert seed1 != seed2  # Different keys give different seeds
        assert seed1 == seed3  # Same key gives same seed


# =============================================================================
# EnvironmentCapture Tests
# =============================================================================


class TestEnvironmentCapture:
    """Tests for EnvironmentCapture class."""

    def test_capture_environment(self):
        """Test capturing environment."""
        capture = EnvironmentCapture(capture_env_vars=False)

        info = capture.capture()

        assert info.python_version is not None
        assert info.platform_info is not None
        assert info.working_directory != ""

    def test_capture_with_env_vars(self):
        """Test capturing with environment variables."""
        os.environ["TEST_VAR_REPRO"] = "test_value"

        capture = EnvironmentCapture(capture_env_vars=True)
        info = capture.capture()

        assert "TEST_VAR_REPRO" in info.env_vars

        del os.environ["TEST_VAR_REPRO"]

    def test_capture_with_prefix_filter(self):
        """Test capturing env vars with prefix filter."""
        os.environ["MYAPP_VAR1"] = "value1"
        os.environ["OTHER_VAR"] = "value2"

        capture = EnvironmentCapture(capture_env_vars=True, env_var_prefix="MYAPP_")
        info = capture.capture()

        assert "MYAPP_VAR1" in info.env_vars
        assert "OTHER_VAR" not in info.env_vars

        del os.environ["MYAPP_VAR1"]
        del os.environ["OTHER_VAR"]

    def test_capture_packages(self):
        """Test capturing installed packages."""
        capture = EnvironmentCapture(capture_env_vars=False)
        info = capture.capture()

        # Should have some packages
        assert len(info.packages) > 0

    def test_compare_environments(self):
        """Test comparing two environments."""
        capture = EnvironmentCapture(capture_env_vars=False)

        env1 = capture.capture()
        env2 = capture.capture()

        comparison = capture.compare(env1, env2)

        assert comparison["python_version_match"] is True
        assert comparison["is_compatible"] is True

    def test_compare_different_environments(self):
        """Test comparing different environments."""
        capture = EnvironmentCapture()

        env1 = EnvironmentInfo(
            python_version="3.11.0",
            platform_info="Linux",
            os_info="Linux",
            packages={"numpy": "1.24.0"},
        )

        env2 = EnvironmentInfo(
            python_version="3.12.0",
            platform_info="Linux",
            os_info="Linux",
            packages={"numpy": "1.25.0"},
        )

        comparison = capture.compare(env1, env2)

        assert comparison["python_version_match"] is False
        assert "numpy" in comparison["package_differences"]


# =============================================================================
# ExperimentSnapshot Tests
# =============================================================================


class TestExperimentSnapshot:
    """Tests for ExperimentSnapshot class."""

    def test_capture_snapshot(self):
        """Test capturing a snapshot."""
        snapshot = ExperimentSnapshot.capture(
            name="test_experiment",
            config={"model": "gpt-4"},
            seed=42,
        )

        assert snapshot.snapshot_id.startswith("snap_")
        assert snapshot.metadata.name == "test_experiment"
        assert snapshot.config.config["model"] == "gpt-4"
        assert snapshot.seed_state.global_seed == 42

    def test_snapshot_checksum(self):
        """Test snapshot checksum calculation."""
        snapshot = ExperimentSnapshot.capture(
            name="test",
            config={"key": "value"},
            seed=42,
        )

        assert snapshot.checksum != ""
        assert len(snapshot.checksum) == 16

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)

        d = snapshot.to_dict()

        assert "snapshot_id" in d
        assert "metadata" in d
        assert "config" in d
        assert "seed_state" in d
        assert "environment" in d

    def test_save_and_load_json(self):
        """Test saving and loading snapshot as JSON."""
        snapshot = ExperimentSnapshot.capture(
            name="test",
            config={"model": "gpt-4"},
            seed=42,
        )
        snapshot.inputs = {"prompt": "Hello"}
        snapshot.outputs = {"response": "World"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            snapshot.save(path)
            loaded = ExperimentSnapshot.load(path)

            assert loaded.snapshot_id == snapshot.snapshot_id
            assert loaded.config.config["model"] == "gpt-4"
            assert loaded.seed_state.global_seed == 42
            assert loaded.inputs["prompt"] == "Hello"
        finally:
            os.unlink(path)


# =============================================================================
# ConfigVersionManager Tests
# =============================================================================


class TestConfigVersionManager:
    """Tests for ConfigVersionManager class."""

    def test_add_version(self):
        """Test adding a config version."""
        manager = ConfigVersionManager()

        snapshot = manager.add_version("v1.0", {"model": "gpt-4"})

        assert snapshot.version == "v1.0"
        assert snapshot.config["model"] == "gpt-4"
        assert snapshot.config_hash != ""

    def test_get_version(self):
        """Test getting a specific version."""
        manager = ConfigVersionManager()
        manager.add_version("v1.0", {"key": "value"})

        version = manager.get_version("v1.0")

        assert version is not None
        assert version.config["key"] == "value"

    def test_get_nonexistent_version(self):
        """Test getting nonexistent version."""
        manager = ConfigVersionManager()

        version = manager.get_version("v999")

        assert version is None

    def test_get_current(self):
        """Test getting current version."""
        manager = ConfigVersionManager()
        manager.add_version("v1.0", {"a": 1})
        manager.add_version("v2.0", {"a": 2})

        current = manager.get_current()

        assert current is not None
        assert current.version == "v2.0"

    def test_list_versions(self):
        """Test listing all versions."""
        manager = ConfigVersionManager()
        manager.add_version("v1.0", {})
        manager.add_version("v2.0", {})

        versions = manager.list_versions()

        assert "v1.0" in versions
        assert "v2.0" in versions

    def test_diff_versions(self):
        """Test diffing two versions."""
        manager = ConfigVersionManager()
        manager.add_version("v1.0", {"model": "gpt-3.5", "temp": 0.7})
        manager.add_version("v2.0", {"model": "gpt-4", "temp": 0.7, "new_key": True})

        diff = manager.diff("v1.0", "v2.0")

        assert "model" in diff["changed"]
        assert diff["changed"]["model"]["old"] == "gpt-3.5"
        assert diff["changed"]["model"]["new"] == "gpt-4"
        assert "new_key" in diff["added"]

    def test_diff_nonexistent_version(self):
        """Test diffing with nonexistent version."""
        manager = ConfigVersionManager()
        manager.add_version("v1.0", {})

        diff = manager.diff("v1.0", "v999")

        assert "error" in diff


# =============================================================================
# ExperimentReplayManager Tests
# =============================================================================


class TestExperimentReplayManager:
    """Tests for ExperimentReplayManager class."""

    def test_register_snapshot(self):
        """Test registering a snapshot."""
        manager = ExperimentReplayManager()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)

        manager.register_snapshot(snapshot)

        assert snapshot.snapshot_id in manager._snapshots

    def test_setup_for_replay(self):
        """Test setting up for replay."""
        manager = ExperimentReplayManager()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)
        manager.register_snapshot(snapshot)

        seed_manager = manager.setup_for_replay(snapshot.snapshot_id)

        assert seed_manager.global_seed == 42

    def test_setup_for_replay_not_found(self):
        """Test setup for nonexistent snapshot."""
        manager = ExperimentReplayManager()

        with pytest.raises(ValueError):
            manager.setup_for_replay("nonexistent")

    def test_verify_replay_match(self):
        """Test verifying matching replay."""
        manager = ExperimentReplayManager()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)
        snapshot.outputs = {"result": "42"}
        manager.register_snapshot(snapshot)

        result = manager.verify_replay(snapshot.snapshot_id, {"result": "42"})

        assert result.success is True
        assert result.matches is True

    def test_verify_replay_mismatch(self):
        """Test verifying mismatching replay."""
        manager = ExperimentReplayManager()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)
        snapshot.outputs = {"result": "42"}
        manager.register_snapshot(snapshot)

        result = manager.verify_replay(snapshot.snapshot_id, {"result": "43"})

        assert result.success is True
        assert result.matches is False
        assert len(result.differences) > 0


# =============================================================================
# DeterministicExecutor Tests
# =============================================================================


class TestDeterministicExecutor:
    """Tests for DeterministicExecutor class."""

    def test_execute_deterministic(self):
        """Test deterministic execution."""

        def random_func():
            return random.random()

        executor = DeterministicExecutor(seed=42)

        result1, _ = executor.execute(random_func)
        result2, _ = executor.execute(random_func)

        assert result1 == result2

    def test_execute_logs_execution(self):
        """Test execution logging."""

        def simple_func(x):
            return x * 2

        executor = DeterministicExecutor(seed=42)
        executor.execute(simple_func, 5)

        log = executor.get_execution_log()

        assert len(log) == 1
        assert log[0]["function"] == "simple_func"
        assert log[0]["seed"] == 42

    def test_execute_with_args_kwargs(self):
        """Test execution with arguments."""

        def add_func(a, b, multiplier=1):
            return (a + b) * multiplier

        executor = DeterministicExecutor(seed=42)
        result, _ = executor.execute(add_func, 2, 3, multiplier=2)

        assert result == 10


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        manager = CheckpointManager()

        checkpoint_id = manager.create_checkpoint(
            "step_1",
            {"progress": 50, "data": [1, 2, 3]},
        )

        assert checkpoint_id.startswith("ckpt_step_1_")

    def test_get_checkpoint(self):
        """Test getting a checkpoint."""
        manager = CheckpointManager()
        ckpt_id = manager.create_checkpoint("test", {"value": 42})

        checkpoint = manager.get_checkpoint(ckpt_id)

        assert checkpoint is not None
        assert checkpoint["state"]["value"] == 42

    def test_get_nonexistent_checkpoint(self):
        """Test getting nonexistent checkpoint."""
        manager = CheckpointManager()

        checkpoint = manager.get_checkpoint("nonexistent")

        assert checkpoint is None

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        manager = CheckpointManager()
        manager.create_checkpoint("step_1", {})
        manager.create_checkpoint("step_2", {})

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2

    def test_restore_checkpoint(self):
        """Test restoring checkpoint."""
        manager = CheckpointManager()
        state = {"progress": 75, "results": [1, 2, 3]}
        ckpt_id = manager.create_checkpoint("mid_run", state)

        restored = manager.restore_checkpoint(ckpt_id)

        assert restored == state

    def test_checkpoint_with_seed_state(self):
        """Test checkpoint with seed state."""
        manager = CheckpointManager()
        seed_state = SeedState(global_seed=42)

        ckpt_id = manager.create_checkpoint(
            "seeded",
            {"data": "value"},
            seed_state=seed_state,
        )

        checkpoint = manager.get_checkpoint(ckpt_id)

        assert checkpoint["seed_state"]["global_seed"] == 42


# =============================================================================
# ReproducibilityChecker Tests
# =============================================================================


class TestReproducibilityChecker:
    """Tests for ReproducibilityChecker class."""

    def test_check_no_reproducibility(self):
        """Test checking with no reproducibility setup."""
        checker = ReproducibilityChecker()

        report = checker.check()

        assert report.level == ReproducibilityLevel.NONE
        assert report.seed_set is False

    def test_check_with_seed_only(self):
        """Test checking with seed only."""
        checker = ReproducibilityChecker()
        seed_manager = SeedManager(global_seed=42)
        seed_manager.set_python_seed()

        report = checker.check(seed_manager=seed_manager)

        assert report.level == ReproducibilityLevel.SEED_ONLY
        assert report.seed_set is True

    def test_check_with_snapshot(self):
        """Test checking with full snapshot."""
        checker = ReproducibilityChecker()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)
        seed_manager = SeedManager(global_seed=42)
        seed_manager.set_python_seed()

        report = checker.check(snapshot=snapshot, seed_manager=seed_manager)

        assert report.level == ReproducibilityLevel.DETERMINISTIC


# =============================================================================
# ExperimentRegistry Tests
# =============================================================================


class TestExperimentRegistry:
    """Tests for ExperimentRegistry class."""

    def test_register_experiment(self):
        """Test registering an experiment."""
        registry = ExperimentRegistry()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)

        exp_id = registry.register(snapshot)

        assert exp_id == snapshot.snapshot_id

    def test_get_experiment(self):
        """Test getting an experiment."""
        registry = ExperimentRegistry()
        snapshot = ExperimentSnapshot.capture(name="test", seed=42)
        registry.register(snapshot)

        retrieved = registry.get(snapshot.snapshot_id)

        assert retrieved is not None
        assert retrieved.metadata.name == "test"

    def test_find_by_tag(self):
        """Test finding experiments by tag."""
        registry = ExperimentRegistry()

        snapshot1 = ExperimentSnapshot.capture(name="exp1", seed=42)
        snapshot1.metadata.tags = ["ml", "test"]
        registry.register(snapshot1)

        snapshot2 = ExperimentSnapshot.capture(name="exp2", seed=43)
        snapshot2.metadata.tags = ["ml", "production"]
        registry.register(snapshot2)

        ml_exps = registry.find_by_tag("ml")
        test_exps = registry.find_by_tag("test")

        assert len(ml_exps) == 2
        assert len(test_exps) == 1

    def test_find_by_name(self):
        """Test finding experiments by name."""
        registry = ExperimentRegistry()

        snapshot1 = ExperimentSnapshot.capture(name="benchmark", seed=42)
        registry.register(snapshot1)

        snapshot2 = ExperimentSnapshot.capture(name="benchmark", seed=43)
        registry.register(snapshot2)

        found = registry.find_by_name("benchmark")

        assert len(found) == 2

    def test_list_all(self):
        """Test listing all experiments."""
        registry = ExperimentRegistry()
        registry.register(ExperimentSnapshot.capture(name="exp1", seed=1))
        registry.register(ExperimentSnapshot.capture(name="exp2", seed=2))

        all_exps = registry.list_all()

        assert len(all_exps) == 2


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_seed_manager_func(self):
        """Test create_seed_manager function."""
        manager = create_seed_manager(42)

        assert manager.global_seed == 42

    def test_create_seed_manager_no_seed(self):
        """Test create_seed_manager without seed."""
        manager = create_seed_manager()

        assert manager.global_seed is not None

    def test_set_global_seed_func(self):
        """Test set_global_seed function."""
        results = set_global_seed(42)

        assert results["python_random"] is True

        # Verify deterministic
        val1 = random.random()
        set_global_seed(42)
        val2 = random.random()

        assert val1 == val2

    def test_capture_snapshot_func(self):
        """Test capture_snapshot function."""
        snapshot = capture_snapshot(
            name="test",
            config={"key": "value"},
            seed=42,
        )

        assert snapshot.metadata.name == "test"
        assert snapshot.seed_state.global_seed == 42

    def test_save_and_load_snapshot_funcs(self):
        """Test save_snapshot and load_snapshot functions."""
        snapshot = capture_snapshot(name="test", seed=42)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_snapshot(snapshot, path)
            loaded = load_snapshot(path)

            assert loaded.snapshot_id == snapshot.snapshot_id
        finally:
            os.unlink(path)

    def test_capture_environment_func(self):
        """Test capture_environment function."""
        env = capture_environment()

        assert env.python_version is not None
        assert env.working_directory != ""

    def test_compare_environments_func(self):
        """Test compare_environments function."""
        env1 = capture_environment()
        env2 = capture_environment()

        comparison = compare_environments(env1, env2)

        assert comparison["python_version_match"] is True

    def test_diff_configs_func(self):
        """Test diff_configs function."""
        config1 = {"a": 1, "b": 2}
        config2 = {"a": 1, "b": 3, "c": 4}

        diff = diff_configs(config1, config2)

        assert "b" in diff["changed"]
        assert "c" in diff["added"]

    def test_check_reproducibility_func(self):
        """Test check_reproducibility function."""
        report = check_reproducibility()

        assert report.level is not None

    def test_run_deterministic_func(self):
        """Test run_deterministic function."""

        def random_func():
            return random.random()

        result1, _ = run_deterministic(random_func, seed=42)
        result2, _ = run_deterministic(random_func, seed=42)

        assert result1 == result2

    def test_derive_seed_func(self):
        """Test derive_seed function."""
        seed1 = derive_seed(42, "task_a")
        seed2 = derive_seed(42, "task_b")
        seed3 = derive_seed(42, "task_a")

        assert seed1 != seed2
        assert seed1 == seed3


# =============================================================================
# Integration Tests
# =============================================================================


class TestReproducibilityIntegration:
    """Integration tests for reproducibility workflow."""

    def test_full_reproducibility_workflow(self):
        """Test complete reproducibility workflow."""
        # 1. Setup reproducibility
        seed_manager = create_seed_manager(42)
        seed_manager.set_all_seeds()

        # 2. Capture initial snapshot
        config = {"model": "gpt-4", "temperature": 0.7}
        snapshot = capture_snapshot(
            name="test_experiment",
            config=config,
            seed=42,
        )

        # 3. Run "experiment"
        random_results = [random.random() for _ in range(5)]
        snapshot.outputs = {"results": random_results}

        # 4. Save snapshot
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_snapshot(snapshot, path)

            # 5. Load and replay
            loaded = load_snapshot(path)
            replay_manager = ExperimentReplayManager()
            replay_manager.register_snapshot(loaded)

            # Setup replay environment
            seed_manager = replay_manager.setup_for_replay(loaded.snapshot_id)

            # Run same experiment
            replay_results = [random.random() for _ in range(5)]

            # 6. Verify
            verify_result = replay_manager.verify_replay(
                loaded.snapshot_id,
                {"results": replay_results},
            )

            assert verify_result.matches is True
        finally:
            os.unlink(path)

    def test_config_versioning_workflow(self):
        """Test config versioning workflow."""
        manager = ConfigVersionManager()

        # Add initial version
        manager.add_version(
            "v1.0",
            {
                "model": "gpt-3.5",
                "max_tokens": 100,
            },
        )

        # Add updated version
        manager.add_version(
            "v2.0",
            {
                "model": "gpt-4",
                "max_tokens": 200,
                "temperature": 0.7,
            },
        )

        # Compare versions
        diff = manager.diff("v1.0", "v2.0")

        assert diff["changed"]["model"]["old"] == "gpt-3.5"
        assert diff["changed"]["model"]["new"] == "gpt-4"
        assert diff["changed"]["max_tokens"]["old"] == 100
        assert diff["added"]["temperature"] == 0.7

    def test_checkpoint_restore_workflow(self):
        """Test checkpoint and restore workflow."""
        checkpoint_manager = CheckpointManager()

        # Simulate experiment progress
        state1 = {"epoch": 1, "loss": 0.5}
        ckpt1 = checkpoint_manager.create_checkpoint("epoch_1", state1)

        state2 = {"epoch": 2, "loss": 0.3}
        checkpoint_manager.create_checkpoint("epoch_2", state2)

        # Restore from earlier checkpoint
        restored = checkpoint_manager.restore_checkpoint(ckpt1)

        assert restored["epoch"] == 1
        assert restored["loss"] == 0.5
