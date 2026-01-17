"""Example: Experiment Tracking

This example demonstrates how to use the experiment tracking module
to log experiments to various backends including local files, W&B,
MLflow, and TensorBoard.
"""

import tempfile
from pathlib import Path

from insideLLMs import (
    # Experiment tracking
    LocalFileTracker,
    MultiTracker,
    TrackingConfig,
    auto_track,
    create_tracker,
    # Types for experiment results
    ExperimentResult,
    ModelInfo,
    ProbeCategory,
    ProbeResult,
    ProbeScore,
    ResultStatus,
)


def basic_local_tracking():
    """Basic example with local file tracking."""
    print("=" * 60)
    print("Basic Local File Tracking")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a local tracker
        tracker = LocalFileTracker(output_dir=tmp_dir)

        # Start a run
        tracker.start_run(run_name="my-first-experiment")

        # Log parameters
        tracker.log_params({
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "probe": "LogicProbe",
        })

        # Log metrics
        tracker.log_metrics({"accuracy": 0.85, "latency_ms": 150})
        tracker.log_metrics({"accuracy": 0.90, "latency_ms": 120}, step=1)
        tracker.log_metrics({"accuracy": 0.92, "latency_ms": 110}, step=2)

        # End the run
        tracker.end_run()

        # List runs
        runs = tracker.list_runs()
        print(f"\nRuns recorded: {runs}")

        # Load a run
        data = tracker.load_run("my-first-experiment")
        print(f"Params: {data['params']}")
        print(f"Metrics: {len(data['metrics'])} entries")


def context_manager_tracking():
    """Using tracker as context manager."""
    print("\n" + "=" * 60)
    print("Context Manager Tracking")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TrackingConfig(
            project="my-llm-project",
            experiment_name="context-test",
            tags=["test", "example"],
            notes="Testing context manager usage",
        )

        tracker = LocalFileTracker(output_dir=tmp_dir, config=config)

        # Use as context manager
        with tracker:
            tracker.log_params({"batch_size": 32})
            tracker.log_metrics({"loss": 0.5})

        print("Run completed and saved automatically!")

        # Verify
        runs = tracker.list_runs()
        print(f"Runs: {runs}")


def auto_track_decorator():
    """Using the auto_track decorator."""
    print("\n" + "=" * 60)
    print("Auto-Track Decorator")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = LocalFileTracker(output_dir=tmp_dir)

        @auto_track(tracker, experiment_name="decorated-experiment")
        def run_evaluation():
            """Simulated evaluation function."""
            # The decorator will track this function
            return {"accuracy": 0.95, "precision": 0.92, "recall": 0.94}

        # Run the decorated function
        result = run_evaluation()
        print(f"Function returned: {result}")

        # Check that it was tracked
        data = tracker.load_run("decorated-experiment")
        print(f"Status: {data['final_state']['status']}")


def log_experiment_result():
    """Logging an ExperimentResult object."""
    print("\n" + "=" * 60)
    print("Logging ExperimentResult Objects")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = LocalFileTracker(output_dir=tmp_dir)

        # Create a mock ExperimentResult
        results = [
            ProbeResult(
                input=f"test input {i}",
                output=f"test output {i}",
                status=ResultStatus.SUCCESS,
                latency_ms=100.0 + i * 10,
            )
            for i in range(10)
        ]

        experiment = ExperimentResult(
            experiment_id="exp-001",
            model_info=ModelInfo(
                name="GPT-4",
                provider="openai",
                model_id="gpt-4-turbo",
            ),
            probe_name="LogicProbe",
            probe_category=ProbeCategory.LOGIC,
            results=results,
            score=ProbeScore(
                accuracy=0.9,
                precision=0.85,
                recall=0.88,
                f1_score=0.865,
                mean_latency_ms=150.0,
            ),
        )

        tracker.start_run(run_name="experiment-result-demo")

        # Log the experiment result
        tracker.log_experiment_result(experiment, prefix="eval_")

        tracker.end_run()

        # Verify
        data = tracker.load_run("experiment-result-demo")
        print(f"Logged params: {list(data['params'].keys())}")
        print(f"Logged metrics: {len(data['metrics'])} entries")


def multi_tracker_example():
    """Using multiple trackers simultaneously."""
    print("\n" + "=" * 60)
    print("Multi-Tracker Example")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create multiple trackers
        tracker1 = LocalFileTracker(output_dir=str(Path(tmp_dir) / "tracker1"))
        tracker2 = LocalFileTracker(output_dir=str(Path(tmp_dir) / "tracker2"))

        # Combine them
        multi = MultiTracker([tracker1, tracker2])

        with multi:
            multi.log_params({"model": "gpt-4"})
            multi.log_metrics({"accuracy": 0.95})

        # Both trackers have the data
        print(f"Tracker 1 runs: {tracker1.list_runs()}")
        print(f"Tracker 2 runs: {tracker2.list_runs()}")


def show_available_backends():
    """Show available tracking backends."""
    print("\n" + "=" * 60)
    print("Available Tracking Backends")
    print("=" * 60)

    print("""
The following backends are supported:

1. LOCAL FILE TRACKER (always available)
   tracker = create_tracker("local", output_dir="./experiments")

2. WEIGHTS & BIASES (requires `pip install wandb`)
   tracker = create_tracker("wandb", project="my-project")

3. MLFLOW (requires `pip install mlflow`)
   tracker = create_tracker("mlflow", experiment_name="my-exp")

4. TENSORBOARD (requires `pip install tensorboard`)
   tracker = create_tracker("tensorboard", log_dir="./logs")

Each tracker supports the same interface:
- start_run(run_name)
- log_params(dict)
- log_metrics(dict, step)
- log_artifact(path, name)
- log_experiment_result(result, prefix)
- end_run()
""")


if __name__ == "__main__":
    basic_local_tracking()
    context_manager_tracking()
    auto_track_decorator()
    log_experiment_result()
    multi_tracker_example()
    show_available_backends()

    print("\n" + "=" * 60)
    print("Done! See the experiment_tracking module for more details.")
    print("=" * 60)
