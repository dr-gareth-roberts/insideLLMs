"""Benchmark command: run comprehensive benchmarks across models and probes."""

import argparse
import json
from pathlib import Path
from typing import Any

from insideLLMs.exceptions import ProbeExecutionError
from insideLLMs.registry import ensure_builtins_registered, model_registry, probe_registry

from .._output import (
    Colors,
    ProgressBar,
    colorize,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
    print_success,
    print_warning,
)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute the benchmark command."""
    ensure_builtins_registered()

    print_header("Running Benchmark Suite")

    models = [m.strip() for m in args.models.split(",")]
    probes = [p.strip() for p in args.probes.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    print_key_value("Models", ", ".join(models))
    print_key_value("Probes", ", ".join(probes))
    if datasets:
        print_key_value("Datasets", ", ".join(datasets))
    print_key_value("Max examples", args.max_examples)

    try:
        from insideLLMs.benchmark_datasets import (
            create_comprehensive_benchmark_suite,
            load_builtin_dataset,
        )

        # Load datasets
        if datasets:
            suite_examples = []
            for ds_name in datasets:
                ds = load_builtin_dataset(ds_name)
                for ex in ds.sample(args.max_examples, seed=42):
                    suite_examples.append(ex)
        else:
            suite = create_comprehensive_benchmark_suite(
                max_examples_per_dataset=args.max_examples, seed=42
            )
            suite_examples = list(suite.sample(args.max_examples * 5, seed=42))

        print_info(f"Loaded {len(suite_examples)} benchmark examples")

        results_all: list[dict[str, Any]] = []

        for model_name in models:
            print_subheader(f"Model: {model_name}")

            try:
                model_or_factory = model_registry.get(model_name)
                model = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
            except Exception as e:
                print_warning(f"Could not load model {model_name}: {e}")
                continue

            for probe_name in probes:
                print(f"  Running probe: {colorize(probe_name, Colors.GREEN)}")

                try:
                    probe_or_factory = probe_registry.get(probe_name)
                    probe = (
                        probe_or_factory()
                        if isinstance(probe_or_factory, type)
                        else probe_or_factory
                    )
                except Exception as e:
                    print_warning(f"  Could not load probe {probe_name}: {e}")
                    continue

                # Create runner and run
                from insideLLMs.runtime.runner import ProbeRunner

                runner = ProbeRunner(model, probe)

                inputs = [ex.input_text for ex in suite_examples[: args.max_examples]]
                progress = ProgressBar(len(inputs), prefix=f"  {probe_name}")

                probe_results = []
                for i, inp in enumerate(inputs):
                    try:
                        result = runner.run_single(inp)
                        probe_results.append(result)
                    except (ProbeExecutionError, RuntimeError):
                        pass
                    progress.update(i + 1)

                progress.finish()

                success_count = sum(
                    1
                    for r in probe_results
                    if getattr(r, "status", None) == "success"
                    or (isinstance(r, dict) and r.get("status") == "success")
                )
                results_all.append(
                    {
                        "model": model_name,
                        "probe": probe_name,
                        "total": len(inputs),
                        "success": success_count,
                        "accuracy": success_count / max(1, len(inputs)),
                    }
                )

        # Summary
        print_subheader("Benchmark Results Summary")
        print(f"\n  {'Model':<15} {'Probe':<15} {'Accuracy':>10} {'Success':>10}")
        print("  " + "-" * 55)
        for r in results_all:
            print(
                f"  {r['model']:<15} {r['probe']:<15} {r['accuracy'] * 100:>9.1f}% {r['success']:>5}/{r['total']}"
            )

        # Save results if output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / "benchmark_results.json"
            with open(results_file, "w") as f:
                json.dump(results_all, f, indent=2)
            print_success(f"Results saved to: {results_file}")

            if args.html_report:
                print_info(
                    "HTML report generation for `benchmark` requires ExperimentResult format; "
                    "use `insidellms harness` + `insidellms report`."
                )

        return 0

    except Exception as e:
        print_error(f"Benchmark error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
