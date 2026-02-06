"""List command: display available models, probes, datasets, and trackers."""

import argparse

from insideLLMs.registry import ensure_builtins_registered, model_registry, probe_registry

from .._output import Colors, colorize, print_subheader, print_warning


def cmd_list(args: argparse.Namespace) -> int:
    """Execute the list command."""
    ensure_builtins_registered()

    filter_str = args.filter.lower() if args.filter else None

    if args.type in ("models", "all"):
        print_subheader("Available Models")
        models = model_registry.list()
        if filter_str:
            models = [m for m in models if filter_str in m.lower()]

        for name in sorted(models):
            info = model_registry.info(name)
            doc = info.get("doc", "").split("\n")[0] if info.get("doc") else ""
            if args.detailed:
                print(f"\n  {colorize(name, Colors.BOLD, Colors.CYAN)}")
                print(f"    {colorize('Description:', Colors.DIM)} {doc[:70]}")
                if info.get("default_kwargs"):
                    print(f"    {colorize('Defaults:', Colors.DIM)} {info['default_kwargs']}")
            else:
                print(f"  {colorize(name, Colors.CYAN):25} {doc[:50]}")

        print(f"\n  {colorize(f'Total: {len(models)} models', Colors.DIM)}")

    if args.type in ("probes", "all"):
        print_subheader("Available Probes")
        probes = probe_registry.list()
        if filter_str:
            probes = [p for p in probes if filter_str in p.lower()]

        for name in sorted(probes):
            info = probe_registry.info(name)
            doc = info.get("doc", "").split("\n")[0] if info.get("doc") else ""
            if args.detailed:
                print(f"\n  {colorize(name, Colors.BOLD, Colors.GREEN)}")
                print(f"    {colorize('Description:', Colors.DIM)} {doc[:70]}")
            else:
                print(f"  {colorize(name, Colors.GREEN):25} {doc[:50]}")

        print(f"\n  {colorize(f'Total: {len(probes)} probes', Colors.DIM)}")

    if args.type in ("datasets", "all"):
        print_subheader("Built-in Benchmark Datasets")
        try:
            from insideLLMs.benchmark_datasets import list_builtin_datasets

            datasets = list_builtin_datasets()

            if filter_str:
                datasets = [d for d in datasets if filter_str in d["name"].lower()]

            for ds in datasets:
                if args.detailed:
                    print(f"\n  {colorize(ds['name'], Colors.BOLD, Colors.YELLOW)}")
                    print(f"    {colorize('Category:', Colors.DIM)} {ds['category']}")
                    print(f"    {colorize('Examples:', Colors.DIM)} {ds['num_examples']}")
                    print(f"    {colorize('Description:', Colors.DIM)} {ds['description'][:60]}")
                    print(
                        f"    {colorize('Difficulties:', Colors.DIM)} {', '.join(ds['difficulties'])}"
                    )
                else:
                    print(
                        f"  {colorize(ds['name'], Colors.YELLOW):15} {ds['num_examples']:4} examples  [{ds['category']}]"
                    )

            print(f"\n  {colorize(f'Total: {len(datasets)} datasets', Colors.DIM)}")
        except ImportError:
            print_warning("Benchmark datasets module not available")

    if args.type in ("trackers", "all"):
        print_subheader("Experiment Tracking Backends")
        trackers = [
            ("local", "Local file-based tracking (always available)"),
            ("wandb", "Weights & Biases (requires: pip install wandb)"),
            ("mlflow", "MLflow tracking (requires: pip install mlflow)"),
            ("tensorboard", "TensorBoard (requires: pip install tensorboard)"),
        ]
        for name, desc in trackers:
            print(f"  {colorize(name, Colors.MAGENTA):15} {desc}")

    return 0
