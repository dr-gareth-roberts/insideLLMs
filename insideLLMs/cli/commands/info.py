"""Info command: show detailed information about resources."""

import argparse
import json

from insideLLMs.registry import ensure_builtins_registered, model_registry, probe_registry

from .._output import Colors, colorize, print_error, print_header, print_key_value, print_subheader


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    ensure_builtins_registered()

    try:
        if args.type == "model":
            info = model_registry.info(args.name)
        elif args.type == "probe":
            info = probe_registry.info(args.name)
        else:  # dataset
            from insideLLMs.benchmark_datasets import load_builtin_dataset

            ds = load_builtin_dataset(args.name)
            stats = ds.get_stats()
            print_header(f"Dataset: {args.name}")
            print_key_value("Description", ds.description)
            print_key_value(
                "Category", ds.category.value if hasattr(ds.category, "value") else ds.category
            )
            print_key_value("Total examples", stats.total_count)
            print_key_value(
                "Categories", ", ".join(stats.categories) if stats.categories else "N/A"
            )
            print_key_value(
                "Difficulties", ", ".join(stats.difficulties) if stats.difficulties else "N/A"
            )

            print_subheader("Sample Examples")
            for i, ex in enumerate(ds.sample(3, seed=42)):
                print(f"\n  {colorize(f'Example {i + 1}', Colors.BOLD)}")
                print(f"    {colorize('Input:', Colors.DIM)} {ex.input_text[:80]}...")
                if ex.expected_output:
                    print(f"    {colorize('Expected:', Colors.DIM)} {ex.expected_output[:50]}")
                print(f"    {colorize('Difficulty:', Colors.DIM)} {ex.difficulty}")

            return 0

        print_header(f"{args.type.capitalize()}: {args.name}")
        print_key_value("Factory", info["factory"])

        if info.get("default_kwargs"):
            print_key_value("Default args", json.dumps(info["default_kwargs"], indent=2))

        if info.get("doc"):
            print_subheader("Description")
            print(f"  {info['doc']}")

        return 0

    except KeyError:
        print_error(f"{args.type.capitalize()} '{args.name}' not found")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1
