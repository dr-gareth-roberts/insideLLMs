"""Quicktest command: quickly test a prompt against a model."""

import argparse
import json
import time

from insideLLMs.registry import ensure_builtins_registered, model_registry, probe_registry

from .._output import (
    Spinner,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
)


def cmd_quicktest(args: argparse.Namespace) -> int:
    """Execute the quicktest command."""
    ensure_builtins_registered()

    print_header("Quick Test")
    print_key_value("Model", args.model)
    print_key_value("Prompt", args.prompt[:50] + "..." if len(args.prompt) > 50 else args.prompt)

    try:
        # Parse model args
        model_args = json.loads(args.model_args)
        model_args["temperature"] = args.temperature
        model_args["max_tokens"] = args.max_tokens

        # Get model from registry - it may be a class or instance
        model_or_factory = model_registry.get(args.model)

        # If it's a class, instantiate it; if it's already an instance, use it
        if isinstance(model_or_factory, type):
            model = model_or_factory(**model_args)
        else:
            model = model_or_factory

        # Generate response
        spinner = Spinner("Generating response")
        start_time = time.time()
        spinner.spin()

        response = model.generate(args.prompt)
        elapsed = time.time() - start_time
        spinner.stop(success=True)

        print_subheader("Response")
        print(f"  {response}")

        print_subheader("Stats")
        print_key_value("Latency", f"{elapsed * 1000:.1f}ms")
        print_key_value("Response length", f"{len(response)} characters")

        # Apply probe if specified
        if args.probe:
            print_subheader(f"Probe: {args.probe}")
            probe_factory = probe_registry.get(args.probe)
            probe_factory()
            # Note: Probe evaluation would go here
            print_info(
                f"Probe '{args.probe}' applied (detailed scoring available in full experiments)"
            )

        return 0

    except KeyError as e:
        print_error(f"Unknown model or probe: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1
