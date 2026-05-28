"""Quicktest command: quickly test a prompt against a model."""

import argparse
import json
import time

from insideLLMs.registry import ensure_builtins_registered, probe_registry

from .._output import (
    Colors,
    Spinner,
    colorize,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
)
from ._run_common import resolve_registered_model


def cmd_quicktest(args: argparse.Namespace) -> int:
    """Execute the quicktest command."""
    ensure_builtins_registered()

    print_header("Quick Test")
    print_key_value("Model", args.model)
    print_key_value("Prompt", args.prompt[:50] + "..." if len(args.prompt) > 50 else args.prompt)

    try:
        model_args = json.loads(args.model_args)
        init_kwargs = dict(model_args)
        init_kwargs.pop("temperature", None)
        init_kwargs.pop("max_tokens", None)

        model = resolve_registered_model(args.model, **init_kwargs)

        spinner = Spinner("Generating response")
        start_time = time.time()
        spinner.spin()

        response = model.generate(
            args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        elapsed = time.time() - start_time
        spinner.stop(success=True)

        print_subheader("Response")
        print(f"  {response}")

        print_subheader("Stats")
        print_key_value("Latency", f"{elapsed * 1000:.1f}ms")
        print_key_value("Response length", f"{len(response)} characters")

        if args.probe:
            print_subheader(f"Probe: {args.probe}")
            probe = probe_registry.get(args.probe)
            probe_result = probe.run(model, args.prompt)
            if isinstance(probe_result, dict):
                for key, value in probe_result.items():
                    print_key_value(str(key), str(value))
            else:
                print_key_value("Result", str(probe_result))
            print_info(
                f"Probe '{args.probe}' applied (detailed scoring available in full experiments)"
            )

        print()
        print_info("Next steps:")
        cmd_init = colorize("insidellms init --template basic", Colors.GREEN)
        cmd_compare = colorize('insidellms compare --models m1,m2 --input "prompt"', Colors.GREEN)
        cmd_list = colorize("insidellms list probes", Colors.GREEN)
        print(f"  Create a config:   {cmd_init}")
        print(f"  Compare models:    {cmd_compare}")
        print(f"  See all probes:    {cmd_list}")

        return 0

    except KeyError as e:
        print_error(f"Unknown model or probe: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        return 1
