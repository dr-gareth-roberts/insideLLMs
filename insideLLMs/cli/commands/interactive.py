"""Interactive command: start an interactive exploration session."""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

from insideLLMs.registry import ensure_builtins_registered, model_registry, probe_registry

from .._output import (
    Colors,
    Spinner,
    colorize,
    print_error,
    print_header,
    print_info,
    print_key_value,
    print_subheader,
    print_success,
    print_warning,
)


def cmd_interactive(args: argparse.Namespace) -> int:
    """Execute the interactive command."""
    ensure_builtins_registered()

    print_header("Interactive Mode")
    print_info(f"Model: {args.model}")
    print_info("Type 'help' for commands, 'quit' to exit")

    try:
        model_or_factory = model_registry.get(args.model)
        model = model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
    except Exception as e:
        print_error(f"Could not load model: {e}")
        return 1

    # Command history
    history: list[str] = []

    # Track the last exchange for probe evaluation
    last_prompt: Optional[str] = None
    last_response: Optional[str] = None

    # Load history file
    history_path = Path(args.history_file)
    if history_path.exists():
        with open(history_path) as f:
            history = [line.strip() for line in f.readlines()]

    print()

    while True:
        try:
            prompt = input(colorize(">>> ", Colors.BRIGHT_CYAN))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        # Save to history
        history.append(prompt)
        with open(history_path, "a") as f:
            f.write(prompt + "\n")

        # Handle commands
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif prompt.lower() == "help":
            print(f"""
{colorize("Available Commands:", Colors.BOLD)}
  help          - Show this help message
  quit/exit/q   - Exit interactive mode
  history       - Show command history
  clear         - Clear the screen
  model <name>  - Switch to a different model
  probe <name>  - Run a probe on the last response

{colorize("Usage:", Colors.BOLD)}
  Just type your prompt and press Enter to get a response.
""")
        elif prompt.lower() == "history":
            print_subheader("Command History")
            for i, h in enumerate(history[-20:], 1):
                print(f"  {i:3}. {h[:60]}")
        elif prompt.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
        elif prompt.lower().startswith("model "):
            new_model = prompt[6:].strip()
            try:
                model_or_factory = model_registry.get(new_model)
                model = (
                    model_or_factory() if isinstance(model_or_factory, type) else model_or_factory
                )
                print_success(f"Switched to model: {new_model}")
            except Exception as e:
                print_error(f"Could not load model: {e}")
        elif prompt.lower().startswith("probe "):
            probe_name = prompt[6:].strip()
            if last_prompt is None or last_response is None:
                print_warning("No previous response to evaluate. Send a prompt first.")
                continue
            try:
                probe_factory = probe_registry.get(probe_name)
                probe = probe_factory() if isinstance(probe_factory, type) else probe_factory
                print_subheader(f"Probe: {probe.name}")
                result = probe.run(model, last_prompt)
                if isinstance(result, dict):
                    for k, v in result.items():
                        print_key_value(str(k), str(v))
                else:
                    print(f"  {result}")
            except KeyError:
                available = ", ".join(probe_registry.list())
                print_error(f"Unknown probe: {probe_name}")
                print_info(f"Available probes: {available}")
            except Exception as e:
                print_error(f"Probe error: {e}")
        else:
            # Regular prompt - generate response
            spinner = Spinner("Thinking")
            start = time.time()
            spinner.spin()

            try:
                response = model.generate(prompt)
                elapsed = time.time() - start
                spinner.stop(success=True)

                last_prompt = prompt
                last_response = response

                print(f"\n{response}\n")
                print(colorize(f"[{elapsed * 1000:.0f}ms]", Colors.DIM))
                print()
            except Exception as e:
                spinner.stop(success=False)
                print_error(f"Error: {e}")

    return 0
