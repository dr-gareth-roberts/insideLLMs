"""Welcome command: friendly introduction for new users."""

import argparse

from .._output import (
    Colors,
    colorize,
    print_header,
    print_subheader,
    print_success,
)


def cmd_welcome(args: argparse.Namespace) -> int:
    """Execute the welcome command."""
    print_header("Welcome to insideLLMs!")

    print("  insideLLMs is a world-class toolkit for probing, evaluating,")
    print("  and testing Large Language Models with a focus on determinism")
    print("  and behavioural regression detection.")
    print()

    print_subheader("🚀 Quick Start")
    print("  The easiest way to see insideLLMs in action is to run a quick test:")
    print(f"  {colorize('insidellms quicktest Hello world --model dummy', Colors.GREEN)}")
    print()

    print_subheader("🛠️ Setting up your first experiment")
    print("  To run a full evaluation suite, you'll need a configuration file.")
    print("  You can generate a sample one using the init command:")
    print(f"  {colorize('insidellms init', Colors.GREEN)}")
    print()
    print("  Then run it:")
    print(f"  {colorize('insidellms run experiment.yaml', Colors.GREEN)}")
    print()

    print_subheader("🩺 Health Check")
    print("  Check if your environment and API keys are ready:")
    print(f"  {colorize('insidellms doctor', Colors.GREEN)}")
    print()

    print_subheader("📚 Documentation & Examples")
    print("  - README: https://github.com/dr-gareth-roberts/insideLLMs")
    print("  - Examples: Look into the 'examples/' directory in the repository")
    print()

    print_success("You're all set! Happy probing.")
    print()

    return 0
