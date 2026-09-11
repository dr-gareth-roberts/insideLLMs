"""Produce two run directories whose agent answers identically by different routes.

    python run_demo.py

Then compare them:

    insidellms diff v1 v2 --fail-on-changes \\
        --output-fingerprint-ignore trace_events,trace_fingerprint,tool_calls   # exit 0
    insidellms diff v1 v2 --fail-on-trajectory-drift                            # exit 5

The first says the user-visible answer did not change, which is true. The second
says the agent took a different path to produce it, which is also true, and is
the regression an output-level evaluation cannot see.
"""

from support_agent import SupportAgent

from insideLLMs.models import DummyModel
from insideLLMs.runtime.runner import ProbeRunner

PROMPTS = [
    {"prompt": "Is my account active?"},
    {"prompt": "Can you check my status?"},
]


def main() -> None:
    for version in (1, 2):
        runner = ProbeRunner(DummyModel(), SupportAgent(version=version))
        runner.run(
            PROMPTS,
            emit_run_artifacts=True,
            run_dir=f"v{version}",
            run_id=f"agent-v{version}",
            overwrite=True,
            return_experiment=False,
            deterministic_artifacts=True,
        )
        print(f"wrote v{version}/records.jsonl")


if __name__ == "__main__":
    main()
