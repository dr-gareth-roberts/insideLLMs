#!/usr/bin/env python3
"""Smoke-test imports supplied by each declared optional dependency extra."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence

EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "openai": ("openai", "insideLLMs.models.openai"),
    "anthropic": ("anthropic", "insideLLMs.models.anthropic"),
    "huggingface": (
        "transformers",
        "huggingface_hub",
        "insideLLMs.models.huggingface",
    ),
    "signing": (
        "oras",
        "insideLLMs.datasets.tuf_client",
        "insideLLMs.publish.oras",
    ),
    "crypto": ("cryptography", "insideLLMs.privacy.encryption"),
    "nlp": (
        "nltk",
        "spacy",
        "sklearn",
        "gensim",
        "insideLLMs.nlp",
    ),
    "visualization": (
        "matplotlib",
        "PIL",
        "pandas",
        "seaborn",
        "insideLLMs.analysis.visualization",
    ),
    "langchain": (
        "langchain",
        "langchain_core",
        "langgraph",
        "insideLLMs.integrations.langchain",
    ),
    "serving": ("fastapi", "uvicorn", "insideLLMs.contrib.deployment"),
    "providers": (
        "openai",
        "anthropic",
        "transformers",
        "huggingface_hub",
        "insideLLMs.models.openai",
        "insideLLMs.models.anthropic",
        "insideLLMs.models.huggingface",
    ),
}


def imports_for(extra: str) -> tuple[str, ...]:
    """Return the deduplicated import list for an extra or for all extras."""
    if extra == "all":
        return tuple(
            dict.fromkeys(module for modules in EXTRA_IMPORTS.values() for module in modules)
        )
    try:
        return EXTRA_IMPORTS[extra]
    except KeyError as error:
        choices = ", ".join((*EXTRA_IMPORTS, "all"))
        raise ValueError(f"Unknown extra {extra!r}; expected one of: {choices}") from error


def check_imports(modules: Sequence[str]) -> list[str]:
    """Import modules and return human-readable failures."""
    failures: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            failures.append(f"{module}: {type(error).__name__}: {error}")
        else:
            print(f"OK {module}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional-import smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extra", choices=(*EXTRA_IMPORTS, "all"))
    args = parser.parse_args(argv)

    failures = check_imports(imports_for(args.extra))
    if failures:
        print("\nOptional import failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
