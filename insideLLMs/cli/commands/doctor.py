"""Doctor command: diagnose environment and optional dependencies."""

import argparse
import json
import os
import platform
import sys
from typing import Any, Optional

from .._output import (
    print_header,
    print_key_value,
    print_subheader,
    print_success,
    print_warning,
)
from .._parsing import _check_nltk_resource, _has_module, _module_version
from .._record_utils import _json_default


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose environment and optional dependencies."""
    checks: list[dict[str, Any]] = []

    def add_check(*, name: str, ok: bool, hint: Optional[str] = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "hint": hint})

    add_check(name="python", ok=True, hint=sys.version.split()[0])
    add_check(name="platform", ok=True, hint=platform.platform())
    add_check(name="insideLLMs", ok=True, hint=_module_version("insideLLMs"))

    # Optional validation/schema tooling
    add_check(name="pydantic", ok=_has_module("pydantic"), hint='pip install ".[dev]"')

    # NLP extras
    add_check(name="nltk", ok=_has_module("nltk"), hint='pip install ".[nlp]"')
    add_check(name="sklearn", ok=_has_module("sklearn"), hint='pip install ".[nlp]"')
    add_check(name="spacy", ok=_has_module("spacy"), hint='pip install ".[nlp]"')
    add_check(name="gensim", ok=_has_module("gensim"), hint='pip install ".[nlp]"')
    add_check(
        name="nltk:punkt",
        ok=_check_nltk_resource("tokenizers/punkt"),
        hint="python -m nltk.downloader punkt",
    )
    add_check(
        name="nltk:vader_lexicon",
        ok=_check_nltk_resource("sentiment/vader_lexicon.zip")
        or _check_nltk_resource("sentiment/vader_lexicon"),
        hint="python -m nltk.downloader vader_lexicon",
    )
    add_check(
        name="spacy:en_core_web_sm",
        ok=_has_module("en_core_web_sm"),
        hint="python -m spacy download en_core_web_sm",
    )

    # Visualization extras
    add_check(
        name="matplotlib", ok=_has_module("matplotlib"), hint='pip install ".[visualization]"'
    )
    add_check(name="pandas", ok=_has_module("pandas"), hint='pip install ".[visualization]"')
    add_check(name="seaborn", ok=_has_module("seaborn"), hint='pip install ".[visualization]"')
    add_check(name="plotly", ok=_has_module("plotly"), hint='pip install ".[visualization]"')
    add_check(name="ipywidgets", ok=_has_module("ipywidgets"), hint="pip install ipywidgets")

    # Optional integrations
    add_check(name="redis", ok=_has_module("redis"), hint="pip install redis")
    add_check(name="datasets", ok=_has_module("datasets"), hint="pip install datasets")

    # API keys (informational)
    add_check(name="OPENAI_API_KEY", ok=bool(os.environ.get("OPENAI_API_KEY")), hint="set env var")
    add_check(
        name="ANTHROPIC_API_KEY", ok=bool(os.environ.get("ANTHROPIC_API_KEY")), hint="set env var"
    )

    run_root = os.environ.get("INSIDELLMS_RUN_ROOT")
    add_check(
        name="INSIDELLMS_RUN_ROOT",
        ok=bool(run_root),
        hint="(optional) override run artifacts root",
    )

    warn_checks = [
        c
        for c in checks
        if c["name"]
        not in {
            "python",
            "platform",
            "insideLLMs",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "INSIDELLMS_RUN_ROOT",
        }
        and not c["ok"]
    ]

    if args.format == "json":
        payload = {"checks": checks, "warnings": warn_checks}
        print(json.dumps(payload, indent=2, default=_json_default))
        return 1 if (args.fail_on_warn and warn_checks) else 0

    print_header("insideLLMs Doctor")
    print_subheader("Environment")
    for item in checks[:3]:
        print_key_value(item["name"], item["hint"] or "-")

    print_subheader("Diagnostics")
    for item in checks[3:]:
        if item["ok"]:
            print_success(item["name"])
        else:
            hint = f" ({item['hint']})" if item.get("hint") else ""
            print_warning(f"{item['name']}{hint}")

    if warn_checks:
        print()
        print_warning(f"{len(warn_checks)} recommended checks failed (optional deps missing).")
    else:
        print()
        print_success("All recommended checks passed.")

    return 1 if (args.fail_on_warn and warn_checks) else 0
