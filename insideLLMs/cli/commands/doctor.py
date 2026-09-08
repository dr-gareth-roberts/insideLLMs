"""Doctor command: diagnose environment and optional dependencies."""

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import sys
import warnings
from dataclasses import asdict
from typing import Any, Optional

from insideLLMs.models.catalogue import PROVIDER_CATALOGUE, ProviderSpec
from insideLLMs.registry import (
    PLUGIN_ENTRYPOINT_GROUP,
    dataset_registry,
    ensure_builtins_registered,
    get_builtin_model_factory,
    model_registry,
    probe_registry,
)

from .._output import (
    print_header,
    print_info,
    print_key_value,
    print_subheader,
    print_success,
    print_warning,
)
from .._parsing import _check_nltk_resource, _has_module, _module_version
from .._record_utils import _json_default

_BUILTIN_PROBES = {
    "logic",
    "factuality",
    "bias",
    "attack",
    "prompt_injection",
    "jailbreak",
    "code_generation",
    "code_explanation",
    "code_debug",
    "instruction_following",
    "multi_step_task",
    "constraint_compliance",
}

_BUILTIN_DATASETS = {"csv", "jsonl", "hf"}


def _plugins_disabled_via_env() -> bool:
    raw = os.environ.get("INSIDELLMS_DISABLE_PLUGINS", "").strip().lower()
    return raw in {"1", "true", "yes"}


def _entrypoint_plugins(group: str) -> list[dict[str, str]]:
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            selected = list(eps.select(group=group))
        else:  # pragma: no cover (older Python)
            selected = list(eps.get(group, []))  # type: ignore[attr-defined]
    except Exception as exc:
        # Surface discovery failures (e.g. malformed package metadata) via the
        # log rather than silently reporting "no plugins" — doctor is a diagnostic.
        logging.getLogger(__name__).warning(
            "Plugin entry-point discovery failed for group %r: %s", group, exc
        )
        return []

    normalized = [
        {"name": str(getattr(ep, "name", "")), "value": str(getattr(ep, "value", ""))}
        for ep in selected
    ]
    return sorted(normalized, key=lambda item: (item["name"], item["value"]))


def _capability_status(
    *,
    modules: list[str],
    credential_env: list[str],
    notes: list[str],
    credential_alternatives: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    missing_modules = sorted(module for module in modules if not _has_module(module))
    missing_credentials = sorted(
        env_name for env_name in credential_env if not bool(os.environ.get(env_name))
    )
    missing_groups = [
        group
        for group in credential_alternatives
        if not any(bool(os.environ.get(name)) for name in group)
    ]
    missing_credentials.extend(" or ".join(group) for group in missing_groups)
    status = "ready"
    if missing_modules and missing_credentials:
        status = "missing_dependencies_and_credentials"
    elif missing_modules:
        status = "missing_dependencies"
    elif missing_credentials:
        status = "missing_credentials"
    elif notes:
        status = "requires_external_service"

    return {
        "status": status,
        "dependency_ready": not missing_modules,
        "credential_ready": not missing_credentials,
        "missing_dependencies": missing_modules,
        "missing_credentials": missing_credentials,
        "notes": notes,
    }


def _provider_capability(name: str, spec: ProviderSpec | None) -> dict[str, Any]:
    common = {"name": name, "live_verification": "not_checked", "budget_support": "unknown"}
    if spec is None:
        return {
            **common,
            "source": "plugin",
            "metadata_status": "unknown",
            "status": "unknown",
            "dependency_ready": None,
            "credential_ready": None,
            "missing_dependencies": None,
            "missing_credentials": None,
            "dependencies": None,
            "credential_alternatives": None,
            "optional_credentials": None,
            "external_requirements": None,
            "declared_capabilities": None,
            "prerequisites_ready": None,
            "notes": ["No static requirement metadata for this registration."],
        }
    resolved = _capability_status(
        modules=[dependency.module for dependency in spec.dependencies],
        credential_env=[],
        notes=[],
        credential_alternatives=spec.credential_alternatives,
    )
    return {
        **common,
        **resolved,
        "source": "builtin",
        "metadata_status": "declared",
        "budget_support": spec.budget_support,
        "prerequisites_ready": resolved["dependency_ready"] and resolved["credential_ready"],
        "dependencies": [
            {
                **asdict(dependency),
                "available": dependency.module not in resolved["missing_dependencies"],
            }
            for dependency in spec.dependencies
        ],
        "credential_alternatives": [list(group) for group in spec.credential_alternatives],
        "optional_credentials": list(spec.optional_credentials),
        "external_requirements": list(spec.external_requirements),
        "declared_capabilities": asdict(spec.capabilities),
        "notes": list(spec.external_requirements),
    }


def _build_capabilities(checks: list[dict[str, Any]]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ensure_builtins_registered(load_plugins=False)

    model_names = sorted(model_registry.list())
    probe_names = sorted(probe_registry.list())
    dataset_names = sorted(dataset_registry.list())

    model_capabilities: list[dict[str, Any]] = []
    for model_name in model_names:
        # Names alone are insufficient: plugins can replace builtin registrations.
        spec = PROVIDER_CATALOGUE.get(model_name)
        if spec is not None and (
            model_registry.get_factory(model_name) is not get_builtin_model_factory(model_name)
        ):
            spec = None
        model_capabilities.append(_provider_capability(model_name, spec))

    probe_capabilities = [
        {
            "name": probe_name,
            "source": "builtin" if probe_name in _BUILTIN_PROBES else "plugin",
            "status": "ready",
        }
        for probe_name in probe_names
    ]
    dataset_capabilities = [
        {
            "name": dataset_name,
            "source": "builtin" if dataset_name in _BUILTIN_DATASETS else "plugin",
            "status": "ready",
        }
        for dataset_name in dataset_names
    ]

    check_map = {item["name"]: item for item in checks}

    def _ok(name: str) -> bool:
        return bool(check_map.get(name, {}).get("ok"))

    extras = [
        {
            "name": "nlp",
            "ready": all(
                _ok(check_name)
                for check_name in (
                    "nltk",
                    "sklearn",
                    "spacy",
                    "gensim",
                    "nltk:punkt",
                    "nltk:vader_lexicon",
                    "spacy:en_core_web_sm",
                )
            ),
            "checks": [
                "nltk",
                "sklearn",
                "spacy",
                "gensim",
                "nltk:punkt",
                "nltk:vader_lexicon",
                "spacy:en_core_web_sm",
            ],
        },
        {
            "name": "visualization",
            "ready": all(
                _ok(check_name)
                for check_name in ("matplotlib", "pandas", "seaborn", "plotly", "ipywidgets")
            ),
            "checks": ["matplotlib", "pandas", "seaborn", "plotly", "ipywidgets"],
        },
        {
            "name": "verifiable_evaluation",
            "ready": all(
                _ok(check_name)
                for check_name in ("ultimate:tuf", "ultimate:cosign", "ultimate:oras")
            ),
            "checks": ["ultimate:tuf", "ultimate:cosign", "ultimate:oras"],
        },
    ]

    report_outputs = [
        {"name": "summary.json", "available": True, "requires": []},
        {"name": "report.html_basic", "available": True, "requires": []},
        {"name": "report.html_interactive", "available": _ok("plotly"), "requires": ["plotly"]},
        {"name": "schema_validate", "available": _ok("pydantic"), "requires": ["pydantic"]},
    ]

    plugin_model_names = [item["name"] for item in model_capabilities if item["source"] == "plugin"]
    plugin_probe_names = [name for name in probe_names if name not in _BUILTIN_PROBES]
    plugin_dataset_names = [name for name in dataset_names if name not in _BUILTIN_DATASETS]

    plugins_disabled = _plugins_disabled_via_env()
    plugin_entrypoints = _entrypoint_plugins(PLUGIN_ENTRYPOINT_GROUP)

    return {
        "models": model_capabilities,
        "probes": probe_capabilities,
        "datasets": dataset_capabilities,
        "extras": extras,
        "reports": report_outputs,
        "plugins": {
            "entry_point_group": PLUGIN_ENTRYPOINT_GROUP,
            "auto_loading_enabled": not plugins_disabled,
            "disabled_by_env": plugins_disabled,
            "discovered_entry_points": plugin_entrypoints,
            "registered_extensions": {
                "models": plugin_model_names,
                "probes": plugin_probe_names,
                "datasets": plugin_dataset_names,
            },
        },
    }


def _print_capabilities_summary(capabilities: dict[str, Any]) -> None:
    print_subheader("Capabilities")
    models = capabilities.get("models", [])
    ready_models = [m for m in models if m.get("status") == "ready"]
    blocked_models = [m for m in models if m.get("status") != "ready"]
    print_key_value("Models (prerequisites ready/total)", f"{len(ready_models)}/{len(models)}")
    print_info("Live provider verification: not checked.")
    for item in blocked_models:
        print_warning(f"model:{item.get('name')} ({item.get('status')})")

    probes = capabilities.get("probes", [])
    datasets = capabilities.get("datasets", [])
    print_key_value("Probes (total)", str(len(probes)))
    print_key_value("Datasets (total)", str(len(datasets)))

    extras = capabilities.get("extras", [])
    for extra in extras:
        if extra.get("ready"):
            print_success(f"extra:{extra.get('name')}")
        else:
            print_warning(f"extra:{extra.get('name')} (missing checks)")

    plugins = capabilities.get("plugins", {})
    discovered = plugins.get("discovered_entry_points", [])
    print_key_value("Plugins discovered", str(len(discovered)))
    if plugins.get("disabled_by_env"):
        print_warning("plugin auto-loading is disabled by INSIDELLMS_DISABLE_PLUGINS")


def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose environment and optional dependencies."""
    checks: list[dict[str, Any]] = []

    def add_check(*, name: str, ok: bool, hint: Optional[str] = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "hint": hint})

    add_check(name="python", ok=True, hint=sys.version.split()[0])
    add_check(name="platform", ok=True, hint=platform.platform())
    add_check(name="insideLLMs", ok=True, hint=_module_version("insideLLMs"))

    # Validation is part of the base package contract.
    add_check(name="pydantic", ok=_has_module("pydantic"), hint="reinstall insideLLMs")

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

    # Ultimate/verifiable-evaluation tooling readiness
    add_check(
        name="ultimate:tuf",
        ok=_has_module("tuf"),
        hint='pip install "tuf>=3.0.0" (note: real TUF dataset verification is '
        "not implemented yet; fetch_dataset refuses production use regardless)",
    )
    add_check(
        name="ultimate:cosign",
        ok=shutil.which("cosign") is not None,
        hint="install cosign: https://docs.sigstore.dev/cosign/system_config/installation/",
    )
    add_check(
        name="ultimate:oras",
        ok=shutil.which("oras") is not None,
        hint="install oras: https://oras.land/docs/installation",
    )

    # API keys (informational)
    add_check(
        name="OPENAI_API_KEY",
        ok=bool(os.environ.get("OPENAI_API_KEY")),
        hint="export OPENAI_API_KEY=sk-...",
    )
    add_check(
        name="ANTHROPIC_API_KEY",
        ok=bool(os.environ.get("ANTHROPIC_API_KEY")),
        hint="export ANTHROPIC_API_KEY=sk-ant-...",
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

    include_capabilities = bool(getattr(args, "capabilities", False))
    capabilities = _build_capabilities(checks) if include_capabilities else None

    if args.format == "json":
        payload = {"checks": checks, "warnings": warn_checks}
        if capabilities is not None:
            payload["capabilities"] = capabilities
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
            if item["name"] in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "INSIDELLMS_RUN_ROOT"}:
                print_info(f"{item['name']}{hint}")
            else:
                print_warning(f"{item['name']}{hint}")

    if capabilities is not None:
        print()
        _print_capabilities_summary(capabilities)

    if warn_checks:
        print()
        print_warning(f"{len(warn_checks)} recommended checks failed (optional deps missing).")
    else:
        print()
        print_success("All recommended checks passed.")

    return 1 if (args.fail_on_warn and warn_checks) else 0
