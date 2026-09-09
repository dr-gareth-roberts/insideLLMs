from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from scripts.architecture_evidence import (
    IMPORT_GRAPH_PATH,
    LAYER_CONFIG_PATH,
    REPO_ROOT,
    matches_prefix,
)

EXCEPTIONS_PATH = REPO_ROOT / "architecture" / "import_exceptions.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _exception_key(exception: dict[str, Any]) -> tuple[str, str]:
    return str(exception["source"]), str(exception["target"])


def test_layer_dependencies_follow_the_global_matrix() -> None:
    config = _load(LAYER_CONFIG_PATH)
    graph = _load(IMPORT_GRAPH_PATH)
    exceptions = _load(EXCEPTIONS_PATH)["exceptions"]
    exception_keys = {_exception_key(exception) for exception in exceptions}
    violations: list[str] = []
    used_exceptions: set[tuple[str, str]] = set()

    for edge in graph["edges"]:
        allowed = config["allowed_dependencies"][edge["source_layer"]]
        if edge["target_layer"] in allowed:
            continue
        key = (edge["source"], edge["target"])
        if key in exception_keys:
            used_exceptions.add(key)
            continue
        violations.append(
            f"{edge['source']} ({edge['source_layer']}) -> "
            f"{edge['target']} ({edge['target_layer']})"
        )

    assert violations == [], "unapproved architecture edges:\n" + "\n".join(violations)
    assert used_exceptions == exception_keys, "remove stale architecture exceptions"


def test_architecture_exceptions_are_owned_narrow_and_unexpired() -> None:
    exceptions = _load(EXCEPTIONS_PATH)["exceptions"]
    keys: set[tuple[str, str]] = set()

    for exception in exceptions:
        assert set(exception) == {"expires", "owner", "reason", "source", "target"}
        key = _exception_key(exception)
        assert key not in keys, f"duplicate architecture exception: {key}"
        keys.add(key)
        assert matches_prefix(key[0], "insideLLMs")
        assert matches_prefix(key[1], "insideLLMs")
        assert str(exception["owner"]).strip()
        assert len(str(exception["reason"]).strip()) >= 20
        assert date.fromisoformat(str(exception["expires"])) >= date.today()


def test_provider_sdks_stay_in_provider_or_labs_layers() -> None:
    config = _load(LAYER_CONFIG_PATH)
    graph = _load(IMPORT_GRAPH_PATH)
    prefixes = tuple(config["provider_sdk_prefixes"])
    allowed_layers = set(config["provider_sdk_allowed_layers"])
    layer_by_module = {module["module"]: module["layer"] for module in graph["modules"]}
    violations = [
        f"{item['source']}: {item['imported']}"
        for item in graph["optional_imports"]
        if any(matches_prefix(item["imported"], prefix) for prefix in prefixes)
        and layer_by_module[item["source"]] not in allowed_layers
    ]

    assert violations == []
