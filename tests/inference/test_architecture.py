import ast
from pathlib import Path

INFERENCE_ROOT = Path(__file__).parents[2] / "insideLLMs" / "inference"
FORBIDDEN_PREFIXES = (
    "insideLLMs.contrib",
    "insideLLMs.models",
    "openai",
    "anthropic",
    "cohere",
    "google.generativeai",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = ["insideLLMs", "inference"]
                parent = package[: len(package) - (node.level - 1)]
                base = ".".join([*parent, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            for alias in node.names:
                imported.add(".".join(part for part in (base, alias.name) if part))
    return imported


def test_import_scanner_resolves_relative_package_imports(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "from ..contrib import retrieval\nfrom .. import models\nfrom insideLLMs import contrib\n",
        encoding="utf-8",
    )

    assert _imports(module) == {
        "insideLLMs.contrib",
        "insideLLMs.contrib.retrieval",
        "insideLLMs.models",
    }


def _matches_prefix(imported: str, prefix: str) -> bool:
    return imported == prefix or imported.startswith(f"{prefix}.")


def test_forbidden_prefix_matching_respects_module_boundaries() -> None:
    assert _matches_prefix("insideLLMs.models.openai", "insideLLMs.models")
    assert not _matches_prefix("insideLLMs.models_extra", "insideLLMs.models")


def test_inference_core_does_not_depend_on_contrib_or_provider_sdks() -> None:
    violations: list[str] = []
    for path in sorted(INFERENCE_ROOT.glob("*.py")):
        for imported in sorted(_imports(path)):
            if any(_matches_prefix(imported, prefix) for prefix in FORBIDDEN_PREFIXES):
                violations.append(f"{path.name}: {imported}")

    assert violations == []


def test_runtime_config_dependency_is_isolated_to_client_adapter() -> None:
    violations: list[str] = []
    for path in sorted(INFERENCE_ROOT.glob("*.py")):
        runtime_imports = {
            imported
            for imported in _imports(path)
            if _matches_prefix(imported, "insideLLMs.runtime")
        }
        if runtime_imports and path.name != "client.py":
            violations.extend(f"{path.name}: {imported}" for imported in sorted(runtime_imports))

    assert violations == []
