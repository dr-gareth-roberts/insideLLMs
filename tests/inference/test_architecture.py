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
    "transformers",
    "huggingface_hub",
)


def _module_package(path: Path, inference_root: Path) -> list[str]:
    relative = path.relative_to(inference_root)
    return ["insideLLMs", "inference", *relative.parent.parts]


def _imports(path: Path, *, inference_root: Path | None = None) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = (
                    _module_package(path, inference_root)
                    if inference_root is not None
                    else ["insideLLMs", "inference"]
                )
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


def test_import_scanner_resolves_relative_imports_from_nested_packages(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    module = nested / "module.py"
    module.write_text("from ... import models\n", encoding="utf-8")

    assert _imports(module, inference_root=tmp_path) == {"insideLLMs.models"}


def _matches_prefix(imported: str, prefix: str) -> bool:
    return imported == prefix or imported.startswith(f"{prefix}.")


def _python_modules(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def test_inference_module_discovery_is_recursive(tmp_path: Path) -> None:
    (tmp_path / "top.py").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.py").write_text("", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in _python_modules(tmp_path)] == [
        "nested/child.py",
        "top.py",
    ]


def test_forbidden_prefix_matching_respects_module_boundaries() -> None:
    assert _matches_prefix("insideLLMs.models.openai", "insideLLMs.models")
    assert not _matches_prefix("insideLLMs.models_extra", "insideLLMs.models")


def test_huggingface_provider_sdks_are_forbidden() -> None:
    assert any(_matches_prefix("transformers", prefix) for prefix in FORBIDDEN_PREFIXES)
    assert any(_matches_prefix("huggingface_hub", prefix) for prefix in FORBIDDEN_PREFIXES)


def test_inference_core_does_not_depend_on_contrib_or_provider_sdks() -> None:
    violations: list[str] = []
    for path in _python_modules(INFERENCE_ROOT):
        for imported in sorted(_imports(path, inference_root=INFERENCE_ROOT)):
            if any(_matches_prefix(imported, prefix) for prefix in FORBIDDEN_PREFIXES):
                module_path = path.relative_to(INFERENCE_ROOT).as_posix()
                violations.append(f"{module_path}: {imported}")

    assert violations == []


def test_runtime_config_dependency_is_isolated_to_client_adapter() -> None:
    violations: list[str] = []
    for path in _python_modules(INFERENCE_ROOT):
        runtime_imports = {
            imported
            for imported in _imports(path, inference_root=INFERENCE_ROOT)
            if _matches_prefix(imported, "insideLLMs.runtime")
        }
        if runtime_imports and path != INFERENCE_ROOT / "client.py":
            module_path = path.relative_to(INFERENCE_ROOT).as_posix()
            violations.extend(f"{module_path}: {imported}" for imported in sorted(runtime_imports))

    assert violations == []
