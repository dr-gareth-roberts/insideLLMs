"""Generate deterministic API and import-graph evidence without importing insideLLMs.

The architecture checks intentionally use only the Python standard library.  This
keeps them usable before optional dependencies are installed and prevents the
scanner itself from hiding import-time coupling.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "insideLLMs"
ARCHITECTURE_ROOT = REPO_ROOT / "architecture"
API_MANIFEST_PATH = ARCHITECTURE_ROOT / "api_manifest.json"
IMPORT_GRAPH_PATH = ARCHITECTURE_ROOT / "import_graph.json"
LAYER_CONFIG_PATH = ARCHITECTURE_ROOT / "layers.json"
STATUS_POLICY_PATH = ARCHITECTURE_ROOT / "api_status_policy.json"
API_STATUS_DOC_PATH = REPO_ROOT / "docs" / "API_STATUS.md"

PUBLIC_STATUSES = {"stable", "experimental", "deprecated", "internal"}
TEXT_EVIDENCE_SUFFIXES = {".md", ".rst", ".sh", ".toml", ".yaml", ".yml"}
EVIDENCE_AREAS: tuple[tuple[str, str], ...] = (
    ("docs", "README.md"),
    ("docs", "API_REFERENCE.md"),
    ("docs", "ARCHITECTURE.md"),
    ("docs", "DOCUMENTATION_INDEX.md"),
    ("docs", "QUICK_REFERENCE.md"),
    ("examples", "examples"),
    ("docs", "docs"),
    ("wiki", "wiki"),
    ("ci", "ci"),
    ("github_action", "action.yml"),
    ("scripts", "scripts"),
    ("benchmarks", "benchmarks"),
    ("compliance_intelligence", "compliance_intelligence"),
    ("cli", "insideLLMs/cli/commands"),
    ("tests", "tests"),
)

OPTIONAL_DEPENDENCY_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "cohere": ("cohere",),
    "crypto": ("cryptography",),
    "gemini": ("google.generativeai",),
    "huggingface": ("huggingface_hub", "transformers"),
    "langchain": ("langchain", "langchain_core", "langgraph"),
    "nlp": ("gensim", "nltk", "sklearn", "spacy"),
    "openai": ("openai",),
    "serving": ("fastapi", "uvicorn"),
    "signing": ("oras", "tuf"),
    "visualization": ("matplotlib", "pandas", "PIL", "seaborn"),
}


@dataclass(frozen=True, order=True)
class ImportReference:
    """One statically observed import."""

    imported: str
    line: int


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def module_name_for_path(
    path: Path,
    *,
    package_root: Path = PACKAGE_ROOT,
    package_name: str = "insideLLMs",
) -> str:
    """Return the importable module name represented by a package source path."""

    relative = path.relative_to(package_root).with_suffix("")
    parts = [package_name, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def containing_package_for_path(
    path: Path,
    *,
    package_root: Path = PACKAGE_ROOT,
    package_name: str = "insideLLMs",
) -> tuple[str, ...]:
    """Return package parts used to resolve relative imports in ``path``."""

    module_parts = module_name_for_path(
        path, package_root=package_root, package_name=package_name
    ).split(".")
    if path.name == "__init__.py":
        return tuple(module_parts)
    return tuple(module_parts[:-1])


def scan_imports(
    path: Path,
    *,
    containing_package: Sequence[str] | None = None,
    package_root: Path = PACKAGE_ROOT,
    package_name: str = "insideLLMs",
) -> tuple[ImportReference, ...]:
    """Parse imports from ``path`` and resolve relative names.

    ``containing_package`` is primarily useful for unit tests with temporary
    files. Real package scans derive it from ``package_root``.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = (
        tuple(containing_package)
        if containing_package is not None
        else containing_package_for_path(
            path,
            package_root=package_root,
            package_name=package_name,
        )
    )
    references: list[ImportReference] = []

    def add_dynamic_reference(value: object, line: int) -> None:
        if isinstance(value, str) and _looks_like_module_name(value):
            references.append(ImportReference(value, line))
        elif isinstance(value, Mapping):
            for item in value.values():
                add_dynamic_reference(item, line)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                add_dynamic_reference(item, line)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend(ImportReference(alias.name, node.lineno) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            levels_up = node.level - 1
            if levels_up > len(package):
                base_parts: tuple[str, ...] = ()
            elif levels_up:
                base_parts = package[:-levels_up]
            else:
                base_parts = package
            if node.module:
                base_parts = (*base_parts, *node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""

        for alias in node.names:
            imported = ".".join(part for part in (base, alias.name) if part)
            references.append(ImportReference(imported, node.lineno))

    # Literal dynamic imports are architecture dependencies too. Capture direct
    # import helpers and the literal maps used by lazy public facades without
    # evaluating package code.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and target.id.upper().endswith("_IMPORTS")
                for target in targets
            ):
                continue
            try:
                value = ast.literal_eval(node.value) if node.value is not None else None
            except (ValueError, TypeError):
                continue
            add_dynamic_reference(value, node.lineno)
            continue

        if not isinstance(node, ast.Call):
            continue
        callable_name = _callable_name(node.func)
        if "import" not in callable_name.lower():
            continue
        for argument in node.args:
            try:
                value = ast.literal_eval(argument)
            except (ValueError, TypeError):
                continue
            add_dynamic_reference(value, node.lineno)

    return tuple(sorted(set(references)))


_MODULE_NAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


def _looks_like_module_name(value: str) -> bool:
    if _MODULE_NAME_RE.fullmatch(value):
        return True
    return any(
        value == prefix for prefixes in OPTIONAL_DEPENDENCY_PREFIXES.values() for prefix in prefixes
    )


def _callable_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _callable_name(node.value)
        return ".".join(part for part in (parent, node.attr) if part)
    return ""


def python_modules(root: Path = PACKAGE_ROOT) -> tuple[Path, ...]:
    """Return package Python files in deterministic order."""

    return tuple(sorted(root.rglob("*.py")))


def _resolve_known_module(imported: str, known_modules: set[str]) -> str | None:
    candidate = imported
    while candidate:
        if candidate in known_modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def _layer_for_module(module: str, config: Mapping[str, Any]) -> str:
    for layer in config["layers"]:
        for prefix in layer["prefixes"]:
            if matches_prefix(module, prefix):
                return str(layer["name"])
    return str(config["default_layer"])


def _optional_extra(imported: str) -> str | None:
    for extra, prefixes in OPTIONAL_DEPENDENCY_PREFIXES.items():
        if any(matches_prefix(imported, prefix) for prefix in prefixes):
            return extra
    return None


def build_import_graph(
    *,
    package_root: Path = PACKAGE_ROOT,
    layer_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build module metrics, internal edges, and optional-dependency evidence."""

    config = dict(layer_config or _read_json(LAYER_CONFIG_PATH))
    paths = python_modules(package_root)
    path_by_module = {module_name_for_path(path, package_root=package_root): path for path in paths}
    known_modules = set(path_by_module)
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    internal_edges: set[tuple[str, str]] = set()
    optional_imports: list[dict[str, object]] = []

    for source, path in sorted(path_by_module.items()):
        for reference in scan_imports(path, package_root=package_root):
            target = _resolve_known_module(reference.imported, known_modules)
            if target is not None and target != source:
                outgoing[source].add(target)
                incoming[target].add(source)
                internal_edges.add((source, target))
            extra = _optional_extra(reference.imported)
            if extra is not None:
                optional_imports.append(
                    {
                        "extra": extra,
                        "imported": reference.imported,
                        "source": source,
                    }
                )

    modules: list[dict[str, object]] = []
    for module, path in sorted(path_by_module.items()):
        text = path.read_text(encoding="utf-8")
        modules.append(
            {
                "fan_in": len(incoming[module]),
                "fan_out": len(outgoing[module]),
                "layer": _layer_for_module(module, config),
                "loc": len(text.splitlines()),
                "module": module,
                "path": path.relative_to(REPO_ROOT).as_posix(),
            }
        )

    edges = [
        {
            "source": source,
            "source_layer": _layer_for_module(source, config),
            "target": target,
            "target_layer": _layer_for_module(target, config),
        }
        for source, target in sorted(internal_edges)
    ]
    layer_edges = Counter((edge["source_layer"], edge["target_layer"]) for edge in edges)

    test_references: Counter[str] = Counter()
    tests_root = REPO_ROOT / "tests"
    if tests_root.exists():
        for path in sorted(tests_root.rglob("*.py")):
            for reference in scan_imports(
                path,
                containing_package=(),
                package_root=tests_root,
                package_name="tests",
            ):
                target = _resolve_known_module(reference.imported, known_modules)
                if target is not None:
                    subtree = ".".join(target.split(".")[:2])
                    test_references[subtree] += 1

    return {
        "schema_version": 1,
        "summary": {
            "internal_edge_count": len(edges),
            "module_count": len(modules),
            "optional_import_count": len(optional_imports),
            "package_loc": sum(int(module["loc"]) for module in modules),
            "static_test_references_by_subtree": dict(sorted(test_references.items())),
        },
        "layer_edges": [
            {"count": count, "source_layer": source, "target_layer": target}
            for (source, target), count in sorted(layer_edges.items())
        ],
        "modules": modules,
        "edges": edges,
        "optional_imports": sorted(
            optional_imports,
            key=lambda item: (str(item["source"]), str(item["imported"])),
        ),
    }


def _literal_assignment(tree: ast.AST, name: str) -> object:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value
            if value is None:
                break
            return ast.literal_eval(value)
    raise ValueError(f"could not find literal assignment for {name}")


def _direct_export_modules(tree: ast.Module) -> dict[str, str]:
    modules: dict[str, str] = {"__version__": "insideLLMs"}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = "insideLLMs"
            if node.module:
                base = f"{base}.{node.module}"
        else:
            base = node.module or "insideLLMs"
        for alias in node.names:
            modules[alias.asname or alias.name] = base
    return modules


def _iter_area_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path != API_STATUS_DOC_PATH:
            yield path
        return
    if path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if (
                candidate != API_STATUS_DOC_PATH
                and candidate.is_file()
                and (candidate.suffix == ".py" or candidate.suffix in TEXT_EVIDENCE_SUFFIXES)
            ):
                yield candidate


def _python_root_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    package_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "insideLLMs":
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "insideLLMs":
                    package_aliases.add(alias.asname or "insideLLMs")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in package_aliases
        ):
            names.add(node.attr)
    return names


_FROM_ROOT_RE = re.compile(
    r"from\s+insideLLMs\s+import\s*(?:\((?P<block>.*?)\)|(?P<line>[^\n]+))",
    re.DOTALL,
)
_ROOT_ATTRIBUTE_RE = re.compile(r"\binsideLLMs\.([A-Za-z_]\w*)")


def _text_root_imports(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return set()
    names = set(_ROOT_ATTRIBUTE_RE.findall(text))
    for match in _FROM_ROOT_RE.finditer(text):
        imported = match.group("block") or match.group("line") or ""
        imported = imported.split("#", 1)[0]
        for name in re.findall(r"\b[A-Za-z_]\w*\b", imported):
            if name not in {"as", "import"}:
                names.add(name)
    return names


def collect_api_usage() -> dict[str, dict[str, object]]:
    """Collect root-symbol usage from repository-owned consumer surfaces."""

    paths_by_name: dict[str, set[str]] = defaultdict(set)
    areas_by_name: dict[str, Counter[str]] = defaultdict(Counter)
    for area, relative in EVIDENCE_AREAS:
        for path in _iter_area_files(REPO_ROOT / relative):
            names = _python_root_imports(path) if path.suffix == ".py" else _text_root_imports(path)
            relative_path = path.relative_to(REPO_ROOT).as_posix()
            for name in names:
                paths_by_name[name].add(relative_path)
                areas_by_name[name][area] += 1
    return {
        name: {
            "areas": dict(sorted(areas_by_name[name].items())),
            "path_count": len(paths),
            "paths": sorted(paths),
        }
        for name, paths in sorted(paths_by_name.items())
    }


def build_api_manifest() -> dict[str, Any]:
    """Build the root public-facade manifest from AST and usage evidence."""

    init_path = PACKAGE_ROOT / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    direct_names = _literal_assignment(tree, "__all__")
    lazy_imports = _literal_assignment(tree, "_LAZY_IMPORTS")
    if not isinstance(direct_names, list) or not all(
        isinstance(name, str) for name in direct_names
    ):
        raise ValueError("insideLLMs.__all__ must be a literal list of strings")
    if not isinstance(lazy_imports, dict) or not all(
        isinstance(name, str) and isinstance(module, str) for name, module in lazy_imports.items()
    ):
        raise ValueError("insideLLMs._LAZY_IMPORTS must be a literal string map")

    policy = _read_json(STATUS_POLICY_PATH)
    default_status = str(policy["default_status"])
    overrides = policy.get("overrides", {})
    if default_status not in PUBLIC_STATUSES:
        raise ValueError(f"invalid default API status: {default_status}")
    if not isinstance(overrides, dict):
        raise ValueError("api_status_policy.overrides must be an object")

    usage = collect_api_usage()
    direct_modules = _direct_export_modules(tree)
    symbols = []
    for name in sorted(set(direct_names) | set(lazy_imports)):
        kinds = []
        if name in direct_names:
            kinds.append("direct")
        if name in lazy_imports:
            kinds.append("lazy")
        status = str(overrides.get(name, default_status))
        if status not in PUBLIC_STATUSES:
            raise ValueError(f"invalid API status for {name}: {status}")
        modules = set()
        if name in direct_names:
            modules.add(direct_modules.get(name, "insideLLMs"))
        if name in lazy_imports:
            modules.add(str(lazy_imports[name]))
        symbols.append(
            {
                "evidence": usage.get(name, {"areas": {}, "path_count": 0, "paths": []}),
                "export_kinds": kinds,
                "name": name,
                "providers": sorted(modules),
                "status": status,
            }
        )

    status_counts = Counter(symbol["status"] for symbol in symbols)
    return {
        "schema_version": 1,
        "surface": "insideLLMs root package",
        "policy": {
            "allowed_statuses": sorted(PUBLIC_STATUSES),
            "source": "architecture/api_status_policy.json",
        },
        "summary": {
            "direct_export_count": len(direct_names),
            "lazy_export_count": len(lazy_imports),
            "status_counts": dict(sorted(status_counts.items())),
            "symbol_count": len(symbols),
            "symbols_with_repository_usage": sum(
                int(symbol["evidence"]["path_count"] > 0) for symbol in symbols
            ),
        },
        "symbols": symbols,
    }


def render_api_status_doc(manifest: Mapping[str, Any]) -> str:
    """Render the human-readable companion to the API manifest."""

    summary = manifest["summary"]
    lines = [
        "# Root API status",
        "",
        "<!-- Generated by scripts/architecture_evidence.py; do not edit by hand. -->",
        "",
        "This inventory describes the **actual** `insideLLMs` root facade. It is generated",
        "statically, so optional providers are never imported while collecting evidence.",
        "Compatibility policy remains defined by `docs/STABILITY.md` and",
        "`docs/STABILITY_MATRIX.md`.",
        "",
        "## Summary",
        "",
        f"- Root symbols: **{summary['symbol_count']}**",
        f"- Direct `__all__` exports: **{summary['direct_export_count']}**",
        f"- Lazy aliases: **{summary['lazy_export_count']}**",
        f"- Symbols referenced by repository-owned consumers: **{summary['symbols_with_repository_usage']}**",
        "",
        "Repository usage is evidence, not an automatic stability promise. The checked-in policy",
        "file assigns every symbol one of `stable`, `experimental`, `deprecated`, or `internal`.",
        "",
        "## Symbols",
        "",
        "| Symbol | Status | Export | Provider | Evidence paths |",
        "|---|---|---|---|---:|",
    ]
    for symbol in manifest["symbols"]:
        lines.append(
            "| `{name}` | {status} | {kinds} | {providers} | {count} |".format(
                name=symbol["name"],
                status=symbol["status"],
                kinds=", ".join(symbol["export_kinds"]),
                providers="<br>".join(f"`{value}`" for value in symbol["providers"]),
                count=symbol["evidence"]["path_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Updating the inventory",
            "",
            "Run `make architecture-update`, review both JSON diffs and this table, then run",
            "`make architecture`. New aliases enter as experimental by default; promotion or",
            "deprecation requires an explicit policy override.",
            "",
        ]
    )
    return "\n".join(lines)


def generated_outputs() -> dict[Path, str]:
    api_manifest = build_api_manifest()
    return {
        API_MANIFEST_PATH: _json_text(api_manifest),
        IMPORT_GRAPH_PATH: _json_text(build_import_graph()),
        API_STATUS_DOC_PATH: render_api_status_doc(api_manifest),
    }


def write_outputs(outputs: Mapping[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")


def check_outputs(outputs: Mapping[Path, str]) -> int:
    stale: list[str] = []
    for path, expected in outputs.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            stale.append(path.relative_to(REPO_ROOT).as_posix())
    if stale:
        print("Architecture evidence is stale:", file=sys.stderr)
        for path in stale:
            print(f"  - {path}", file=sys.stderr)
        print("Run `make architecture-update` and review the generated diff.", file=sys.stderr)
        return 1
    print("Architecture evidence is up to date.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail when checked-in output is stale")
    mode.add_argument("--write", action="store_true", help="write checked-in evidence (default)")
    args = parser.parse_args(argv)
    outputs = generated_outputs()
    if args.check:
        return check_outputs(outputs)
    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
