#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*]\(([^)]+)\)")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _iter_wiki_markdown_files(wiki_dir: Path) -> Iterable[Path]:
    return sorted(p for p in wiki_dir.glob("*.md") if p.is_file())


def _wiki_page_targets(wiki_dir: Path) -> set[str]:
    return {p.stem for p in _iter_wiki_markdown_files(wiki_dir)}


def _extract_link_targets(markdown: str) -> list[str]:
    return [m.group(1).strip() for m in _MARKDOWN_LINK_RE.finditer(markdown)]


def _normalize_target(raw_target: str) -> tuple[str, str | None]:
    # Strip optional angle brackets: (</path>) is valid markdown.
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">") and len(target) >= 2:
        target = target[1:-1].strip()

    if "#" in target:
        path_part, anchor = target.split("#", 1)
        return path_part.strip(), anchor.strip()
    return target, None


def _is_external_target(path_part: str) -> bool:
    if not path_part:
        return True
    if path_part.startswith("#"):
        return True
    return bool(_SCHEME_RE.match(path_part))


def _resolve_target_path(
    *,
    wiki_dir: Path,
    current_file: Path,
    path_part: str,
    page_names: set[str],
) -> Path | None:
    # GitHub wiki-style links typically omit ".md": (Getting-Started)
    if "/" not in path_part and not path_part.endswith(".md") and path_part in page_names:
        return wiki_dir / f"{path_part}.md"

    # If it's an md file without a path, interpret as wiki-relative.
    if "/" not in path_part and path_part.endswith(".md"):
        return wiki_dir / path_part

    # Otherwise, resolve relative to the current markdown file (handles ../README.md etc).
    if path_part.endswith(".md") or "/" in path_part or path_part.startswith("."):
        return (current_file.parent / path_part).resolve()

    return None


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    wiki_dir = repo_root / "wiki"
    if not wiki_dir.exists():
        print("wiki/ directory not found; skipping.", file=sys.stderr)
        return 0

    page_names = _wiki_page_targets(wiki_dir)
    failures: list[str] = []

    for md_file in _iter_wiki_markdown_files(wiki_dir):
        text = md_file.read_text(encoding="utf-8")
        for raw_target in _extract_link_targets(text):
            path_part, _anchor = _normalize_target(raw_target)
            if _is_external_target(path_part):
                continue

            resolved = _resolve_target_path(
                wiki_dir=wiki_dir,
                current_file=md_file,
                path_part=path_part,
                page_names=page_names,
            )
            if resolved is None:
                failures.append(
                    f"{md_file.relative_to(repo_root)}: unsupported link target: {raw_target}"
                )
                continue

            if not resolved.exists():
                failures.append(f"{md_file.relative_to(repo_root)}: broken link: {raw_target}")

    if failures:
        print("Broken wiki links detected:", file=sys.stderr)
        for line in failures:
            print(f"- {line}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
