#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*]\(([^)]+)\)")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _extract_front_matter(markdown: str) -> dict[str, str] | None:
    lines = markdown.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        return None
    idx += 1
    data: dict[str, str] = {}
    while idx < len(lines):
        line = lines[idx].rstrip("\n")
        if line.strip() == "---":
            return data
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('"').strip("'")
        idx += 1
    return None


def _iter_markdown_headings(markdown: str) -> list[str]:
    headings: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line.startswith("#"):
            continue
        if line.startswith("####"):
            title = line.lstrip("#").strip()
        else:
            title = line.lstrip("#").strip()
        if title:
            headings.append(title)
    return headings


def _slugify_anchor(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"`(.+?)`", r"\\1", slug)
    slug = re.sub(r"[^a-z0-9\s\-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


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
    seen_titles: dict[str, Path] = {}

    for md_file in _iter_wiki_markdown_files(wiki_dir):
        text = md_file.read_text(encoding="utf-8")

        for line_no, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                failures.append(
                    f"{md_file.relative_to(repo_root)}:{line_no}: contains a tab character"
                )
            if line.rstrip("\n") != line.rstrip("\n").rstrip(" "):
                failures.append(
                    f"{md_file.relative_to(repo_root)}:{line_no}: trailing whitespace"
                )

        front_matter = _extract_front_matter(text)
        if front_matter is None:
            failures.append(f"{md_file.relative_to(repo_root)}: missing YAML front matter")
        else:
            if not front_matter.get("title"):
                failures.append(f"{md_file.relative_to(repo_root)}: missing front matter 'title'")
            title = front_matter.get("title")
            if title:
                existing = seen_titles.get(title)
                if existing is not None:
                    failures.append(
                        f"{md_file.relative_to(repo_root)}: duplicate title {title!r} (also in {existing.relative_to(repo_root)})"
                    )
                else:
                    seen_titles[title] = md_file

            nav_order = front_matter.get("nav_order")
            if not nav_order:
                failures.append(
                    f"{md_file.relative_to(repo_root)}: missing front matter 'nav_order'"
                )
            elif not str(nav_order).isdigit():
                failures.append(
                    f"{md_file.relative_to(repo_root)}: non-numeric front matter 'nav_order'"
                )

        headings = _iter_markdown_headings(text)
        anchors = {_slugify_anchor(h) for h in headings if h}

        for raw_target in _extract_link_targets(text):
            path_part, anchor = _normalize_target(raw_target)
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
                continue

            if anchor:
                # Anchor-only links within the current page.
                if not path_part or path_part.startswith("#"):
                    if _slugify_anchor(anchor) not in anchors:
                        failures.append(
                            f"{md_file.relative_to(repo_root)}: broken anchor: {raw_target}"
                        )
                    continue

                # Anchors into other markdown files (best-effort).
                if resolved.suffix == ".md":
                    try:
                        target_text = resolved.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    target_anchors = {
                        _slugify_anchor(h)
                        for h in _iter_markdown_headings(target_text)
                        if h
                    }
                    if _slugify_anchor(anchor) not in target_anchors:
                        failures.append(
                            f"{md_file.relative_to(repo_root)}: broken anchor: {raw_target}"
                        )

    if failures:
        print("Wiki documentation issues detected:", file=sys.stderr)
        for line in failures:
            print(f"- {line}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
