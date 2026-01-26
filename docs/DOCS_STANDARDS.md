# Documentation Standards (GitHub Pages + Repo Docs)

This repository maintains two documentation surfaces:

- **Docs Site (GitHub Pages)**: Markdown source in `wiki/` rendered via Jekyll (Just the Docs).
- **Repo Docs**: Markdown files at repo root and under `docs/` (README, architecture, determinism, stability, etc.).

The goal is **high signal, low ambiguity** documentation with consistent structure, excellent navigation, and reproducible visuals.

## 1) Information architecture

- `README.md`: 1-page product overview + quickstart + links to the docs site.
- `wiki/`: user-facing guides and workflows (Pages site).
- `docs/`: contributor-facing policies and deep-dive specs (determinism, stability, plugins, golden path).
- `API_REFERENCE.md`: exhaustive API reference (generated/curated; avoid prose duplication).

## 2) Style and consistency rules

- Use **sentence case** for headings.
- Prefer **short paragraphs** and **bulleted lists** over long blocks of text.
- Use **imperative verbs** for procedural steps (“Run…”, “Configure…”, “Validate…”).
- Use consistent term casing:
  - `insideLLMs` (library/package)
  - `insidellms` (CLI)
  - “run directory”, “run artefacts”, “records.jsonl”, “manifest.json”, “diff.json”
- Avoid ambiguous pronouns (“it”, “this”) when referring to artefacts or commands.

## 3) Docs site page requirements (`wiki/*.md`)

Every page must include YAML front matter:

```yaml
---
title: <Page Title>
nav_order: <integer>
---
```

- `title` must be unique.
- `nav_order` must be stable; avoid renumbering unless restructuring navigation.

## 4) Visual documentation standards (Mermaid)

Use Mermaid diagrams for:

- Execution flows (CLI → runner → artefacts)
- Data flow (records → report → diff)
- Component relationships (registries/models/probes/datasets)

Rules:

- Diagrams must be **readable at 100% zoom**.
- Prefer **left-to-right** flow (`flowchart LR`) for pipelines.
- Keep diagrams small; if a diagram exceeds ~25 nodes, split it.
- Use explicit labels on nodes/edges.

## 5) Artefact contract documentation

Docs that describe artefacts must:

- Link to the relevant schema name (e.g., `ResultRecord`, `RunManifest`, `DiffReport`).
- Include at least one minimal example.
- Explicitly identify stable vs volatile fields.

## 6) Link hygiene

- Prefer wiki-relative links between wiki pages: `(Getting-Started)` or `(Getting-Started.md)`.
- For repo-root docs from wiki pages, use explicit relative paths: `(../README.md)`.
- Do not link to generated `_site/` output.

## 7) Review checklist (PR gate)

When changing docs or adding pages:

- The docs site builds successfully (Jekyll build).
- No broken internal links in `wiki/`.
- New pages have front matter with `title` and `nav_order`.
- Any new diagrams render correctly and are legible.
- Examples are copy/paste runnable (or explicitly marked as pseudo-code).
