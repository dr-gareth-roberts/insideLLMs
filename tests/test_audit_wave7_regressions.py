"""Regression tests for MONSTER_LOOP wave 7 fixes."""


def test_visualization_shim_sunset_documented_consistently():
    """IMPORT_PATHS and CHANGELOG must agree on visualization shim removal in v2.0.0."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    import_paths = (repo_root / "docs" / "IMPORT_PATHS.md").read_text(encoding="utf-8")
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    shim_doc = (repo_root / "insideLLMs" / "visualization.py").read_text(encoding="utf-8")

    assert "removal in v2.0.0" in import_paths
    assert "removed in v2.0.0" in changelog
    assert "removed in v2.0.0" in shim_doc
    assert "indefinitely" not in shim_doc.lower()
