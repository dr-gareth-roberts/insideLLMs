import pytest


@pytest.fixture(autouse=True)
def _insidellms_run_root(tmp_path, monkeypatch):
    run_root = tmp_path / "insidellms_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INSIDELLMS_RUN_ROOT", str(run_root))
    nltk_data = tmp_path / "nltk_data"
    nltk_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NLTK_DATA", str(nltk_data))
    return run_root
