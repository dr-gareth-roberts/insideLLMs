import pytest


@pytest.fixture(autouse=True)
def _insidellms_run_root(tmp_path, monkeypatch):
    run_root = tmp_path / "insidellms_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INSIDELLMS_RUN_ROOT", str(run_root))
    return run_root
