import pytest


@pytest.fixture(scope="session", autouse=True)
def _nltk_data_root(tmp_path_factory):
    # NLTK downloads data to NLTK_DATA; keep it stable across the whole test run
    # so resources downloaded in one test are available in the next.
    monkeypatch = pytest.MonkeyPatch()
    nltk_data = tmp_path_factory.mktemp("nltk_data")
    monkeypatch.setenv("NLTK_DATA", str(nltk_data))
    yield nltk_data
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def _insidellms_run_root(tmp_path, monkeypatch):
    run_root = tmp_path / "insidellms_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INSIDELLMS_RUN_ROOT", str(run_root))
    return run_root
