#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/insidellms-clean-install.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

wheelhouse="${work_dir}/wheelhouse"
source_dir="${work_dir}/source"
venv_dir="${work_dir}/venv"
smoke_dir="${work_dir}/smoke"
baseline_dir="${work_dir}/baseline"
candidate_dir="${work_dir}/candidate"
mkdir -p "${wheelhouse}" "${smoke_dir}"

echo "Copying the working tree into a disposable build context..."
REPO_ROOT="${repo_root}" SOURCE_DIR="${source_dir}" "${python_bin}" - <<'PY'
import os
import shutil
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
source_dir = Path(os.environ["SOURCE_DIR"])
shutil.copytree(
    repo_root,
    source_dir,
    ignore=shutil.ignore_patterns(
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp",
        ".venv*",
        "*.egg-info",
        "__pycache__",
        "build",
        "dist",
    ),
)
PY

echo "Building the project wheel and mandatory dependency closure..."
PIP_DISABLE_PIP_VERSION_CHECK=1 "${python_bin}" -m pip wheel \
  --wheel-dir "${wheelhouse}" \
  "${source_dir}"

echo "Installing from the local wheelhouse into an isolated virtual environment..."
"${python_bin}" -m venv "${venv_dir}"
PIP_DISABLE_PIP_VERSION_CHECK=1 "${venv_dir}/bin/python" -m pip install \
  --no-index \
  --find-links "${wheelhouse}" \
  insideLLMs

echo "Verifying the default install contains no optional feature dependencies..."
(
cd "${smoke_dir}"
REPO_ROOT="${repo_root}" VENV_DIR="${venv_dir}" \
  env -u PYTHONHOME -u PYTHONPATH "${venv_dir}/bin/python" -I - <<'PY'
import importlib.metadata
import importlib.util
import os
import re
import site
from pathlib import Path

requires = importlib.metadata.requires("insideLLMs") or []
mandatory_names = {
    re.split(r"\s*(?:\[|[<>=!~])", requirement.split(";", 1)[0], maxsplit=1)[0]
    .strip()
    .lower()
    .replace("_", "-")
    .replace(".", "-")
    for requirement in requires
    if "extra ==" not in requirement.lower()
}
assert mandatory_names == {"pyyaml"}, (mandatory_names, requires)

for module in ("anthropic", "fastapi", "matplotlib", "openai", "pydantic", "transformers"):
    assert importlib.util.find_spec(module) is None, f"optional dependency leaked: {module}"

import insideLLMs
package_path = Path(insideLLMs.__file__).resolve()
repo_root = Path(os.environ["REPO_ROOT"]).resolve()
site_roots = tuple(Path(path).resolve() for path in site.getsitepackages())
assert any(package_path.is_relative_to(root) for root in site_roots), (package_path, site_roots)
assert not package_path.is_relative_to(repo_root), package_path
print(f"import insideLLMs: {insideLLMs.__version__}")
print(f"installed package: {package_path}")
PY

echo "Verifying installed CLI help and deterministic harness -> diff workflow..."
env -u PYTHONHOME -u PYTHONPATH "${venv_dir}/bin/insidellms" --help >/dev/null
env -u PYTHONHOME -u PYTHONPATH "${venv_dir}/bin/insidellms" \
  --no-color harness "${repo_root}/ci/harness.yaml" \
  --run-dir "${baseline_dir}" --overwrite --skip-report
env -u PYTHONHOME -u PYTHONPATH "${venv_dir}/bin/insidellms" \
  --no-color harness "${repo_root}/ci/harness.yaml" \
  --run-dir "${candidate_dir}" --overwrite --skip-report
env -u PYTHONHOME -u PYTHONPATH "${venv_dir}/bin/insidellms" \
  --no-color diff "${baseline_dir}" "${candidate_dir}" \
  --fail-on-changes
)

for artifact in .insidellms_run config.resolved.yaml manifest.json records.jsonl results.jsonl summary.json; do
  cmp "${baseline_dir}/${artifact}" "${candidate_dir}/${artifact}"
done

echo "Clean-install golden path passed."
