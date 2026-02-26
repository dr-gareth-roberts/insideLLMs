# Design Plan: End-to-End Implementation of Core Stubs

## 1. `insideLLMs/claims/compiler.py` (`compile_claims`)
**Purpose**: Compile a `claims.yaml` document against the executed evaluation statistics (found in `summary.json` and `records.jsonl`) to produce `claims.json` and `verification.json`.

**Approach**: 
- Parse `claims.yaml` using `pyyaml` (already in the project dependencies).
- Load the `summary.json` generated from the run inside the specified `run_dir`.
- Iterate through each claim. A claim should have a predicate (e.g., `accuracy >= 0.9`). We will evaluate its predicate against the empirical metrics extracted from `summary.json`.
- Generate `claims.json` (the structured claims) and `verification.json` (the results of the predicate checks).
- **Integration**: Hook this into `insideLLMs/runtime/_ultimate.py` (specifically `run_ultimate_post_artifact()`), computing SHA-256 digests of these two new files and feeding them to the `build_attestation_07_claims()` builder.

## 2. `insideLLMs/evalbom.py` (`emit_cyclonedx` and `emit_spdx3`)
**Purpose**: Generate standard Software Bill of Materials (SBOMs) for the evaluation run.

**Approach**:
- **Manual Construction (Recommended)**: To minimize heavy external dependencies, we will construct the CycloneDX (v1.4) and SPDX (v3.0) JSON schemas manually.
- We will parse `manifest.json` and `config.resolved.yaml` from the `run_dir` to extract details about the model, probe, dataset, and runtime environment (e.g., python version, platform).
- We will construct the dependency graph of evaluation artifacts.
- **Integration**: Hook into `insideLLMs/runtime/_ultimate.py`, run this before `build_attestation_01_env()`, write the generated JSON to `evalbom.cyclonedx.json` and `evalbom.spdx3.json`, compute their digests, and pass the digest of `evalbom.cyclonedx.json` into the `sbom_digest` field of the environment attestation.

## 3. `insideLLMs/datasets/tuf_client.py` (`fetch_dataset`)
**Purpose**: Securely fetch datasets using The Update Framework (TUF) to guarantee integrity and provenance.

**Approach**:
- **Use `tuf` library (Recommended)**: We will add the official `tuf` (or `securesystemslib`) library to our dependencies in `pyproject.toml`.
- The `fetch_dataset` function will initialize a TUF updater targeting a provided `base_url`, verify the target metadata (signatures, expiration, etc.), and securely download the dataset file to the local cache.
- It will return the local `Path` to the downloaded dataset and a dictionary containing the TUF verification metadata.
- **Integration**: Hook into `insideLLMs/runtime/_config_loader.py` within the `_load_dataset_from_config()` flow when the source format is specified as `tuf`. The TUF metadata dictionary is then passed into the dataset spec and eventually into `build_attestation_02_dataset()`.