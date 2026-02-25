"""Build in-toto Statements for each attestation step (00–09).

Each builder returns a Statement dict; the caller wraps it in a DSSE envelope
and writes to run_dir/attestations/NN.name.dsse.json. Builders accept minimal
context (digests, roots, config snapshots) so they can be called after the run
has produced manifest, records, receipts, etc.
"""

from __future__ import annotations

from typing import Any

from insideLLMs.attestations.statement import (
    PREDICATE_TYPE_EVAL_STEP,
    PREDICATE_TYPE_SLSA_PROVENANCE,
    build_statement,
)


def build_attestation_00_source(
    subject: list[dict[str, Any]],
    git_commit: str | None = None,
    git_dirty: bool | None = None,
    pyproject_digest: str | None = None,
    lock_digest: str | None = None,
    insidellms_version: str | None = None,
) -> dict[str, Any]:
    """00. Source attestation: git, pyproject.toml, lock file, insideLLMs version."""
    predicate: dict[str, Any] = {"step": "source"}
    if git_commit is not None:
        predicate["git_commit"] = git_commit
    if git_dirty is not None:
        predicate["git_dirty"] = git_dirty
    if pyproject_digest is not None:
        predicate["pyproject_digest"] = pyproject_digest
    if lock_digest is not None:
        predicate["lock_digest"] = lock_digest
    if insidellms_version is not None:
        predicate["insidellms_version"] = insidellms_version
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_01_env(
    subject: list[dict[str, Any]],
    python_version: str | None = None,
    platform: str | None = None,
    container_digest: str | None = None,
    sbom_digest: str | None = None,
) -> dict[str, Any]:
    """01. Environment attestation: container, python, platform, optional SBOM."""
    predicate: dict[str, Any] = {"step": "env"}
    if python_version is not None:
        predicate["python_version"] = python_version
    if platform is not None:
        predicate["platform"] = platform
    if container_digest is not None:
        predicate["container_digest"] = container_digest
    if sbom_digest is not None:
        predicate["sbom_digest"] = sbom_digest
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_02_dataset(
    subject: list[dict[str, Any]],
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    dataset_merkle_root: str | None = None,
    tuf_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """02. Dataset attestation: ID/version, Merkle root, TUF proof if used."""
    predicate: dict[str, Any] = {"step": "dataset"}
    if dataset_id is not None:
        predicate["dataset_id"] = dataset_id
    if dataset_version is not None:
        predicate["dataset_version"] = dataset_version
    if dataset_merkle_root is not None:
        predicate["dataset_merkle_root"] = dataset_merkle_root
    if tuf_verification is not None:
        predicate["tuf_verification"] = tuf_verification
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_03_promptset(
    subject: list[dict[str, Any]],
    template_digest: str | None = None,
    transform_pipeline_digest: str | None = None,
    promptset_merkle_root: str | None = None,
    sampling_seed: int | None = None,
    sampling_strategy: str | None = None,
) -> dict[str, Any]:
    """03. Promptset attestation: template, transform pipeline, Merkle root, sampling."""
    predicate: dict[str, Any] = {"step": "promptset"}
    if template_digest is not None:
        predicate["template_digest"] = template_digest
    if transform_pipeline_digest is not None:
        predicate["transform_pipeline_digest"] = transform_pipeline_digest
    if promptset_merkle_root is not None:
        predicate["promptset_merkle_root"] = promptset_merkle_root
    if sampling_seed is not None:
        predicate["sampling_seed"] = sampling_seed
    if sampling_strategy is not None:
        predicate["sampling_strategy"] = sampling_strategy
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_04_execution(
    subject: list[dict[str, Any]],
    records_digest: str | None = None,
    manifest_digest: str | None = None,
    records_merkle_root: str | None = None,
    receipts_merkle_root: str | None = None,
    model_identity_snapshot: dict[str, Any] | None = None,
    runner_config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """04. Execution attestation: records, manifest, Merkle roots, model identity, runner config."""
    builder = {"id": "https://insidellms.dev/runner/1.0"}
    invocation: dict[str, Any] = {}
    if runner_config_snapshot is not None:
        invocation["parameters"] = runner_config_snapshot
    materials: list[dict[str, Any]] = []
    if records_digest is not None:
        materials.append({"digest": {"sha256": records_digest}, "uri": "records.jsonl"})
    if manifest_digest is not None:
        materials.append({"digest": {"sha256": manifest_digest}, "uri": "manifest.json"})
    predicate: dict[str, Any] = {
        "buildType": "https://insidellms.dev/eval/v1",
        "builder": builder,
        "invocation": invocation,
        "materials": materials,
        "step": "execution",
    }
    if records_merkle_root is not None:
        predicate["records_merkle_root"] = records_merkle_root
    if receipts_merkle_root is not None:
        predicate["receipts_merkle_root"] = receipts_merkle_root
    if model_identity_snapshot is not None:
        predicate["model_identity_snapshot"] = model_identity_snapshot
    return build_statement(subject, PREDICATE_TYPE_SLSA_PROVENANCE, predicate)


def build_attestation_05_scoring(
    subject: list[dict[str, Any]],
    metrics_versions: dict[str, Any] | None = None,
    judge_committee_config: dict[str, Any] | None = None,
    analysis_plan_digest: str | None = None,
) -> dict[str, Any]:
    """05. Scoring attestation: metrics versions/params, judge config, analysis plan digest."""
    predicate: dict[str, Any] = {"step": "scoring"}
    if metrics_versions is not None:
        predicate["metrics_versions"] = metrics_versions
    if judge_committee_config is not None:
        predicate["judge_committee_config"] = judge_committee_config
    if analysis_plan_digest is not None:
        predicate["analysis_plan_digest"] = analysis_plan_digest
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_06_report(
    subject: list[dict[str, Any]],
    materials_digests: list[str] | None = None,
) -> dict[str, Any]:
    """06. Report attestation: subject = report.html, summary.json; materials from records/scoring."""
    predicate: dict[str, Any] = {"step": "report"}
    if materials_digests is not None:
        predicate["materials_digests"] = materials_digests
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_07_claims(
    subject: list[dict[str, Any]],
    claims_file_digest: str | None = None,
    verification_output_digest: str | None = None,
) -> dict[str, Any]:
    """07. Claims attestation: claims file digest, verification output digest."""
    predicate: dict[str, Any] = {"step": "claims"}
    if claims_file_digest is not None:
        predicate["claims_file_digest"] = claims_file_digest
    if verification_output_digest is not None:
        predicate["verification_output_digest"] = verification_output_digest
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_08_policy(
    subject: list[dict[str, Any]],
    policy_file_digest: str | None = None,
    verdict_digest: str | None = None,
    passed: bool | None = None,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    """08. Policy verdict attestation: policy digest, verdict digest, pass/fail, reasons."""
    predicate: dict[str, Any] = {"step": "policy"}
    if policy_file_digest is not None:
        predicate["policy_file_digest"] = policy_file_digest
    if verdict_digest is not None:
        predicate["verdict_digest"] = verdict_digest
    if passed is not None:
        predicate["passed"] = passed
    if reasons is not None:
        predicate["reasons"] = reasons
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)


def build_attestation_09_publish(
    subject: list[dict[str, Any]],
    oci_ref: str | None = None,
    oci_digest: str | None = None,
    signature_bundle_digests: list[str] | None = None,
) -> dict[str, Any]:
    """09. Publication attestation: OCI ref + digest, signature bundle digests."""
    predicate: dict[str, Any] = {"step": "publish"}
    if oci_ref is not None:
        predicate["oci_ref"] = oci_ref
    if oci_digest is not None:
        predicate["oci_digest"] = oci_digest
    if signature_bundle_digests is not None:
        predicate["signature_bundle_digests"] = signature_bundle_digests
    return build_statement(subject, PREDICATE_TYPE_EVAL_STEP, predicate)
