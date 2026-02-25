"""Tests for attestations (DSSE envelope, Statement, step builders)."""

import json

import pytest

from insideLLMs.attestations import build_dsse_envelope, build_statement, parse_dsse_envelope
from insideLLMs.attestations.steps.builders import (
    build_attestation_00_source,
    build_attestation_04_execution,
    build_attestation_09_publish,
)


def test_build_dsse_envelope_roundtrip() -> None:
    """Build envelope, parse it, payload matches."""
    payload = {"_type": "Statement", "subject": [], "predicateType": "x", "predicate": {}}
    env = build_dsse_envelope(payload)
    assert env["payloadType"] == "application/vnd.in-toto+json"
    assert "payload" in env
    assert env["signatures"] == []
    parsed, pt = parse_dsse_envelope(env)
    assert pt == env["payloadType"]
    assert parsed == payload


def test_build_statement_has_type_and_subject() -> None:
    """Statement has _type, subject, predicateType, predicate."""
    s = build_statement(
        [{"name": "foo", "digest": {"sha256": "abc"}}],
        "https://example.org/pred/v1",
        {"key": "value"},
    )
    assert s["_type"] == "https://in-toto.io/Statement/v1"
    assert len(s["subject"]) == 1
    assert s["subject"][0]["name"] == "foo"
    assert s["predicateType"] == "https://example.org/pred/v1"
    assert s["predicate"] == {"key": "value"}


def test_build_attestation_00_source() -> None:
    """00 source attestation includes git and version fields."""
    st = build_attestation_00_source(
        subject=[{"name": "repo", "digest": {"sha256": "x"}}],
        git_commit="abc123",
        git_dirty=False,
        insidellms_version="0.1.0",
    )
    assert st["predicate"]["step"] == "source"
    assert st["predicate"]["git_commit"] == "abc123"
    assert st["predicate"]["insidellms_version"] == "0.1.0"


def test_build_attestation_04_execution() -> None:
    """04 execution attestation has SLSA predicate type and materials."""
    st = build_attestation_04_execution(
        subject=[{"name": "manifest.json", "digest": {"sha256": "m1"}}],
        records_digest="r1",
        manifest_digest="m1",
        records_merkle_root="root1",
    )
    assert "https://slsa.dev" in st["predicateType"]
    assert "materials" in st["predicate"]
    assert st["predicate"]["records_merkle_root"] == "root1"


def test_build_attestation_09_publish() -> None:
    """09 publish attestation has OCI ref and signature digests."""
    st = build_attestation_09_publish(
        subject=[{"name": "run", "digest": {"sha256": "bundle"}}],
        oci_ref="registry.io/repo:tag",
        oci_digest="sha256:abc",
        signature_bundle_digests=["d1", "d2"],
    )
    assert st["predicate"]["step"] == "publish"
    assert st["predicate"]["oci_ref"] == "registry.io/repo:tag"
    assert st["predicate"]["signature_bundle_digests"] == ["d1", "d2"]


def test_pae_format() -> None:
    """PAE encoding is deterministic for signers."""
    from insideLLMs.attestations.dsse import pae

    out = pae("application/vnd.in-toto+json", b'{"a":1}')
    assert out.startswith(b"DSSEV1 ")
    assert b"application" in out
    assert b'{"a":1}' in out
