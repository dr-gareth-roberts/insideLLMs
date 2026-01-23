from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _assert_jsonable(value: Any, *, where: str) -> Any:
    try:
        _canonical_json_bytes(value)
    except TypeError as e:
        raise ValueError(f"{where} must be JSON-serialisable: {e}") from e
    return value


def _normalise_sha256_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    if v.startswith("sha256:"):
        v = v[len("sha256:") :]
    return v


class TraceCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events_total: int = Field(ge=0)
    events_stored: int = Field(ge=0)
    by_kind: Dict[str, int] = Field(default_factory=dict)

    @field_validator("by_kind")
    @classmethod
    def _validate_by_kind(cls, v: Dict[str, int]) -> Dict[str, int]:
        for k, n in v.items():
            if not isinstance(k, str) or not k:
                raise ValueError("counts.by_kind keys must be non-empty strings")
            if not isinstance(n, int) or n < 0:
                raise ValueError("counts.by_kind values must be integers >= 0")
        return v

    @model_validator(mode="after")
    def _validate_totals(self) -> "TraceCounts":
        if self.events_total < self.events_stored:
            raise ValueError("counts.events_total must be >= counts.events_stored")
        return self


class TraceFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    alg: Literal["sha256"] = "sha256"
    value: Optional[str] = None
    basis: Optional[Literal["normalised_full_trace"]] = None

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: Optional[str]) -> Optional[str]:
        v = _normalise_sha256_value(v)
        if v is None:
            return None
        if not HEX64_RE.match(v):
            raise ValueError(
                "fingerprint.value must be a 64-hex sha256 "
                "(optionally prefixed with 'sha256:')"
            )
        return v.lower()

    @model_validator(mode="after")
    def _validate_enabled_semantics(self) -> "TraceFingerprint":
        if self.enabled:
            if self.value is None:
                raise ValueError("fingerprint.enabled=true requires fingerprint.value")
            if self.basis is None:
                raise ValueError("fingerprint.enabled=true requires fingerprint.basis")
        else:
            if self.value is not None:
                raise ValueError("fingerprint.enabled=false requires fingerprint.value=null")
            if self.basis is not None:
                raise ValueError("fingerprint.enabled=false requires fingerprint.basis=null")
        return self


class TraceNormaliser(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: Literal["builtin", "import"]
    name: Optional[str] = None
    import_path: Optional[str] = Field(default=None, alias="import")
    config_hash: str

    @field_validator("config_hash")
    @classmethod
    def _validate_config_hash(cls, v: str) -> str:
        v2 = _normalise_sha256_value(v) or ""
        if not HEX64_RE.match(v2):
            raise ValueError(
                "normaliser.config_hash must be a 64-hex sha256 "
                "(optionally prefixed with 'sha256:')"
            )
        return v2.lower()

    @model_validator(mode="after")
    def _validate_kind_semantics(self) -> "TraceNormaliser":
        if self.kind == "builtin":
            if not self.name:
                raise ValueError("normaliser.kind='builtin' requires normaliser.name")
            if self.import_path is not None:
                raise ValueError("normaliser.kind='builtin' forbids normaliser.import")
        if self.kind == "import":
            if not self.import_path:
                raise ValueError("normaliser.kind='import' requires normaliser.import")
            if self.name is not None:
                raise ValueError("normaliser.kind='import' forbids normaliser.name")
        return self


class TraceContractsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    fail_fast: bool = False
    violations_total: int = Field(ge=0)
    violations_stored: int = Field(ge=0)
    by_code: Dict[str, int] = Field(default_factory=dict)

    @field_validator("by_code")
    @classmethod
    def _validate_by_code(cls, v: Dict[str, int]) -> Dict[str, int]:
        for k, n in v.items():
            if not isinstance(k, str) or not k:
                raise ValueError("contracts.by_code keys must be non-empty strings")
            if not isinstance(n, int) or n < 0:
                raise ValueError("contracts.by_code values must be integers >= 0")
        return v

    @model_validator(mode="after")
    def _validate_totals(self) -> "TraceContractsSummary":
        if self.violations_total < self.violations_stored:
            raise ValueError("contracts.violations_total must be >= contracts.violations_stored")
        return self


class TraceViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    event_seq: Optional[int] = Field(default=None, ge=0)
    path: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("meta")
    @classmethod
    def _validate_meta_jsonable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        _assert_jsonable(v, where="violations[].meta")
        return v


class TraceEventStored(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0)
    kind: str = Field(min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _validate_payload_jsonable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        _assert_jsonable(v, where="events[].payload")
        return v


class TruncationEvents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    policy: Literal["head"] = "head"
    max_events: Optional[int] = Field(default=None, ge=0)
    dropped: int = Field(default=0, ge=0)
    dropped_by_kind: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_applied(self) -> "TruncationEvents":
        if self.applied:
            if self.max_events is None:
                raise ValueError(
                    "truncation.events.applied=true requires truncation.events.max_events"
                )
        return self


class TruncationPayloads(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    max_bytes: Optional[int] = Field(default=None, gt=0)
    omitted_fields: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_applied(self) -> "TruncationPayloads":
        if self.applied and self.max_bytes is None:
            raise ValueError(
                "truncation.payloads.applied=true requires truncation.payloads.max_bytes"
            )
        return self


class TruncationViolations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: bool
    max_violations: Optional[int] = Field(default=None, ge=0)
    dropped: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_applied(self) -> "TruncationViolations":
        if self.applied and self.max_violations is None:
            raise ValueError(
                "truncation.violations.applied=true requires "
                "truncation.violations.max_violations"
            )
        return self


class TraceTruncation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: TruncationEvents = Field(default_factory=lambda: TruncationEvents(applied=False))
    payloads: TruncationPayloads = Field(default_factory=lambda: TruncationPayloads(applied=False))
    violations: TruncationViolations = Field(
        default_factory=lambda: TruncationViolations(applied=False)
    )


class DerivedToolCalls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    sequence: List[str] = Field(default_factory=list)
    by_tool: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "DerivedToolCalls":
        if self.count != len(self.sequence):
            raise ValueError("derived.tool_calls.count must equal len(derived.tool_calls.sequence)")
        total = sum(self.by_tool.values())
        if total != self.count:
            raise ValueError(
                "sum(derived.tool_calls.by_tool.values()) must equal derived.tool_calls.count"
            )
        for k, n in self.by_tool.items():
            if not k:
                raise ValueError("derived.tool_calls.by_tool keys must be non-empty strings")
            if n < 0:
                raise ValueError("derived.tool_calls.by_tool counts must be >= 0")
        return self


class TraceDerived(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: DerivedToolCalls = Field(default_factory=lambda: DerivedToolCalls(count=0))


class TraceBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["insideLLMs.custom.trace@1"]
    mode: Literal["compact", "full"]
    counts: TraceCounts
    fingerprint: TraceFingerprint
    normaliser: TraceNormaliser
    contracts: TraceContractsSummary
    violations: List[TraceViolation] = Field(default_factory=list)
    events_view: Optional[Literal["raw", "normalised"]] = None
    events: Optional[List[TraceEventStored]] = None
    truncation: TraceTruncation = Field(default_factory=TraceTruncation)
    derived: TraceDerived = Field(default_factory=TraceDerived)

    @model_validator(mode="after")
    def _validate_bundle_semantics(self) -> "TraceBundleV1":
        if self.mode == "compact":
            if self.events is not None:
                raise ValueError("mode='compact' forbids events")
            if self.events_view is not None:
                raise ValueError("mode='compact' forbids events_view")
            if self.counts.events_stored != 0:
                raise ValueError("mode='compact' requires counts.events_stored == 0")

        if self.mode == "full":
            if self.events is None:
                raise ValueError("mode='full' requires events (can be empty list)")
            if self.events_view is None:
                raise ValueError("mode='full' requires events_view")
            if self.counts.events_stored != len(self.events):
                raise ValueError("counts.events_stored must equal len(events) in mode='full'")

            for i in range(1, len(self.events)):
                if self.events[i].seq < self.events[i - 1].seq:
                    raise ValueError("events[].seq must be non-decreasing")

        if not self.contracts.enabled:
            if self.violations:
                raise ValueError("contracts.enabled=false requires violations=[]")
            if self.contracts.violations_total != 0 or self.contracts.violations_stored != 0:
                raise ValueError(
                    "contracts.enabled=false requires violations_total=violations_stored=0"
                )

        if self.contracts.violations_stored != len(self.violations):
            raise ValueError("contracts.violations_stored must equal len(violations)")

        return self
