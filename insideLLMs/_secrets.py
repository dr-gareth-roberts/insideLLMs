"""Remove credentials from configuration and metadata before persistence.

This is deliberately independent of provider SDKs and optional privacy packages.
It does not attempt to classify arbitrary prompt or model-output text as secrets.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

REDACTED = "[REDACTED]"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SECRET_KEYS = {
    "apikey",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "bearertoken",
    "idtoken",
    "token",
    "apitoken",
    "sessiontoken",
    "hftoken",
    "huggingfacetoken",
    "githubtoken",
    "gitlabtoken",
    "password",
    "passwd",
    "secret",
    "clientsecret",
    "secretkey",
    "privatekey",
    "authorization",
    "proxyauthorization",
    "cookie",
    "setcookie",
    "credentials",
    "secretaccesskey",
    "subscriptionkey",
}


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _is_secret_key(key: str) -> bool:
    normalized = _normalized_key(key)
    # Suffix matching covers namespaced keys and common HTTP header prefixes,
    # without hiding generation options such as max_tokens or token_budget.
    return normalized in _SECRET_KEYS or normalized.endswith(
        (
            "apikey",
            "accesstoken",
            "refreshtoken",
            "password",
            "clientsecret",
            "privatekey",
            "secretaccesskey",
            "subscriptionkey",
            "apitoken",
            "authtoken",
            "sessiontoken",
            "hftoken",
            "huggingfacetoken",
            "githubtoken",
            "gitlabtoken",
        )
    )


def _redact_url(value: str) -> str:
    if "://" not in value:
        return value
    try:
        parsed = urlsplit(value)
        if not parsed.netloc:
            return value
        netloc = parsed.netloc
        changed = "@" in netloc
        if changed:
            netloc = f"{quote(REDACTED, safe='')}@{netloc.rsplit('@', 1)[1]}"
        pairs = parse_qsl(parsed.query, keep_blank_values=True)

        def is_auth_query(key: str) -> bool:
            return _is_secret_key(key) or _normalized_key(key) in {"key", "sig", "signature"}

        query_has_secrets = any(is_auth_query(key) for key, _ in pairs)
        if query_has_secrets:
            pairs = [(key, REDACTED if is_auth_query(key) else val) for key, val in pairs]
        if changed or query_has_secrets:
            query = urlencode(pairs) if query_has_secrets else parsed.query
            return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    except ValueError:
        # Invalid URLs with embedded userinfo must not evade redaction.
        if "@" in value:
            return REDACTED
    return value


def redact_config_secrets(value: Any) -> Any:
    """Return a detached, recursively scrubbed configuration or metadata value.

    Credential environment variable *names* (for example ``api_key_env``) remain
    in provenance; their values are never read. Literal credentials are replaced
    with a stable marker, so rotating credentials does not change a run ID.
    """
    if callable(getattr(value, "get_secret_value", None)):
        return REDACTED
    if is_dataclass(value) and not isinstance(value, type):
        return redact_config_secrets(asdict(value))
    if isinstance(value, BaseModel):
        return redact_config_secrets(value.model_dump())
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if isinstance(key, str) and _normalized_key(key).endswith("env"):
                if _is_secret_key(key[:-3]):
                    result[key] = (
                        item
                        if item is None or (isinstance(item, str) and _ENV_NAME.fullmatch(item))
                        else REDACTED
                    )
                    continue
            if isinstance(key, str) and _is_secret_key(key):
                result[key] = None if item is None else REDACTED
            else:
                result[key] = redact_config_secrets(item)
        return result
    if isinstance(value, list):
        return [redact_config_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_config_secrets(item) for item in value)
    if isinstance(value, str):
        return _redact_url(value)
    return copy.deepcopy(value)
