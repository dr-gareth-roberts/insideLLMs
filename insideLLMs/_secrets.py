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


# Auth-oriented query/fragment/header param names (conservative allowlist).
# Avoid bare "signature" substrings that would redact innocuous keys such as
# "signature_dish"; require the full normalized name or a known auth suffix.
_AUTH_PARAM_KEYS = frozenset(
    {
        "key",
        "sig",
        "signature",
        "xamzsignature",
        "xamzsecuritytoken",
        "xamzcredential",
        "securitytoken",
        "credential",
    }
)


def _is_secret_key(key: str) -> bool:
    normalized = _normalized_key(key)
    # Suffix matching covers namespaced keys and common HTTP header prefixes,
    # without hiding generation options such as max_tokens or token_budget.
    # _AUTH_PARAM_KEYS ("key", "sig", ...) are auth-bearing only as URL
    # parameters, so they are matched in _is_auth_url_param, not here.
    if normalized in _SECRET_KEYS:
        return True
    return normalized.endswith(
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
            "securitytoken",
            "webhooksecret",
            "secret",
            "signature",
        )
    )


def _is_auth_url_param(key: str) -> bool:
    return _is_secret_key(key) or _normalized_key(key) in _AUTH_PARAM_KEYS


def _redact_param_pairs(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], bool]:
    has_secrets = any(_is_auth_url_param(key) for key, _ in pairs)
    if not has_secrets:
        return pairs, False
    return (
        [(key, REDACTED if _is_auth_url_param(key) else val) for key, val in pairs],
        True,
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
        query_pairs, query_has_secrets = _redact_param_pairs(
            parse_qsl(parsed.query, keep_blank_values=True)
        )
        # OAuth-style tokens often land in the fragment (#access_token=...).
        fragment_pairs, fragment_has_secrets = _redact_param_pairs(
            parse_qsl(parsed.fragment, keep_blank_values=True)
        )
        if changed or query_has_secrets or fragment_has_secrets:
            query = urlencode(query_pairs) if query_has_secrets else parsed.query
            fragment = urlencode(fragment_pairs) if fragment_has_secrets else parsed.fragment
            return urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))
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
