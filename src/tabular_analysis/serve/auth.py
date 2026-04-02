from __future__ import annotations

import hashlib
from typing import Iterable


def api_key_fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:12]


def verify_api_key(value: str | None, expected_keys: Iterable[str]) -> bool:
    if not expected_keys:
        return True
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text in {str(item).strip() for item in expected_keys if str(item).strip()}


def resolve_principal(value: str | None, expected_keys: Iterable[str]) -> str:
    if not expected_keys:
        return "anonymous"
    if not verify_api_key(value, expected_keys):
        return "unauthorized"
    fingerprint = api_key_fingerprint(value)
    return f"api_key:{fingerprint}" if fingerprint else "api_key:unknown"

