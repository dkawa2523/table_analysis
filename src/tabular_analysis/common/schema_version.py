from __future__ import annotations

import re
from typing import Any

from .config_utils import normalize_str as _normalize_str


def normalize_schema_version(value: Any, *, default: str = "unknown") -> str:
    text = _normalize_str(value)
    if not text:
        return default
    lowered = text.lower()
    if lowered == "unknown":
        return default
    if lowered.startswith("schema:"):
        lowered = lowered.split(":", 1)[1]
    lowered = lowered.strip("_")
    suffix = re.sub(r"^v+", "", lowered)
    if not suffix:
        return default
    return f"v{suffix}"


def build_schema_tag(value: Any, *, default: str = "unknown") -> str:
    return f"schema:{normalize_schema_version(value, default=default)}"

