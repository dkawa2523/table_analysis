from __future__ import annotations
from typing import Any, Iterable, Mapping
def sanitize_component(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    return result or "item"
def needs_quote(text: str) -> bool:
    if not text:
        return True
    for ch in text:
        if ch.isspace() or ch in "[]{}(),=":
            return True
    return False
def quote_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"
def format_list(values: Iterable[Any]) -> str:
    items = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, bool):
            items.append("true" if item else "false")
        elif isinstance(item, (int, float)):
            items.append(str(item))
        else:
            items.append(quote_string(str(item)))
    return "[" + ",".join(items) + "]"
def format_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return format_list(value)
    text = str(value)
    if needs_quote(text):
        return quote_string(text)
    return text
def overrides_to_args(overrides: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        formatted = format_value(value)
        if formatted is None:
            continue
        args.append(f"{key}={formatted}")
    return args
def overrides_to_params(overrides: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in overrides.items():
        formatted = format_value(value)
        if formatted is None:
            continue
        params[key] = formatted
    return params
