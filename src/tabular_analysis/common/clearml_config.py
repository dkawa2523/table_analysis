from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_API_KEYS = ("web_server", "api_server", "files_server", "host", "files")


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def iter_clearml_config_candidates(*, repo_root: Path | None = None, config_file: str | Path | None = None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    explicit = _normalize_optional_str(config_file) or _normalize_optional_str(os.getenv("CLEARML_CONFIG_FILE"))
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if repo_root is not None:
        candidates.append(Path(repo_root).expanduser() / "clearml.conf")
    candidates.extend(
        [
            Path.cwd() / "clearml.conf",
            Path.home() / "clearml.conf",
            Path.home() / ".clearml.conf",
            Path.home() / ".config" / "clearml.conf",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


def _parse_hocon_api_section(text: str) -> dict[str, str]:
    api_match = re.search(r"(^|\n)\s*api\s*\{", text)
    if not api_match:
        return {}
    depth = 0
    start = api_match.end()
    end = start
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                end = idx
                break
            depth -= 1
    block = text[start:end]
    result: dict[str, str] = {}
    for key in _API_KEYS:
        match = re.search(rf'^\s*{re.escape(key)}\s*[:=]\s*"?([^"\r\n]+?)"?\s*$', block, flags=re.MULTILINE)
        if match:
            result[key] = match.group(1).strip()
    return result


def read_clearml_api_section(*, repo_root: Path | None = None, config_file: str | Path | None = None) -> dict[str, str]:
    for candidate in iter_clearml_config_candidates(repo_root=repo_root, config_file=config_file):
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = _parse_hocon_api_section(text)
        if parsed:
            return parsed
    return {}


def resolve_clearml_endpoint_summary(*, repo_root: Path | None = None, config_file: str | Path | None = None) -> dict[str, str | None]:
    api_section = read_clearml_api_section(repo_root=repo_root, config_file=config_file)
    api_host = _normalize_optional_str(os.getenv("CLEARML_API_HOST")) or _normalize_optional_str(api_section.get("api_server")) or _normalize_optional_str(api_section.get("host"))
    web_host = _normalize_optional_str(os.getenv("CLEARML_WEB_HOST")) or _normalize_optional_str(api_section.get("web_server")) or api_host
    files_host = _normalize_optional_str(os.getenv("CLEARML_FILES_HOST")) or _normalize_optional_str(api_section.get("files_server")) or _normalize_optional_str(api_section.get("files"))
    return {
        "api_host": api_host,
        "web_host": web_host,
        "files_host": files_host,
        "config_file": next((str(path) for path in iter_clearml_config_candidates(repo_root=repo_root, config_file=config_file) if path.exists()), None),
    }


__all__ = [
    "iter_clearml_config_candidates",
    "read_clearml_api_section",
    "resolve_clearml_endpoint_summary",
]
