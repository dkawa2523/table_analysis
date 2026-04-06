from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

from .common.clearml_config import read_clearml_api_section
from .common.repo_utils import resolve_repo_root_fallback as _resolve_repo_root


class PlatformAdapterError(RuntimeError):
    pass


_CLEARML_TASK_CACHE: dict[str, Any] = {}
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError)
RECOVERABLE_CLEARML_ERRORS = _RECOVERABLE_ERRORS


def _in_docker() -> bool:
    return Path('/.dockerenv').exists()


def _normalize_files_host(url: str, *, port_override: int | None = None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url if '://' in url else f'http://{url}')
    host = parsed.hostname
    if not host:
        return None
    scheme = parsed.scheme or 'http'
    port = port_override if port_override is not None else parsed.port
    if port is None:
        port = 8081
    return f'{scheme}://{host}:{port}'


def _read_clearml_config_api_section() -> dict[str, str]:
    return read_clearml_api_section(repo_root=_resolve_repo_root(fallback=Path(__file__).resolve().parents[2]))


def _resolve_clearml_files_host_fallback() -> str | None:
    api_section = _read_clearml_config_api_section()
    files_host = os.getenv('CLEARML_FILES_HOST') or api_section.get('files_server') or api_section.get('files')
    if files_host:
        normalized = _normalize_files_host(files_host)
        if normalized and urlparse(normalized).hostname not in {'localhost', '127.0.0.1'}:
            return normalized
    api_host = os.getenv('CLEARML_API_HOST') or os.getenv('CLEARML_WEB_HOST')
    if not api_host:
        api_host = api_section.get('host') or api_section.get('api_server') or api_section.get('web_server')
    if api_host:
        parsed = urlparse(api_host if '://' in api_host else f'http://{api_host}')
        host = parsed.hostname
        if host and host not in {'localhost', '127.0.0.1'}:
            port = parsed.port
            if port is None or port in {8008, 8080}:
                port = 8081
            return _normalize_files_host(api_host, port_override=port)
    return None


def _apply_clearml_files_host_substitution() -> None:
    in_docker = _in_docker()
    try:
        from clearml.backend_api.session import Session
        from clearml.storage.helper import StorageHelper
    except _RECOVERABLE_ERRORS:
        return
    try:
        files_host = Session.get_files_server_host()
    except _RECOVERABLE_ERRORS:
        files_host = None
    normalized = _normalize_files_host(files_host or '')
    if not in_docker and normalized:
        host = urlparse(normalized).hostname
        if host in {'clearml-fileserver', 'host.docker.internal'}:
            normalized = None
    if normalized:
        host = urlparse(normalized).hostname
        if host in {'localhost', '127.0.0.1'}:
            normalized = None
    if not normalized:
        normalized = _resolve_clearml_files_host_fallback()
    if not normalized:
        if in_docker:
            return
        normalized = _normalize_files_host(
            os.getenv('CLEARML_FILES_HOST') or _read_clearml_config_api_section().get('files_server') or ''
        )
        if not normalized:
            return
    current_env = _normalize_files_host(os.getenv('CLEARML_FILES_HOST') or '')
    current_host = urlparse(current_env).hostname if current_env else None
    should_override_env = (
        not current_env
        or (not in_docker and current_host in {'clearml-fileserver', 'host.docker.internal'})
        or (in_docker and current_host in {'localhost', '127.0.0.1'})
    )
    if should_override_env:
        os.environ['CLEARML_FILES_HOST'] = normalized
    try:
        existing = {rule.registered_prefix: rule.local_prefix for rule in StorageHelper._path_substitutions}
    except _RECOVERABLE_ERRORS:
        existing = {}
    extra_prefixes: tuple[str, ...] = ()
    if not in_docker and urlparse(normalized).hostname in {'localhost', '127.0.0.1'}:
        extra_prefixes = (
            'http://host.docker.internal:8081',
            'https://host.docker.internal:8081',
            'http://clearml-fileserver:8081',
            'https://clearml-fileserver:8081',
        )
    for prefix in (
        'http://localhost:8081',
        'http://127.0.0.1:8081',
        'https://localhost:8081',
        'https://127.0.0.1:8081',
        *extra_prefixes,
    ):
        if existing.get(prefix) == normalized:
            continue
        try:
            StorageHelper.add_path_substitution(prefix, normalized)
        except _RECOVERABLE_ERRORS:
            continue


def _dedupe_tags(tags: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag is None:
            continue
        tag_str = str(tag).strip()
        if not tag_str or tag_str in seen:
            continue
        seen.add(tag_str)
        result.append(tag_str)
    return result


def _normalize_requirement_lines(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items: Iterable[Any] = values.splitlines()
    elif isinstance(values, Iterable) and (not isinstance(values, Mapping)):
        items = values
    else:
        items = [values]
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text.lstrip().startswith('#'):
            continue
        normalized.append(text)
    return normalized


def _fully_unquote_text(text: str, *, max_rounds: int = 8) -> str:
    value = str(text).strip()
    for _ in range(max_rounds):
        collapsed = re.sub(r"%(?:%)+(?=[0-9A-Fa-f]{2})", "%", value)
        decoded = unquote(collapsed)
        if decoded == value:
            return decoded
        value = decoded
    return value


def _parameter_key_priority(raw_key: str) -> int:
    text = str(raw_key)
    if "%" in text:
        return 10
    if "/" in text and "." not in text:
        return 20
    if "." in text:
        return 30
    return 40


def _normalize_parameter_path(text: str, *, slash_as_dot: bool = True) -> str:
    normalized = _fully_unquote_text(text)
    if slash_as_dot:
        normalized = normalized.replace("/", ".")
    return normalized


def _flatten_parameter_payload(
    payload: Any,
    out: dict[str, Any],
    priorities: dict[str, int],
    *,
    prefix: str = "",
    source_priority: int = 40,
) -> None:
    if isinstance(payload, Mapping):
        for (key, value) in payload.items():
            raw_key = str(key)
            normalized_key = _normalize_parameter_path(raw_key)
            if not normalized_key:
                continue
            next_prefix = f"{prefix}.{normalized_key}" if prefix else normalized_key
            next_priority = min(source_priority, _parameter_key_priority(raw_key))
            _flatten_parameter_payload(
                value,
                out,
                priorities,
                prefix=next_prefix,
                source_priority=next_priority,
            )
        return
    if not prefix:
        return
    if priorities.get(prefix, -1) > source_priority:
        return
    priorities[prefix] = source_priority
    out[prefix] = payload


def _merge_nested_parameter_value(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        return
    node = target
    for key in parts[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[parts[-1]] = value


def _canonicalize_clearml_section_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    priorities: dict[str, int] = {}
    _flatten_parameter_payload(payload or {}, flattened, priorities)
    normalized: dict[str, Any] = {}
    for (path, value) in flattened.items():
        _merge_nested_parameter_value(normalized, path, value)
    return normalized


def _canonicalize_clearml_args_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    priorities: dict[str, int] = {}
    _flatten_parameter_payload(payload or {}, flattened, priorities)
    return {str(key): value for (key, value) in flattened.items() if key}


def _resolve_clearml_task(target: Any) -> Any:
    for name in ('task', '_task', 'pipeline_task'):
        if hasattr(target, name):
            value = getattr(target, name)
            if value is not None:
                return value
    return target


def _existing_user_properties(task: Any) -> dict[str, Any]:
    getter = getattr(task, 'get_user_properties', None)
    if callable(getter):
        try:
            existing = getter()
        except _RECOVERABLE_ERRORS:
            existing = None
        if isinstance(existing, Mapping):
            return dict(existing)
    return {}


def is_clearml_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg.run.clearml, 'enabled', False)) and str(getattr(cfg.run.clearml, 'execution', 'local')) != 'local'


apply_clearml_files_host_substitution = _apply_clearml_files_host_substitution
canonicalize_clearml_args_payload = _canonicalize_clearml_args_payload
canonicalize_clearml_section_payload = _canonicalize_clearml_section_payload
dedupe_tags = _dedupe_tags
existing_user_properties = _existing_user_properties
fully_unquote_text = _fully_unquote_text
normalize_requirement_lines = _normalize_requirement_lines
normalize_files_host = _normalize_files_host
resolve_clearml_task = _resolve_clearml_task


__all__ = [
    "PlatformAdapterError",
    "RECOVERABLE_CLEARML_ERRORS",
    "apply_clearml_files_host_substitution",
    "canonicalize_clearml_args_payload",
    "canonicalize_clearml_section_payload",
    "dedupe_tags",
    "existing_user_properties",
    "fully_unquote_text",
    "is_clearml_enabled",
    "normalize_files_host",
    "normalize_requirement_lines",
    "resolve_clearml_task",
]
