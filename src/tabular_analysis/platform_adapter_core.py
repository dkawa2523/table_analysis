from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import sys
from .common.config_utils import cfg_value as _cfg_value, set_cfg_value as _set_cfg_value
from .common.clearml_config import read_clearml_api_section
from .common.collection_utils import stringify_payload as _stringify_payload, to_container as _to_container
from .common.schema_version import build_schema_tag as _build_schema_tag, normalize_schema_version as _normalize_schema_version
from .common.repo_utils import resolve_repo_root_fallback as _resolve_repo_root
from collections.abc import Iterable as IterableABC
from dataclasses import make_dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Iterable, Mapping, Optional
from .platform_adapter_task_context import TaskContext
ClearMLScriptSpec = make_dataclass('ClearMLScriptSpec', [('repository', str | None), ('branch', str | None), ('entry_point', str | None), ('working_dir', str | None), ('version_policy', str), ('version_num', str | None)], frozen=True)
class PlatformAdapterError(RuntimeError):
    pass
_CLEARML_TASK_CACHE: dict[str, Any] = {}
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError)
def _in_docker() -> bool:
    return Path('/.dockerenv').exists()
def _normalize_files_host(url: str, *, port_override: int | None=None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url if '://' in url else f'http://{url}')
    host = parsed.hostname
    if not host:
        return None
    scheme = parsed.scheme or 'http'
    port = parsed.port
    if port_override is not None:
        port = port_override
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
    if normalized:
        host = urlparse(normalized).hostname
        if host in {'localhost', '127.0.0.1'}:
            normalized = None
    if not normalized:
        normalized = _resolve_clearml_files_host_fallback()
    if not normalized:
        if in_docker:
            return
        normalized = _normalize_files_host(os.getenv('CLEARML_FILES_HOST') or _read_clearml_config_api_section().get('files_server') or '')
        if not normalized:
            return
    os.environ.setdefault('CLEARML_FILES_HOST', normalized)
    try:
        existing = {rule.registered_prefix: rule.local_prefix for rule in StorageHelper._path_substitutions}
    except _RECOVERABLE_ERRORS:
        existing = {}
    extra_prefixes: tuple[str, ...] = ()
    if not in_docker and urlparse(normalized).hostname in {'localhost', '127.0.0.1'}:
        extra_prefixes = ('http://host.docker.internal:8081', 'https://host.docker.internal:8081')
    for prefix in ('http://localhost:8081', 'http://127.0.0.1:8081', 'https://localhost:8081', 'https://127.0.0.1:8081', *extra_prefixes):
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
        if not text:
            continue
        if text.lstrip().startswith('#'):
            continue
        normalized.append(text)
    return normalized
def _run_git_command(cmd: Iterable[str]) -> str | None:
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except _RECOVERABLE_ERRORS:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None
def _normalize_git_remote_url(value: str) -> str:
    text = value.strip()
    if text.startswith('git@') and ':' in text:
        (host_part, path) = text.split(':', 1)
        host = host_part.split('@', 1)[-1]
        text = f'https://{host}/{path}'
    elif text.startswith('ssh://') and '@' in text:
        rest = text[len('ssh://'):]
        (host_part, _, path) = rest.partition('/')
        host = host_part.split('@', 1)[-1]
        text = f'https://{host}/{path}'
    if text.endswith('.git'):
        text = text[:-4]
    return text
def detect_git_repository_url(repo_root: Path) -> str | None:
    value = _run_git_command(['git', '-C', str(repo_root), 'remote', 'get-url', 'origin'])
    if not value:
        return None
    return _normalize_git_remote_url(value)
def detect_git_branch(repo_root: Path) -> str | None:
    value = _run_git_command(['git', '-C', str(repo_root), 'rev-parse', '--abbrev-ref', 'HEAD'])
    if not value or value == 'HEAD':
        return None
    return value
def _normalize_code_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
def normalize_clearml_repository(value: Any) -> str | None:
    text = _normalize_code_ref(value)
    if not text:
        return None
    return _normalize_git_remote_url(text).rstrip('/')
def normalize_clearml_branch(value: Any) -> str | None:
    return _normalize_code_ref(value)
def normalize_clearml_entry_point(value: Any) -> str | None:
    text = _normalize_code_ref(value)
    if not text:
        return None
    parts = text.split()
    if parts and parts[0] in {'python', 'python3'}:
        parts = parts[1:]
    normalized = ' '.join(parts).strip()
    if not normalized:
        return None
    (command, _) = _split_entry_point_command(normalized)
    return command or None
def normalize_clearml_version_num(value: Any) -> str | None:
    return _normalize_code_ref(value)
def hydra_list(values: list[str]) -> str:
    return '[' + ','.join(values) + ']'
def _parse_json_list(text: str) -> list[Any] | None:
    if not (text.startswith('[') and text.endswith(']')):
        return None
    try:
        parsed = json.loads(text)
    except _RECOVERABLE_ERRORS:
        return None
    if isinstance(parsed, list):
        return parsed
    return None
def _split_bracket_list(text: str) -> list[str] | None:
    if not (text.startswith('[') and text.endswith(']')):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    for item in inner.split(','):
        cleaned = item.strip().strip('\'"').strip()
        if cleaned:
            items.append(cleaned)
    return items
def _coerce_hydra_list_value(value: Any) -> list[str]:
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _parse_json_list(text)
        if parsed is None:
            parsed = _split_bracket_list(text)
        items = parsed if parsed is not None else [text]
    elif isinstance(value, Mapping):
        return []
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, IterableABC):
        items = list(value)
    else:
        items = [value]
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized
_ENTRYPOINT_OVERRIDE_RE = re.compile('\\s[+~]?[^\\s=]+=')
def _split_entry_point_command(entry_point: str) -> tuple[str, str]:
    text = str(entry_point).strip()
    if not text:
        return ('', '')
    match = _ENTRYPOINT_OVERRIDE_RE.search(text)
    if not match:
        return (text, '')
    idx = match.start()
    return (text[:idx].strip(), text[idx:].strip())
def _canonicalize_pipeline_entrypoint(cfg: Any, entry_point: str | None, fallback: str | None, task_name_override: str | None=None, canonicalize_pipeline: bool=True) -> str | None:
    if not canonicalize_pipeline:
        return entry_point
    task_name = _normalize_code_ref(task_name_override) or _normalize_code_ref(_cfg_value(cfg, 'task.name'))
    if task_name != 'pipeline':
        return entry_point
    return entry_point or fallback or 'tools/clearml_entrypoint.py'
def _resolve_clearml_entrypoint(cfg: Any, current_entry_point: Any, entry_point_override: str | None, task_name_override: str | None=None, canonicalize_pipeline: bool=True) -> str | None:
    current_text = _normalize_code_ref(current_entry_point)
    base = current_text
    if entry_point_override is not None:
        override_text = str(entry_point_override).strip()
        base = override_text or current_text
    return _canonicalize_pipeline_entrypoint(cfg, base, entry_point_override or current_text, task_name_override=task_name_override, canonicalize_pipeline=canonicalize_pipeline)
def resolve_clearml_code_reference(cfg: Any) -> tuple[str | None, str | None]:
    repo_value = _normalize_code_ref(_cfg_value(cfg, 'run.clearml.code_ref.repository'))
    branch_value = _normalize_code_ref(_cfg_value(cfg, 'run.clearml.code_ref.branch'))
    repo_root = _resolve_repo_root()
    if repo_value and repo_value.lower() == 'auto':
        repo_value = detect_git_repository_url(repo_root)
    if branch_value and branch_value.lower() == 'auto':
        branch_value = detect_git_branch(repo_root)
    return (repo_value, branch_value)
def _resolve_clearml_entrypoint_override(cfg: Any) -> str | None:
    execution_value = _normalize_code_ref(_cfg_value(cfg, 'run.clearml.execution'))
    if execution_value is None:
        return None
    execution = execution_value.lower()
    if execution in {'agent', 'clone', 'pipeline_controller'}:
        return 'tools/clearml_entrypoint.py'
    return None
def _resolve_clearml_code_ref_mode(cfg: Any, *, override: str | None=None) -> str:
    text = _normalize_code_ref(override) or _normalize_code_ref(_cfg_value(cfg, 'run.clearml.code_ref.mode'))
    if not text:
        return 'branch_head'
    lowered = text.lower()
    if lowered in {'branch_head', 'branch', 'head'}:
        return 'branch_head'
    if lowered in {'pin_commit', 'commit', 'pinned'}:
        return 'pin_commit'
    if lowered in {'none', 'off', 'disabled'}:
        return 'none'
    return 'branch_head'
def _resolve_clearml_version_num(cfg: Any, *, version_mode_override: str | None=None) -> tuple[str, str | None]:
    mode = _resolve_clearml_code_ref_mode(cfg, override=version_mode_override)
    if mode == 'pin_commit':
        commit_override = _normalize_code_ref(_cfg_value(cfg, 'run.clearml.code_ref.commit'))
        if commit_override and commit_override.lower() != 'auto':
            return (mode, commit_override)
        repo_root = _resolve_repo_root()
        commit = _run_git_command(['git', '-C', str(repo_root), 'rev-parse', 'HEAD'])
        if not commit:
            raise PlatformAdapterError('Failed to resolve git commit for pin_commit.')
        return (mode, commit)
    if mode == 'none':
        return (mode, None)
    return (mode, '')
def resolve_clearml_script_spec(cfg: Any, *, current_entry_point: Any | None=None, repo_override: str | None=None, branch_override: str | None=None, entry_point_override: str | None=None, working_dir_override: str | None=None, version_mode_override: str | None=None, task_name_override: str | None=None, canonicalize_pipeline: bool=True) -> ClearMLScriptSpec:
    (repo_value, branch_value) = resolve_clearml_code_reference(cfg)
    if repo_override is not None:
        repo_value = repo_override
    if branch_override is not None:
        branch_value = branch_override
    entry_override = entry_point_override if entry_point_override is not None else _resolve_clearml_entrypoint_override(cfg)
    entry_point = _resolve_clearml_entrypoint(cfg, current_entry_point, entry_override, task_name_override=task_name_override, canonicalize_pipeline=canonicalize_pipeline)
    working_dir = _normalize_code_ref(working_dir_override) or _normalize_code_ref(_cfg_value(cfg, 'run.clearml.working_dir'))
    (version_policy, version_num) = _resolve_clearml_version_num(cfg, version_mode_override=version_mode_override)
    return ClearMLScriptSpec(repository=repo_value, branch=branch_value, entry_point=entry_point, working_dir=working_dir, version_policy=version_policy, version_num=version_num)
def _commit_matches(expected: str | None, actual: str | None) -> bool:
    if not expected:
        return not actual
    if not actual:
        return False
    expected_norm = expected.lower()
    actual_norm = actual.lower()
    if expected_norm == actual_norm:
        return True
    if expected_norm.startswith(actual_norm) or actual_norm.startswith(expected_norm):
        return min(len(expected_norm), len(actual_norm)) >= 7
    return False
def clearml_script_mismatches(spec: ClearMLScriptSpec, script: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_repo = normalize_clearml_repository(spec.repository)
    if expected_repo:
        actual_repo = normalize_clearml_repository(script.get('repository'))
        if actual_repo != expected_repo:
            errors.append(f"repository mismatch: {actual_repo or 'none'}")
    expected_branch = normalize_clearml_branch(spec.branch)
    if expected_branch:
        actual_branch = normalize_clearml_branch(script.get('branch'))
        if actual_branch != expected_branch:
            errors.append(f"branch mismatch: {actual_branch or 'none'}")
    expected_entry = normalize_clearml_entry_point(spec.entry_point)
    if expected_entry:
        actual_entry = normalize_clearml_entry_point(script.get('entry_point'))
        if actual_entry != expected_entry:
            errors.append(f"entry_point mismatch: {actual_entry or 'none'}")
    expected_working = _normalize_code_ref(spec.working_dir)
    if expected_working:
        actual_working = _normalize_code_ref(script.get('working_dir'))
        if actual_working != expected_working:
            errors.append(f"working_dir mismatch: {actual_working or 'none'}")
    actual_version = normalize_clearml_version_num(script.get('version_num'))
    if spec.version_policy == 'branch_head':
        if actual_version:
            errors.append(f'version_num mismatch: {actual_version}')
    elif spec.version_policy == 'none':
        return errors
    elif spec.version_policy == 'pin_commit':
        expected_version = normalize_clearml_version_num(spec.version_num)
        if not _commit_matches(expected_version, actual_version):
            errors.append(f"version_num mismatch: {actual_version or 'none'}")
    return errors
def _resolve_clearml_task(target: Any) -> Any:
    for name in ('task', '_task', 'pipeline_task'):
        if hasattr(target, name):
            value = getattr(target, name)
            if value is not None:
                return value
    return target
def _task_tags(task: Any) -> list[str]:
    from . import platform_adapter_task_ops as _task_ops
    return _task_ops._task_tags(task)
def _task_script(task: Any) -> dict[str, Any]:
    from . import platform_adapter_task_ops as _task_ops
    return _task_ops._task_script(task)
def _task_parameters(task: Any) -> dict[str, Any]:
    from . import platform_adapter_task_ops as _task_ops
    return _task_ops._task_parameters(task)
def _set_clearml_task_script(task: Any, payload: Mapping[str, Any]) -> None:
    setter = getattr(task, 'set_script', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_script is not available.')
    try:
        setter(**payload)
        return
    except TypeError:
        pass
    if 'version_num' in payload:
        commit_payload = dict(payload)
        commit_payload['commit'] = commit_payload.pop('version_num')
        try:
            setter(**commit_payload)
            return
        except TypeError:
            pass
    try:
        setter(script=dict(payload))
        return
    except TypeError:
        if 'version_num' not in payload:
            raise
        trimmed = dict(payload)
        trimmed.pop('version_num', None)
        setter(**trimmed)
def _apply_clearml_task_script_override(target: Any, cfg: Any) -> bool:
    task = _resolve_clearml_task(target)
    current = _task_script(task)
    current_repo = current.get('repository')
    current_branch = current.get('branch')
    current_entry_point = current.get('entry_point')
    spec = resolve_clearml_script_spec(cfg, current_entry_point=current_entry_point)
    if not clearml_script_mismatches(spec, current):
        return False
    setter = getattr(task, 'set_script', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_script is not available.')
    payload: dict[str, Any] = {}
    repo_to_set = spec.repository if spec.repository is not None else current_repo
    branch_to_set = spec.branch if spec.branch is not None else current_branch
    if repo_to_set is not None:
        payload['repository'] = str(repo_to_set)
    if branch_to_set is not None:
        payload['branch'] = str(branch_to_set)
    entry_point = spec.entry_point if spec.entry_point is not None else current_entry_point
    if entry_point is not None:
        payload['entry_point'] = str(entry_point)
    working_dir = spec.working_dir if spec.working_dir is not None else current.get('working_dir')
    if working_dir is not None:
        payload['working_dir'] = working_dir
    if spec.version_num is not None:
        payload['version_num'] = spec.version_num
    if not payload:
        return False
    _set_clearml_task_script(task, payload)
    return True
def _apply_clearml_task_args(task: Any, args: Mapping[str, Any]) -> bool:
    if not args:
        return False
    params = _task_parameters(task)
    existing_args: dict[str, str] = {}
    for (key, value) in params.items():
        if isinstance(key, str) and key.startswith('Args/'):
            existing_args[key[5:]] = '' if value is None else str(value)
    updates: dict[str, str] = {}
    for (key, value) in args.items():
        expected = '' if value is None else str(value)
        if existing_args.get(str(key)) != expected:
            updates[str(key)] = expected
    if not updates:
        return False
    updated_params = dict(params)
    for (key, value) in updates.items():
        updated_params[f'Args/{key}'] = value
    setter = getattr(task, 'set_parameters', None)
    if callable(setter):
        setter(updated_params)
        return True
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        merged = {**existing_args, **updates}
        setter({'Args': merged})
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters is not available.')
def _apply_clearml_pipeline_args(target: Any, cfg: Any) -> bool:
    task = _resolve_clearml_task(target)
    preprocess_variants = _coerce_hydra_list_value(_cfg_value(cfg, 'pipeline.grid.preprocess_variants'))
    model_variants = _coerce_hydra_list_value(_cfg_value(cfg, 'pipeline.grid.model_variants'))
    if not preprocess_variants and (not model_variants):
        return False
    args: dict[str, Any] = {}
    if preprocess_variants:
        args['pipeline.grid.preprocess_variants'] = hydra_list(preprocess_variants)
    if model_variants:
        args['pipeline.grid.model_variants'] = hydra_list(model_variants)
    return _apply_clearml_task_args(task, args)
def _apply_clearml_task_requirements(task: Any, requirements: Iterable[str]) -> bool:
    normalized = _normalize_requirement_lines(requirements)
    if not normalized:
        return False
    setter = getattr(task, 'set_packages', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_packages is not available.')
    try:
        setter(normalized)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to set task requirements via ClearML: {exc}') from exc
    return True
def _resolve_clearml_pipeline_requirements(cfg: Any) -> list[str]:
    _ = cfg
    return ['clearml>=1.15.0', 'uv>=0.5.0']
def _resolve_version_props(cfg: Any, *, clearml_enabled: bool) -> dict[str, str]:
    try:
        from ml_platform.versioning import get_code_version, get_platform_version, get_schema_version
    except _RECOVERABLE_ERRORS as exc:
        if clearml_enabled:
            raise PlatformAdapterError('ml_platform.versioning is required for ClearML runs. Install/update ml_platform.') from exc
        return {'code_version': 'unknown', 'platform_version': 'unknown', 'schema_version': 'unknown'}
    schema_version = _cfg_value(cfg, 'run.schema_version') or get_schema_version(cfg, default='unknown')
    return {'code_version': get_code_version(repo_root=Path.cwd()), 'platform_version': get_platform_version(), 'schema_version': _normalize_schema_version(schema_version, default='unknown')}
def build_clearml_properties(cfg: Any, *, stage: str, task_name: str, extra: Optional[Mapping[str, Any]], clearml_enabled: bool) -> dict[str, Any]:
    usecase_id = _cfg_value(cfg, 'run.usecase_id') or _cfg_value(cfg, 'usecase_id') or 'unknown'
    process = _cfg_value(cfg, 'task.name') or task_name or stage or 'unknown'
    versions = _resolve_version_props(cfg, clearml_enabled=clearml_enabled)
    grid_run_id = _cfg_value(cfg, 'run.grid_run_id')
    retrain_run_id = _cfg_value(cfg, 'run.retrain_run_id')
    base: dict[str, Any] = {'usecase_id': usecase_id, 'process': process, 'schema_version': versions.get('schema_version', 'unknown'), 'code_version': versions.get('code_version', 'unknown'), 'platform_version': versions.get('platform_version', 'unknown'), 'grid_run_id': grid_run_id}
    if retrain_run_id:
        base['retrain_run_id'] = retrain_run_id
    merged = dict(base)
    if extra:
        merged.update(dict(extra))
    for (key, value) in base.items():
        merged.setdefault(key, value)
    return merged
def build_clearml_tags(cfg: Any, *, process: str, schema_version: str, grid_run_id: Any, retrain_run_id: Any, extra_tags: Optional[Iterable[str]], tags: Optional[Iterable[str]]) -> list[str]:
    usecase_id = _cfg_value(cfg, 'run.usecase_id') or _cfg_value(cfg, 'usecase_id') or 'unknown'
    base = [f'usecase:{usecase_id}', f'process:{process}', _build_schema_tag(schema_version)]
    if grid_run_id:
        base.append(f'grid:{grid_run_id}')
    if retrain_run_id:
        base.append(f'retrain:{retrain_run_id}')
    return _dedupe_tags([*base, *(extra_tags or []), *(tags or [])])
def _ensure_clearml_names(cfg: Any, *, project_name: str, task_name: str, clearml_enabled: bool) -> None:
    if _cfg_value(cfg, 'run.clearml.project_name') != project_name:
        _set_cfg_value(cfg, 'run.clearml.project_name', project_name)
    if _cfg_value(cfg, 'run.clearml.task_name') != task_name:
        _set_cfg_value(cfg, 'run.clearml.task_name', task_name)
    if clearml_enabled:
        if _cfg_value(cfg, 'run.clearml.project_name') != project_name:
            raise PlatformAdapterError('Failed to set run.clearml.project_name for ClearML init.')
        if _cfg_value(cfg, 'run.clearml.task_name') != task_name:
            raise PlatformAdapterError('Failed to set run.clearml.task_name for ClearML init.')
def _load_clearml_module(clearml_enabled: bool):
    try:
        from ml_platform.integrations import clearml as platform_clearml
    except _RECOVERABLE_ERRORS as exc:
        if clearml_enabled:
            raise PlatformAdapterError('ml_platform.integrations.clearml is required for ClearML runs. Install/update ml_platform.') from exc
        return None
    _patch_platform_clearml(platform_clearml)
    return platform_clearml
def _normalize_clearml_user_properties(properties: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for (key, value) in properties.items():
        if isinstance(value, Mapping) and 'value' in value:
            value = value.get('value')
        normalized[str(key)] = '' if value is None else str(value)
    return normalized
def _patch_platform_clearml(platform_clearml: Any) -> None:
    if platform_clearml is None:
        return
    if getattr(platform_clearml, '_ta_user_properties_patch', False):
        return
    def _safe_set_user_properties(task: Any, properties: Mapping[str, Any] | None) -> None:
        if not properties:
            return
        setter = getattr(task, 'set_user_properties', None)
        if not callable(setter):
            return
        normalized = _normalize_clearml_user_properties(dict(properties))
        try:
            setter(**normalized)
            return
        except TypeError:
            pass
        try:
            setter(*normalized.items())
            return
        except TypeError:
            setter(normalized)
    platform_clearml.set_user_properties = _safe_set_user_properties
    platform_clearml._ta_user_properties_patch = True
def _load_clearml_dataset(clearml_enabled: bool):
    try:
        from clearml import Dataset as ClearMLDataset
    except _RECOVERABLE_ERRORS as exc:
        if clearml_enabled:
            raise PlatformAdapterError('clearml.Dataset is required for ClearML dataset registration. Install/update clearml.') from exc
        return None
    return ClearMLDataset
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
def clearml_task_type_controller() -> str:
    return 'controller'
def _apply_clearml_system_tags(task: Any, system_tags: Iterable[str] | None) -> None:
    if not system_tags:
        return
    getter = getattr(task, 'get_system_tags', None)
    setter = getattr(task, 'set_system_tags', None)
    if not callable(getter) or not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_system_tags is not available.')
    try:
        current = getter() or []
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to read ClearML system tags: {exc}') from exc
    merged = _dedupe_tags([*current, *system_tags])
    try:
        setter(merged)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to set ClearML system tags: {exc}') from exc


def _load_clearml_project_record(project_name: str | None) -> tuple[str | None, Mapping[str, Any] | None]:
    if not project_name:
        return (None, None)
    try:
        from clearml.backend_api.session.client import APIClient
    except _RECOVERABLE_ERRORS:
        APIClient = None
    if APIClient is not None:
        try:
            client = APIClient()
            projects = client.projects.get_all(name=str(project_name))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'ClearML project lookup failed: {exc}') from exc
        for candidate in list(projects or []):
            name = getattr(candidate, 'name', None)
            if str(name or '') == str(project_name):
                project_id = getattr(candidate, 'id', None)
                return (str(project_id) if project_id else None, candidate)
        if projects:
            candidate = projects[0]
            project_id = getattr(candidate, 'id', None)
            return (str(project_id) if project_id else None, candidate)
    try:
        from clearml.backend_api.session import Session
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'ClearML Session is not available: {exc}') from exc
    try:
        session = Session()
        response = session.send_request(service='projects', action='get_all', json={'name': project_name, 'search_hidden': True, 'size': 10})
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'ClearML project lookup failed: {exc}') from exc
    if not getattr(response, 'ok', False):
        raise PlatformAdapterError(f'ClearML project lookup returned non-ok response for {project_name!r}.')
    try:
        payload = response.json() or {}
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to parse ClearML project lookup payload: {exc}') from exc
    projects = []
    if isinstance(payload, Mapping) and isinstance(payload.get('projects'), list):
        projects = [proj for proj in payload.get('projects', []) if isinstance(proj, Mapping)]
    elif isinstance(payload, Mapping) and isinstance(payload.get('data'), Mapping):
        candidates = payload.get('data', {}).get('projects')
        if isinstance(candidates, list):
            projects = [proj for proj in candidates if isinstance(proj, Mapping)]
    for candidate in projects:
        name = candidate.get('name') or candidate.get('full_name') or candidate.get('path')
        if name == project_name:
            project_id = candidate.get('id') or candidate.get('project') or candidate.get('project_id')
            return (str(project_id) if project_id else None, candidate)
    if projects:
        candidate = projects[0]
        project_id = candidate.get('id') or candidate.get('project') or candidate.get('project_id')
        return (str(project_id) if project_id else None, candidate)
    return (None, None)


def _get_clearml_project_system_tags(project_name: str | None) -> list[str]:
    (_, project) = _load_clearml_project_record(project_name)
    if project is None:
        return []
    system_tags = getattr(project, 'system_tags', None)
    if system_tags is None and isinstance(project, Mapping):
        system_tags = project.get('system_tags')
    if isinstance(system_tags, list):
        return _dedupe_tags(system_tags)
    try:
        return _dedupe_tags(list(system_tags or []))
    except _RECOVERABLE_ERRORS:
        return []


def _ensure_clearml_project_system_tags(project_name: str | None, add_tags: Iterable[str] | None=None, *, remove_tags: Iterable[str] | None=None) -> None:
    if not project_name:
        return
    add_list = _dedupe_tags(add_tags or [])
    remove_set = {tag for tag in _dedupe_tags(remove_tags or []) if tag}
    if not add_list and (not remove_set):
        return
    (project_id, project) = _load_clearml_project_record(project_name)
    if not project_id:
        raise PlatformAdapterError(f"ClearML project not found: {project_name!r}")
    system_tags = _get_clearml_project_system_tags(project_name)
    merged = _dedupe_tags([*system_tags, *add_list])
    if remove_set:
        merged = [tag for tag in merged if tag not in remove_set]
    if merged == system_tags:
        return
    try:
        from clearml.backend_api.session import Session
        session = Session()
        update = session.send_request(service='projects', action='update', json={'project': project_id, 'system_tags': merged})
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'ClearML project update failed: {exc}') from exc
    if not getattr(update, 'ok', False):
        raise PlatformAdapterError(f'ClearML project update returned non-ok response for {project_name!r}.')
    if _get_clearml_project_system_tags(project_name) != merged:
        raise PlatformAdapterError(f'ClearML project system tags did not persist for {project_name!r}.')

def _apply_clearml_tags(task: Any, tags: Iterable[str] | None) -> None:
    tag_list = _dedupe_tags(tags or [])
    if not tag_list:
        return
    existing = _task_tags(task)
    merged = _dedupe_tags([*existing, *tag_list])
    setter = getattr(task, 'set_tags', None)
    if callable(setter):
        try:
            setter(merged)
            return
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to set ClearML tags: {exc}') from exc
    adder = getattr(task, 'add_tags', None)
    if not callable(adder):
        raise PlatformAdapterError('ClearML Task.add_tags is not available.')
    missing = [tag for tag in tag_list if tag not in existing]
    if not missing:
        return
    try:
        adder(missing)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to add ClearML tags: {exc}') from exc
def _apply_clearml_task_type(task: Any, task_type: str | None) -> None:
    if not task_type:
        return
    setter = getattr(task, 'set_task_type', None)
    if not callable(setter):
        return
    try:
        from clearml import Task as ClearMLTask
        normalized = task_type
        if isinstance(task_type, str) and task_type.lower() == 'controller':
            normalized = ClearMLTask.TaskTypes.controller
        setter(normalized)
    except _RECOVERABLE_ERRORS:
        try:
            setter(task_type)
        except _RECOVERABLE_ERRORS:
            return
def _hash_artifact_payload(payload: Any, *, symbol: str) -> str:
    try:
        from ml_platform import artifacts as platform_artifacts
    except _RECOVERABLE_ERRORS as exc:
        platform_artifacts = None
    hasher = getattr(platform_artifacts, symbol, None)
    if callable(hasher):
        return hasher(payload)
    normalized = _stringify_payload(_to_container(payload, resolve=True))
    data = json.dumps(normalized, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(data).hexdigest()
def hash_config(payload: Any) -> str:
    return _hash_artifact_payload(payload, symbol='hash_config')
def hash_split(payload: Any) -> str:
    return _hash_artifact_payload(payload, symbol='hash_split')
def hash_recipe(payload: Any) -> str:
    return _hash_artifact_payload(payload, symbol='hash_recipe')
def is_clearml_enabled(cfg) -> bool:
    return bool(getattr(cfg.run.clearml, 'enabled', False)) and str(getattr(cfg.run.clearml, 'execution', 'local')) != 'local'
def resolve_version_props(cfg: Any, *, clearml_enabled: Optional[bool]=None) -> dict[str, str]:
    clearml_enabled = is_clearml_enabled(cfg) if clearml_enabled is None else clearml_enabled
    return _resolve_version_props(cfg, clearml_enabled=bool(clearml_enabled))
def resolve_output_dir(cfg, stage: str) -> Path:
    base = Path(getattr(cfg.run, 'output_dir', 'outputs'))
    return base / stage
def update_registry_model_tags(*, model_id: str, add_tags: Iterable[str] | None=None, remove_prefixes: Iterable[str] | None=None) -> list[str]:
    from . import platform_adapter_model as _model
    return _model.update_registry_model_tags(model_id=model_id, add_tags=add_tags, remove_prefixes=remove_prefixes)
def ensure_clearml_task_tags(task_id: str, tags: Iterable[str]) -> bool:
    from . import platform_adapter_task_ops as _task_ops
    return _task_ops.ensure_clearml_task_tags(task_id, tags)
def write_manifest(ctx: TaskContext, manifest: dict[str, Any]) -> Path:
    """manifest.json を保存。

    本来は platform の manifest builder（P201〜P204）を使うべき。
    ここは Codex が platform 実装に合わせて置き換える。
    """
    try:
        from ml_platform.artifacts import write_manifest as platform_write_manifest
    except ImportError as exc:
        if ctx.task is not None:
            raise PlatformAdapterError('ml_platform.artifacts.write_manifest not available.') from exc
        platform_write_manifest = None
    if platform_write_manifest is not None:
        try:
            return platform_write_manifest(manifest, output_dir=ctx.output_dir, task=ctx.task, filename='manifest.json')
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to write manifest via ml_platform: {exc}') from exc
    path = ctx.output_dir / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return path
def register_dataset(cfg: Any, *, dataset_path: Path, dataset_name: str, dataset_project: str | None=None, dataset_tags: Optional[Iterable[str]]=None, dataset_version: str | None=None, description: str | None=None, parent_dataset_ids: Optional[Iterable[str]]=None, task_sections: Optional[Mapping[str, Mapping[str, Any]]]=None, task_section_order: Optional[Iterable[str]]=None) -> str:
    from .platform_adapter_dataset import register_dataset as _impl
    return _impl(cfg, dataset_path=dataset_path, dataset_name=dataset_name, dataset_project=dataset_project, dataset_tags=dataset_tags, dataset_version=dataset_version, description=description, parent_dataset_ids=parent_dataset_ids, task_sections=task_sections, task_section_order=task_section_order)
def get_dataset_local_copy(cfg: Any, dataset_id: str) -> Path:
    from .platform_adapter_dataset import get_dataset_local_copy as _impl
    return _impl(cfg, dataset_id)
def get_dataset_info(cfg: Any, dataset_id: str) -> dict[str, Any]:
    from .platform_adapter_dataset import get_dataset_info as _impl
    return _impl(cfg, dataset_id)
def _load_clearml_pipeline_utils(clearml_enabled: bool):
    try:
        from ml_platform.integrations.clearml import pipeline_utils as platform_pipeline_utils
    except ImportError as exc:
        if clearml_enabled:
            raise PlatformAdapterError('ml_platform.integrations.clearml.pipeline_utils not available for ClearML runs.') from exc
        return None
    return platform_pipeline_utils
def create_pipeline_controller(cfg: Any, *, name: str | None=None, tags: Iterable[str] | None=None, properties: Mapping[str, Any] | None=None, default_queue: str | None=None) -> Any:
    from .platform_adapter_pipeline import create_pipeline_controller as _impl
    return _impl(cfg, name=name, tags=tags, properties=properties, default_queue=default_queue)
def pipeline_require_clearml_agent(queue_name: str | None=None) -> None:
    from .platform_adapter_pipeline import pipeline_require_clearml_agent as _impl
    _impl(queue_name)
def pipeline_step_task_id_ref(step_name: str) -> str:
    from .platform_adapter_pipeline import pipeline_step_task_id_ref as _impl
    return _impl(step_name)
