from __future__ import annotations
from collections.abc import Sequence as SequenceABC
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from .platform_adapter_artifacts import get_task_artifact_local_copy as _get_task_artifact_local_copy_impl, resolve_clearml_task_url as _resolve_clearml_task_url_impl
from .platform_adapter_common import PlatformAdapterError, _CLEARML_TASK_CACHE, _RECOVERABLE_ERRORS, _dedupe_tags, _existing_user_properties, _normalize_requirement_lines, _resolve_clearml_task, canonicalize_clearml_args_payload, canonicalize_clearml_section_payload, fully_unquote_text
from .platform_adapter_task_query import (
    _get_clearml_task,
    _task_parameters,
    _task_script,
    _task_tags,
    clearml_task_exists,
    clearml_task_id,
    clearml_task_project_name,
    clearml_task_script,
    clearml_task_status_from_obj,
    clearml_task_tags,
    clearml_task_type_from_obj,
    find_clearml_task_id_by_tags,
    get_clearml_task_configuration,
    get_clearml_task_script,
    get_clearml_task_status,
    get_clearml_task_tags,
    list_clearml_tasks_by_tags,
)
def create_clearml_task(*, project_name: str, task_name: str, module: str | None=None, script: str | None=None, args: Iterable[str] | None=None, repo: str | None=None, branch: str | None=None, working_dir: str | None=None, task_type: str | None=None, tags: Iterable[str] | None=None, properties: Mapping[str, Any] | None=None, requirements: Iterable[str] | None=None) -> str:
    if module and script:
        raise PlatformAdapterError('Specify either module or script for ClearML task creation.')
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise PlatformAdapterError('clearml is required to create ClearML tasks.') from exc
    try:
        task = ClearMLTask.create(project_name=str(project_name), task_name=str(task_name), task_type=task_type, repo=repo, branch=branch, script=script, working_directory=working_dir, module=module, argparse_args=list(args) if args else None)
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        raise PlatformAdapterError(f'Failed to create ClearML task: {exc}') from exc
    if tags:
        tag_list = _dedupe_tags(tags)
        if tag_list:
            try:
                task.add_tags(tag_list)
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                raise PlatformAdapterError(f'Failed to set task tags via ClearML: {exc}') from exc
    if properties:
        setter = getattr(task, 'set_user_properties', None)
        if callable(setter):
            normalized = {str(key): '' if value is None else str(value) for (key, value) in properties.items()}
            try:
                setter(*normalized.items())
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                raise PlatformAdapterError(f'Failed to set task properties via ClearML: {exc}') from exc
    if requirements:
        setter = getattr(task, 'set_packages', None)
        if not callable(setter):
            raise PlatformAdapterError('ClearML Task.set_packages is not available.')
        normalized = _normalize_requirement_lines(requirements)
        if normalized:
            try:
                setter(normalized)
            except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                raise PlatformAdapterError(f'Failed to set task requirements via ClearML: {exc}') from exc
    task_id = getattr(task, 'id', None)
    if not task_id:
        raise PlatformAdapterError('ClearML task id is missing after creation.')
    return str(task_id)

def _task_parameter_sections(task: Any) -> dict[str, Any]:
    getter = getattr(task, 'get_parameters_as_dict', None)
    if callable(getter):
        try:
            params = getter(cast=False)
        except _RECOVERABLE_ERRORS:
            params = None
        if isinstance(params, Mapping):
            normalized: dict[str, Any] = {}
            for (section, values) in params.items():
                section_name = fully_unquote_text(str(section))
                if section_name == 'Args':
                    normalized[section_name] = canonicalize_clearml_args_payload(values if isinstance(values, Mapping) else {})
                else:
                    normalized[section_name] = canonicalize_clearml_section_payload(values if isinstance(values, Mapping) else {})
            return normalized
    params = _task_parameters(task)
    raw_sections: dict[str, dict[str, Any]] = {}
    for (key, value) in params.items():
        if not isinstance(key, str) or '/' not in key:
            continue
        section, param_key = key.split('/', 1)
        section_name = fully_unquote_text(section)
        raw_sections.setdefault(section_name, {})[param_key] = value
    sections: dict[str, Any] = {}
    for (section_name, values) in raw_sections.items():
        if section_name == 'Args':
            sections[section_name] = canonicalize_clearml_args_payload(values)
        else:
            sections[section_name] = canonicalize_clearml_section_payload(values)
    return sections
def _property_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if 'value' in value:
            return value.get('value')
    return value
def _parse_task_args(args: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in args:
        text = str(item).strip()
        if not text:
            continue
        if '=' not in text:
            raise PlatformAdapterError(f'override must be key=value: {text}')
        (key, value) = text.split('=', 1)
        parsed[str(key)] = str(value)
    return parsed


def _set_task_script_payload(task: Any, payload: Mapping[str, Any]) -> None:
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


def _apply_task_args(task: Any, args: Mapping[str, Any]) -> bool:
    if not args:
        return False
    sections = _task_parameter_sections(task)
    existing_args = dict(sections.get('Args', {})) if isinstance(sections.get('Args'), Mapping) else {}
    updates: dict[str, str] = {}
    for (key, value) in args.items():
        expected = '' if value is None else str(value)
        if existing_args.get(str(key)) != expected:
            updates[str(key)] = expected
    if not updates:
        return False
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        payload = dict(sections)
        payload['Args'] = {**existing_args, **updates}
        setter(payload)
        return True
    params = _task_parameters(task)
    updated_params = dict(params)
    for (key, value) in updates.items():
        updated_params[f'Args/{key}'] = value
    setter = getattr(task, 'set_parameters', None)
    if callable(setter):
        setter(updated_params)
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters is not available.')
def get_clearml_task_args(task_id: str) -> dict[str, str]:
    task = _get_clearml_task(task_id)
    args_section = _task_parameter_sections(task).get('Args', {})
    if not isinstance(args_section, Mapping):
        return {}
    return {str(key): '' if value is None else str(value) for (key, value) in dict(args_section).items()}
def clone_clearml_task(*, source_task_id: str | None=None, source_task: Any | None=None, task_name: str | None=None, parent_task_id: str | None=None) -> str:
    if not source_task_id and source_task is None:
        raise PlatformAdapterError('source_task_id or source_task is required to clone a task.')
    try:
        from clearml import Task as ClearMLTask
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError('clearml is required to clone ClearML tasks.') from exc
    source = source_task if source_task is not None else str(source_task_id)
    try:
        cloned = ClearMLTask.clone(source_task=source, name=task_name, parent=parent_task_id)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to clone ClearML task: {exc}') from exc
    task_id = getattr(cloned, 'id', None) or getattr(cloned, 'task_id', None)
    if not task_id:
        raise PlatformAdapterError('Cloned ClearML task id is missing.')
    _CLEARML_TASK_CACHE[str(task_id)] = cloned
    return str(task_id)
def set_clearml_task_entry_point(task_id: str, entry_point: str) -> None:
    task = _get_clearml_task(task_id)
    script = _task_script(task)
    payload: dict[str, Any] = {}
    for key in ('repository', 'branch', 'working_dir', 'version_num'):
        value = script.get(key)
        if value is not None:
            payload[key] = value
    payload['entry_point'] = entry_point
    _set_task_script_payload(task, payload)

def _normalize_parameter_input_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_parameter_input_value(item) for (key, item) in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_parameter_input_value(item) for item in value]
    return value


def _flatten_parameter_payload(section: str, payload: Any, out: dict[str, Any], *, prefix: str = '') -> None:
    if isinstance(payload, Mapping):
        for (key, value) in payload.items():
            text = str(key)
            next_prefix = f'{prefix}/{text}' if prefix else text
            _flatten_parameter_payload(section, value, out, prefix=next_prefix)
        return
    if not prefix:
        return
    out[f'{section}/{prefix}'] = payload


def _flatten_hyperparameter_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for (section, value) in payload.items():
        section_name = str(section)
        if section_name == 'Args' and isinstance(value, Mapping):
            for (key, item) in value.items():
                flattened[f'Args/{key}'] = item
            continue
        _flatten_parameter_payload(section_name, value, flattened)
    return flattened


def _delete_task_parameters(task: Any, keys: Iterable[str]) -> None:
    deleter = getattr(task, 'delete_parameter', None)
    if not callable(deleter):
        return
    for key in keys:
        try:
            deleter(str(key))
        except _RECOVERABLE_ERRORS:
            continue


def _replace_task_hyperparameter_payload(task: Any, desired_payload: Mapping[str, Any]) -> bool:
    current_sections = _task_parameter_sections(task)
    if current_sections == dict(desired_payload):
        return False
    current_flat = _task_parameters(task)
    desired_flat = _flatten_hyperparameter_payload(desired_payload)
    tracked_sections = {str(key) for key in current_sections.keys()} | {str(key) for key in desired_payload.keys()}
    stale_keys = [
        key
        for key in current_flat.keys()
        if isinstance(key, str)
        and '/' in key
        and key.split('/', 1)[0] in tracked_sections
        and key not in desired_flat
    ]
    if stale_keys:
        _delete_task_parameters(task, stale_keys)
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        setter(dict(desired_payload))
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters_as_dict is not available.')


def set_clearml_task_parameters(task_id: str, parameters: Mapping[str, Any], *, section: str='Args') -> bool:
    if not parameters:
        return False
    task = _get_clearml_task(task_id)
    json_keys = {'infer.input_json', 'infer.batch.inputs_json', 'infer.validation.inputs_json', 'infer.optimize.search_space'}
    normalized: dict[str, Any] = {}
    for (key, value) in parameters.items():
        key_text = str(key)
        if value is None:
            normalized[key_text] = ''
            continue
        if key_text in json_keys and isinstance(value, (Mapping, SequenceABC)) and (not isinstance(value, (str, bytes))):
            try:
                normalized[key_text] = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                continue
            except _RECOVERABLE_ERRORS:
                pass
        normalized[key_text] = _normalize_parameter_input_value(value)
    payload = _task_parameter_sections(task)
    section_values: Any = payload.get(section)
    if section == 'Args':
        merged_args = dict(section_values) if isinstance(section_values, Mapping) else {}
        merged_args.update(normalized)
        payload[section] = canonicalize_clearml_args_payload(merged_args)
    else:
        merged_section = dict(section_values) if isinstance(section_values, Mapping) else {}
        merged_section.update(normalized)
        payload[section] = canonicalize_clearml_section_payload(merged_section)
    updated_section = payload[section]
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        setter(payload)
        return True
    setter = getattr(task, 'set_parameter', None)
    if callable(setter):
        for (key, value) in updated_section.items():
            setter(f'{section}/{key}', value)
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters_as_dict is not available.')


def replace_clearml_task_parameter_sections(task_id: str, sections: Mapping[str, Mapping[str, Any]]) -> bool:
    if not sections:
        return False
    task = _get_clearml_task(task_id)
    payload = _task_parameter_sections(task)
    changed = False
    for (section, values) in sections.items():
        section_name = str(section)
        normalized_section = canonicalize_clearml_section_payload(_normalize_parameter_input_value(values))
        if payload.get(section_name) != normalized_section:
            changed = True
        payload[section_name] = normalized_section
    if not changed:
        return False
    return _replace_task_hyperparameter_payload(task, payload)


def _desired_hyperparameter_payload(
    *,
    args: Iterable[str] | None = None,
    sections: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    desired_payload: dict[str, Any] = {}
    if args is not None:
        normalized_args = canonicalize_clearml_args_payload(_parse_task_args(args))
        if normalized_args:
            desired_payload['Args'] = normalized_args
    for (section, values) in dict(sections or {}).items():
        section_name = str(section)
        if not section_name or section_name == 'Args':
            continue
        desired_payload[section_name] = canonicalize_clearml_section_payload(_normalize_parameter_input_value(values))
    return desired_payload


def replace_clearml_task_object_hyperparameters(task: Any, *, args: Iterable[str] | None=None, sections: Mapping[str, Mapping[str, Any]] | None=None) -> bool:
    desired_payload = _desired_hyperparameter_payload(args=args, sections=sections)
    if not desired_payload:
        return False
    return _replace_task_hyperparameter_payload(task, desired_payload)


def replace_clearml_task_hyperparameters(task_id: str, *, args: Iterable[str] | None=None, sections: Mapping[str, Mapping[str, Any]] | None=None) -> bool:
    task = _get_clearml_task(task_id)
    return replace_clearml_task_object_hyperparameters(task, args=args, sections=sections)
def set_clearml_task_configuration(task_id: str, config: Mapping[str, Any], *, name: str='effective', description: str | None=None) -> bool:
    if not config:
        return False
    task = _get_clearml_task(task_id)
    setter = getattr(task, 'set_configuration_object', None)
    if callable(setter):
        try:
            setter(
                str(name),
                description=description,
                config_type='dictionary',
                config_dict=dict(config),
            )
            return True
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to set task configuration {name!r}: {exc}') from exc
    connector = getattr(task, 'connect_configuration', None)
    if callable(connector):
        try:
            connector(dict(config), name=str(name))
            return True
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to connect task configuration {name!r}: {exc}') from exc
    raise PlatformAdapterError('ClearML task configuration API is not available.')
def set_clearml_task_project(task_id: str, project_name: str) -> bool:
    task = _get_clearml_task(task_id)
    actual_project = clearml_task_project_name(task)
    if str(actual_project or '') == str(project_name):
        return False
    mover = getattr(task, 'move_to_project', None)
    if callable(mover):
        try:
            mover(new_project_name=str(project_name))
            return True
        except _RECOVERABLE_ERRORS:
            pass
    setter = getattr(task, 'set_project', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_project is not available.')
    try:
        setter(project_name=str(project_name))
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to move task to project {project_name!r}: {exc}') from exc
    return True
def upload_clearml_task_artifact(task_id: str, name: str, path: Path) -> bool:
    task = _get_clearml_task(task_id)
    uploader = getattr(task, 'upload_artifact', None)
    if not callable(uploader):
        raise PlatformAdapterError('ClearML Task.upload_artifact is not available.')
    try:
        uploader(str(name), artifact_object=Path(path))
    except TypeError:
        try:
            uploader(str(name), Path(path))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to upload artifact {name!r}: {exc}') from exc
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to upload artifact {name!r}: {exc}') from exc
    return True


def set_clearml_task_runtime_properties(task_id: str, properties: Mapping[str, Any]) -> bool:
    if not properties:
        return False
    task = _get_clearml_task(task_id)
    setter = getattr(task, '_set_runtime_properties', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task runtime properties API is not available.')
    try:
        setter(dict(properties))
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to set runtime properties: {exc}') from exc
    return True


def enqueue_clearml_task(task_id: str, queue_name: str, *, force: bool=False) -> None:
    if not queue_name:
        raise PlatformAdapterError('queue_name is required to enqueue a ClearML task.')
    try:
        from clearml import Task as ClearMLTask
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError('clearml is required to enqueue ClearML tasks.') from exc
    try:
        ClearMLTask.enqueue(task=str(task_id), queue_name=str(queue_name), force=bool(force))
    except TypeError:
        try:
            ClearMLTask.enqueue(task_id=str(task_id), queue_name=str(queue_name), force=bool(force))
        except TypeError:
            ClearMLTask.enqueue(str(task_id), queue_name=str(queue_name))
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to enqueue ClearML task: {exc}') from exc
def reset_clearml_task(task_id: str, *, force: bool=False, set_started_on_success: bool=False) -> bool:
    task = _get_clearml_task(task_id)
    resetter = getattr(task, 'reset', None)
    if not callable(resetter):
        raise PlatformAdapterError('ClearML Task.reset is not available.')
    try:
        resetter(
            set_started_on_success=bool(set_started_on_success),
            force=bool(force),
        )
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to reset ClearML task: {exc}') from exc
    return True
def ensure_clearml_task_tags(task_id: str, tags: Iterable[str]) -> bool:
    desired = _dedupe_tags(tags)
    if not desired:
        return False
    task = _get_clearml_task(task_id)
    existing = set(_task_tags(task))
    missing = [tag for tag in desired if tag not in existing]
    if not missing:
        return False
    adder = getattr(task, 'add_tags', None)
    if not callable(adder):
        raise PlatformAdapterError('ClearML Task.add_tags is not available.')
    adder(missing)
    return True


def replace_clearml_task_tags(task_id: str, tags: Iterable[str]) -> bool:
    normalized = _dedupe_tags(tags)
    task = _get_clearml_task(task_id)
    existing = _task_tags(task)
    if existing == normalized:
        return False
    setter = getattr(task, 'set_tags', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_tags is not available.')
    setter(normalized)
    return True


def update_clearml_task_tags(task_id: str, *, add: Iterable[str] | None=None, remove: Iterable[str] | None=None) -> bool:
    add_list = _dedupe_tags(add or [])
    remove_set = set(_dedupe_tags(remove or []))
    if not add_list and (not remove_set):
        return False
    task = _get_clearml_task(task_id)
    existing = _task_tags(task)
    updated = [tag for tag in existing if tag not in remove_set]
    updated = _dedupe_tags([*updated, *add_list])
    if updated == existing:
        return False
    setter = getattr(task, 'set_tags', None)
    if callable(setter):
        setter(updated)
        return True
    if remove_set:
        raise PlatformAdapterError('ClearML Task.set_tags is not available for tag removal.')
    adder = getattr(task, 'add_tags', None)
    if callable(adder):
        adder([tag for tag in add_list if tag not in existing])
        return True
    raise PlatformAdapterError('ClearML Task tag update is not available.')
def ensure_clearml_task_requirements(task_id: str, requirements: Iterable[str]) -> bool:
    desired = _normalize_requirement_lines(requirements)
    if not desired:
        return False
    task = _get_clearml_task(task_id)
    getter = getattr(task, 'get_requirements', None)
    if not callable(getter):
        raise PlatformAdapterError('ClearML Task.get_requirements is not available.')
    try:
        existing = getter()
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to read task requirements via ClearML: {exc}') from exc
    existing_pip = existing.get('pip') if isinstance(existing, Mapping) else None
    if _normalize_requirement_lines(existing_pip) == desired:
        return False
    setter = getattr(task, 'set_packages', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_packages is not available.')
    try:
        setter(desired)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to set task requirements via ClearML: {exc}') from exc
    return True
def ensure_clearml_task_properties(task_id: str, properties: Mapping[str, Any]) -> bool:
    if not properties:
        return False
    task = _get_clearml_task(task_id)
    existing = _existing_user_properties(task)
    updates: dict[str, Any] = {}
    for (key, value) in properties.items():
        expected = '' if value is None else str(value)
        current = _property_value(existing.get(str(key)))
        if current is None or str(current) != expected:
            updates[str(key)] = expected
    if not updates:
        return False
    setter = getattr(task, 'set_user_properties', None)
    if not callable(setter):
        raise PlatformAdapterError('ClearML Task.set_user_properties is not available.')
    setter(*updates.items())
    return True
def ensure_clearml_task_args(task_id: str, args: Iterable[str]) -> bool:
    desired = canonicalize_clearml_args_payload(_parse_task_args(args))
    if not desired:
        return False
    task = _get_clearml_task(task_id)
    sections = _task_parameter_sections(task)
    existing_args = dict(sections.get('Args', {})) if isinstance(sections.get('Args'), Mapping) else {}
    updates: dict[str, str] = {}
    for (key, value) in desired.items():
        if existing_args.get(key) != value:
            updates[key] = value
    if not updates:
        return False
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        payload = dict(sections)
        payload['Args'] = {**existing_args, **updates}
        setter(payload)
        return True
    params = _task_parameters(task)
    updated_params = dict(params)
    for (key, value) in updates.items():
        updated_params[f'Args/{key}'] = value
    setter = getattr(task, 'set_parameters', None)
    if callable(setter):
        setter(updated_params)
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters is not available.')
def reset_clearml_task_args(task_id: str, args: Iterable[str]) -> bool:
    desired = canonicalize_clearml_args_payload(_parse_task_args(args))
    task = _get_clearml_task(task_id)
    normalized = {str(key): '' if value is None else str(value) for (key, value) in desired.items()}
    sections = _task_parameter_sections(task)
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        payload = dict(sections)
        if normalized:
            payload['Args'] = dict(normalized)
        else:
            payload.pop('Args', None)
        return _replace_task_hyperparameter_payload(task, payload)
    params = _task_parameters(task)
    updated = {k: v for (k, v) in params.items() if not (isinstance(k, str) and k.startswith('Args/'))}
    for (key, value) in normalized.items():
        updated[f'Args/{key}'] = value
    setter = getattr(task, 'set_parameters', None)
    if callable(setter):
        setter(updated)
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters is not available.')
def apply_clearml_task_overrides(target: Any, overrides: Iterable[str]) -> bool:
    desired = _parse_task_args(overrides)
    if not desired:
        return False
    task = _resolve_clearml_task(target)
    return _apply_task_args(task, desired)
def ensure_clearml_task_script(task_id: str, *, repo: str | None, branch: str | None, entry_point: str | None, working_dir: str | None, version_num: str | None=None, diff: str | None=None) -> bool:
    if repo is None and branch is None and (entry_point is None) and (working_dir is None) and (version_num is None) and (diff is None):
        return False
    task = _get_clearml_task(task_id)
    current = _task_script(task)
    changed = False
    if repo is not None and str(current.get('repository') or '') != str(repo):
        changed = True
    if branch is not None and str(current.get('branch') or '') != str(branch):
        changed = True
    if entry_point is not None and str(current.get('entry_point') or '') != str(entry_point):
        changed = True
    if working_dir is not None and str(current.get('working_dir') or '') != str(working_dir):
        changed = True
    if version_num is not None and str(current.get('version_num') or '') != str(version_num):
        changed = True
    if diff is not None and str(current.get('diff') or '') != str(diff):
        changed = True
    if not changed:
        return False
    payload: dict[str, Any] = {'repository': repo, 'branch': branch, 'working_dir': working_dir, 'entry_point': entry_point}
    if version_num is not None:
        payload['version_num'] = version_num
    if diff is not None:
        payload['diff'] = diff
    _set_task_script_payload(task, payload)
    return True
def get_task_artifact_local_copy(cfg: Any, task_id: str, artifact_name: str) -> Path:
    return _get_task_artifact_local_copy_impl(cfg, task_id, artifact_name)
def resolve_clearml_task_url(cfg: Any, task_id: str) -> str | None:
    return _resolve_clearml_task_url_impl(cfg, task_id)
__all__ = ['clearml_task_exists', 'create_clearml_task', 'list_clearml_tasks_by_tags', 'clearml_task_id', 'clearml_task_tags', 'clearml_task_script', 'clearml_task_status_from_obj', 'clearml_task_type_from_obj', 'clearml_task_project_name', 'find_clearml_task_id_by_tags', 'get_clearml_task_tags', 'get_clearml_task_script', 'get_clearml_task_args', 'clone_clearml_task', 'set_clearml_task_entry_point', 'set_clearml_task_parameters', 'replace_clearml_task_parameter_sections', 'replace_clearml_task_object_hyperparameters', 'replace_clearml_task_hyperparameters', 'set_clearml_task_configuration', 'get_clearml_task_configuration', 'set_clearml_task_project', 'upload_clearml_task_artifact', 'set_clearml_task_runtime_properties', 'enqueue_clearml_task', 'reset_clearml_task', 'get_clearml_task_status', 'ensure_clearml_task_tags', 'replace_clearml_task_tags', 'update_clearml_task_tags', 'ensure_clearml_task_requirements', 'ensure_clearml_task_properties', 'ensure_clearml_task_args', 'reset_clearml_task_args', 'apply_clearml_task_overrides', 'ensure_clearml_task_script', 'get_task_artifact_local_copy', 'resolve_clearml_task_url']
