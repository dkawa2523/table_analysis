from __future__ import annotations
from collections.abc import Sequence as SequenceABC
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse
from .platform_adapter_core import PlatformAdapterError, _CLEARML_TASK_CACHE, _RECOVERABLE_ERRORS, _apply_clearml_files_host_substitution, _dedupe_tags, _existing_user_properties, _normalize_files_host, _normalize_requirement_lines, _resolve_clearml_task, is_clearml_enabled
def _get_clearml_task(task_id: str) -> Any:
    cached = _CLEARML_TASK_CACHE.get(task_id)
    if cached is not None:
        return cached
    try:
        from clearml import Task as ClearMLTask
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError('clearml is required for task artifact retrieval.') from exc
    try:
        task = ClearMLTask.get_task(task_id=str(task_id))
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to load ClearML task: {task_id}') from exc
    _CLEARML_TASK_CACHE[task_id] = task
    return task
def clearml_task_exists(task_id: str) -> bool:
    _get_clearml_task(task_id)
    return True
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
def list_clearml_tasks_by_tags(tags: Iterable[str], *, project_name: str | None=None, task_name: str | None=None, allow_archived: bool=True, order_by: Iterable[str] | None=None) -> list[Any]:
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise PlatformAdapterError('clearml is required for ClearML task queries.') from exc
    tag_list = _dedupe_tags(tags)
    if not tag_list:
        raise PlatformAdapterError('tags are required to query ClearML tasks.')
    task_filter = {'order_by': list(order_by) if order_by else ['-last_update']}
    try:
        tasks = ClearMLTask.get_tasks(project_name=project_name, task_name=task_name, tags=tag_list, allow_archived=allow_archived, task_filter=task_filter)
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        raise PlatformAdapterError(f'Failed to query ClearML tasks by tags: {exc}') from exc
    required = set(tag_list)
    filtered: list[Any] = []
    for task in list(tasks or []):
        task_tags = set(_task_tags(task))
        if required.issubset(task_tags):
            filtered.append(task)
    return filtered
def clearml_task_id(task: Any) -> str | None:
    task_id = getattr(task, 'id', None) or getattr(task, 'task_id', None)
    return str(task_id) if task_id else None
def clearml_task_tags(task: Any) -> list[str]:
    return _dedupe_tags(_task_tags(task))
def clearml_task_script(task: Any) -> dict[str, Any]:
    return _task_script(task)
def clearml_task_status_from_obj(task: Any) -> str | None:
    status = getattr(task, 'status', None)
    if status:
        return str(status)
    getter = getattr(task, 'get_status', None)
    if callable(getter):
        try:
            status = getter()
        except _RECOVERABLE_ERRORS:
            status = None
        if status:
            return str(status)
    return None
def clearml_task_type_from_obj(task: Any) -> str | None:
    for attr in ('task_type', 'type'):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    data = getattr(task, 'data', None)
    if isinstance(data, Mapping):
        value = data.get('task_type') or data.get('type')
        if value:
            return str(value)
    if data is not None:
        for attr in ('task_type', 'type'):
            value = getattr(data, attr, None)
            if value:
                return str(value)
    return None
def clearml_task_project_name(task: Any) -> str | None:
    getter = getattr(task, 'get_project_name', None)
    if callable(getter):
        try:
            value = getter()
        except _RECOVERABLE_ERRORS:
            value = None
        if value:
            return str(value)
    for attr in ('project_name', 'project'):
        value = getattr(task, attr, None)
        if value:
            text = str(value)
            if attr == 'project' and '/' not in text:
                continue
            return text
    data = getattr(task, 'data', None)
    if isinstance(data, Mapping):
        for key in ('project_name', 'project'):
            value = data.get(key)
            if value:
                text = str(value)
                if key == 'project' and '/' not in text:
                    continue
                return text
    if data is not None:
        for attr in ('project_name', 'project'):
            value = getattr(data, attr, None)
            if value:
                text = str(value)
                if attr == 'project' and '/' not in text:
                    continue
                return text
    return None
def find_clearml_task_id_by_tags(tags: Iterable[str], *, project_name: str | None=None, task_name: str | None=None, allow_archived: bool=True) -> str | None:
    """Resolve the most recently updated ClearML task id matching tags."""
    try:
        from clearml import Task as ClearMLTask
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError('clearml is required for ClearML task queries.') from exc
    tag_list = _dedupe_tags(tags)
    if not tag_list:
        raise PlatformAdapterError('tags are required to query ClearML tasks.')
    try:
        tasks = ClearMLTask.get_tasks(project_name=project_name, task_name=task_name, tags=tag_list, allow_archived=allow_archived, task_filter={'order_by': ['-last_update']})
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to query ClearML tasks by tags: {exc}') from exc
    if not tasks:
        return None
    task = tasks[0]
    task_id = getattr(task, 'id', None) or getattr(task, 'task_id', None)
    return str(task_id) if task_id else None
def _task_tags(task: Any) -> list[str]:
    getter = getattr(task, 'get_tags', None)
    if callable(getter):
        try:
            tags = getter()
        except _RECOVERABLE_ERRORS:
            tags = None
        if tags is not None:
            if isinstance(tags, (list, tuple, set)):
                return [str(tag) for tag in tags if tag is not None]
            return [str(tags)]
    tags = getattr(task, 'tags', None)
    if tags is not None:
        if isinstance(tags, (list, tuple, set)):
            return [str(tag) for tag in tags if tag is not None]
        return [str(tags)]
    data = getattr(task, 'data', None)
    if isinstance(data, Mapping):
        tags = data.get('tags')
        if isinstance(tags, (list, tuple, set)):
            return [str(tag) for tag in tags if tag is not None]
    if data is not None:
        tags = getattr(data, 'tags', None)
        if isinstance(tags, (list, tuple, set)):
            return [str(tag) for tag in tags if tag is not None]
    return []
def _task_script(task: Any) -> dict[str, Any]:
    script: dict[str, Any] = {}
    getter = getattr(task, 'get_script', None)
    if callable(getter):
        try:
            script_value = getter()
        except _RECOVERABLE_ERRORS:
            script_value = None
        if isinstance(script_value, Mapping):
            script.update(dict(script_value))
    data = getattr(task, 'data', None)
    script_obj = getattr(data, 'script', None) if data is not None else None
    if script_obj is not None:
        fallback = {'repository': getattr(script_obj, 'repository', None), 'branch': getattr(script_obj, 'branch', None), 'entry_point': getattr(script_obj, 'entry_point', None), 'working_dir': getattr(script_obj, 'working_dir', None), 'version_num': getattr(script_obj, 'version_num', None), 'diff': getattr(script_obj, 'diff', None)}
        for (key, value) in fallback.items():
            if key not in script or script[key] is None:
                script[key] = value
    return script
def _task_parameters(task: Any) -> dict[str, Any]:
    getter = getattr(task, 'get_parameters', None)
    if callable(getter):
        try:
            params = getter()
        except _RECOVERABLE_ERRORS:
            params = None
        if isinstance(params, Mapping):
            return dict(params)
    getter = getattr(task, 'get_parameters_as_dict', None)
    if callable(getter):
        try:
            params = getter()
        except _RECOVERABLE_ERRORS:
            params = None
        if isinstance(params, Mapping):
            flat: dict[str, Any] = {}
            for (section, values) in params.items():
                if not isinstance(values, Mapping):
                    continue
                for (key, value) in values.items():
                    flat[f'{section}/{key}'] = value
            return flat
    return {}


def _fully_unquote_text(text: str, *, max_rounds: int = 8) -> str:
    value = str(text).strip()
    for _ in range(max_rounds):
        collapsed = re.sub(r"%(?:%)+(?=[0-9A-Fa-f]{2})", "%", value)
        decoded = unquote(collapsed)
        if decoded == value:
            return decoded
        value = decoded
    return value


def _flatten_parameter_mapping(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for (key, item) in value.items():
            normalized_key = _normalize_parameter_key(str(key))
            next_prefix = f'{prefix}.{normalized_key}' if prefix else normalized_key
            _flatten_parameter_mapping(next_prefix, item, out)
        return
    if prefix:
        out[prefix] = value


def _task_parameter_sections(task: Any) -> dict[str, Any]:
    getter = getattr(task, 'get_parameters_as_dict', None)
    if callable(getter):
        try:
            params = getter(cast=False)
        except _RECOVERABLE_ERRORS:
            params = None
        if isinstance(params, Mapping):
            normalized = _normalize_parameter_payload(params)
            if isinstance(normalized, Mapping):
                return dict(normalized)
    params = _task_parameters(task)
    sections: dict[str, dict[str, Any]] = {}
    for (key, value) in params.items():
        if not isinstance(key, str) or '/' not in key:
            continue
        section, param_key = key.split('/', 1)
        section_name = _normalize_parameter_key(section)
        normalized_key = _normalize_parameter_key(param_key)
        sections.setdefault(section_name, {})[normalized_key] = value
    return sections


def _flatten_args_section(values: Mapping[str, Any]) -> dict[str, str]:
    flattened: dict[str, Any] = {}
    _flatten_parameter_mapping('', values, flattened)
    return {str(key): '' if value is None else str(value) for (key, value) in flattened.items() if key}
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
    existing_args = _flatten_args_section(sections.get('Args', {})) if isinstance(sections.get('Args'), Mapping) else {}
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
def get_clearml_task_tags(task_id: str) -> list[str]:
    task = _get_clearml_task(task_id)
    return _dedupe_tags(_task_tags(task))
def get_clearml_task_script(task_id: str) -> dict[str, Any]:
    task = _get_clearml_task(task_id)
    script = _task_script(task)
    return {'repository': script.get('repository'), 'branch': script.get('branch'), 'entry_point': script.get('entry_point'), 'working_dir': script.get('working_dir'), 'version_num': script.get('version_num')}
def get_clearml_task_args(task_id: str) -> dict[str, str]:
    task = _get_clearml_task(task_id)
    args_section = _task_parameter_sections(task).get('Args', {})
    if not isinstance(args_section, Mapping):
        return {}
    return _flatten_args_section(args_section)
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


def _normalize_parameter_key(text: str) -> str:
    return _fully_unquote_text(text)


def _normalize_parameter_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    normalized: dict[str, Any] = {}
    priorities: dict[str, int] = {}
    for (key, item) in value.items():
        raw_key = str(key)
        normalized_key = _normalize_parameter_key(raw_key)
        priority = 10 if "%" in raw_key else 20
        existing_priority = priorities.get(normalized_key, -1)
        if existing_priority > priority:
            continue
        normalized[normalized_key] = _normalize_parameter_payload(item)
        priorities[normalized_key] = priority
    return normalized


def _normalize_parameter_input_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_parameter_payload(
            {str(key): _normalize_parameter_input_value(item) for (key, item) in value.items()}
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_parameter_input_value(item) for item in value]
    return value


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
    getter = getattr(task, 'get_parameters_as_dict', None)
    existing = None
    if callable(getter):
        try:
            existing = getter(cast=False)
        except _RECOVERABLE_ERRORS:
            existing = None
    payload: dict[str, Any] = {}
    if isinstance(existing, Mapping):
        normalized_existing = _normalize_parameter_payload(existing)
        if isinstance(normalized_existing, Mapping):
            payload.update(normalized_existing)
        else:
            payload.update(existing)
    section_values: dict[str, Any] = {}
    if isinstance(payload.get(section), Mapping):
        section_values.update(dict(payload.get(section)))
    section_values.update(normalized)
    payload[section] = section_values
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        setter(payload)
        return True
    setter = getattr(task, 'set_parameter', None)
    if callable(setter):
        for (key, value) in section_values.items():
            setter(f'{section}/{key}', value)
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters_as_dict is not available.')


def replace_clearml_task_parameter_sections(task_id: str, sections: Mapping[str, Mapping[str, Any]]) -> bool:
    if not sections:
        return False
    task = _get_clearml_task(task_id)
    getter = getattr(task, 'get_parameters_as_dict', None)
    payload: dict[str, Any] = {}
    if callable(getter):
        try:
            existing = getter(cast=False)
        except _RECOVERABLE_ERRORS:
            existing = None
        if isinstance(existing, Mapping):
            normalized_existing = _normalize_parameter_payload(existing)
            if isinstance(normalized_existing, Mapping):
                payload.update(normalized_existing)
    changed = False
    for (section, values) in sections.items():
        section_name = str(section)
        normalized_section = _normalize_parameter_input_value(values)
        if payload.get(section_name) != normalized_section:
            changed = True
        payload[section_name] = normalized_section
    if not changed:
        return False
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        setter(payload)
        return True
    raise PlatformAdapterError('ClearML Task.set_parameters_as_dict is not available.')
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
def get_clearml_task_configuration(task_id: str, *, name: str='effective') -> Any | None:
    task = _get_clearml_task(task_id)
    getter = getattr(task, 'get_configuration_object_as_dict', None)
    if callable(getter):
        try:
            return getter(str(name))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to read task configuration {name!r}: {exc}') from exc
    getter = getattr(task, 'get_configuration_object', None)
    if callable(getter):
        try:
            value = getter(str(name))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to read task configuration {name!r}: {exc}') from exc
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
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
def get_clearml_task_status(task_id: str) -> str | None:
    task = _get_clearml_task(task_id)
    status = getattr(task, 'status', None)
    if status:
        return str(status)
    getter = getattr(task, 'get_status', None)
    if callable(getter):
        try:
            status = getter()
        except _RECOVERABLE_ERRORS:
            status = None
        if status:
            return str(status)
    return None
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
    desired = _parse_task_args(args)
    if not desired:
        return False
    task = _get_clearml_task(task_id)
    sections = _task_parameter_sections(task)
    existing_args = _flatten_args_section(sections.get('Args', {})) if isinstance(sections.get('Args'), Mapping) else {}
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
    desired = _parse_task_args(args)
    task = _get_clearml_task(task_id)
    normalized = {str(key): '' if value is None else str(value) for (key, value) in desired.items()}
    sections = _task_parameter_sections(task)
    setter = getattr(task, 'set_parameters_as_dict', None)
    if callable(setter):
        payload = dict(sections)
        payload['Args'] = dict(normalized)
        setter(payload)
        return True
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
def _resolve_task_artifact(task: Any, artifact_name: str) -> Any | None:
    artifacts = getattr(task, 'artifacts', None)
    if isinstance(artifacts, Mapping):
        if artifact_name in artifacts:
            return artifacts[artifact_name]
    elif artifacts is not None and (not isinstance(artifacts, (str, bytes))):
        try:
            for item in artifacts:
                if isinstance(item, Mapping):
                    key = item.get('key') or item.get('name')
                    if key == artifact_name:
                        return item
                else:
                    key = getattr(item, 'key', None) or getattr(item, 'name', None)
                    if key == artifact_name:
                        return item
        except _RECOVERABLE_ERRORS:
            pass
    getter = getattr(task, 'get_artifact', None)
    if callable(getter):
        try:
            return getter(artifact_name)
        except _RECOVERABLE_ERRORS:
            return None
    return None
def _artifact_local_copy(artifact: Any) -> str | None:
    if artifact is None:
        return None
    if isinstance(artifact, Path):
        return str(artifact)
    if isinstance(artifact, str):
        return artifact
    getter = getattr(artifact, 'get_local_copy', None)
    if callable(getter):
        try:
            return getter()
        except _RECOVERABLE_ERRORS:
            return None
    if isinstance(artifact, Mapping):
        for key in ('local_copy', 'local_path', 'path', 'artifact_local_path'):
            value = artifact.get(key)
            if value:
                return str(value)
    return None
def get_task_artifact_local_copy(cfg: Any, task_id: str, artifact_name: str) -> Path:
    if not is_clearml_enabled(cfg):
        raise PlatformAdapterError('ClearML is disabled; cannot fetch task artifacts.')
    _apply_clearml_files_host_substitution()
    task = _get_clearml_task(task_id)
    artifact = _resolve_task_artifact(task, artifact_name)
    local_path = _artifact_local_copy(artifact)
    if not local_path:
        uri = None
        if isinstance(artifact, Mapping):
            uri = artifact.get('uri') or artifact.get('url')
        else:
            uri = getattr(artifact, 'uri', None) or getattr(artifact, 'url', None)
        if uri:
            try:
                from clearml.backend_api import Session
            except ImportError:
                Session = None
            try:
                import requests
            except ImportError:
                requests = None
            if Session is not None and requests is not None:
                try:
                    session = Session()
                    files_host = os.getenv('CLEARML_FILES_HOST') or session.config.get('api.files_server')
                    if files_host:
                        normalized = _normalize_files_host(files_host)
                        if normalized:
                            parsed = urlparse(uri)
                            if parsed.hostname in {'host.docker.internal', 'clearml-fileserver'}:
                                uri = uri.replace(f'{parsed.scheme}://{parsed.netloc}', normalized)
                    creds = session.config.get('api.credentials')
                    token_resp = session.send_request(service='auth', action='login', json={'access_key': creds['access_key'], 'secret_key': creds['secret_key']})
                    token = token_resp.json()['data']['token']
                    headers = {'Authorization': f'Bearer {token}'}
                    response = requests.get(uri, headers=headers, timeout=30)
                    response.raise_for_status()
                    target_dir = Path('/tmp/clearml_artifacts') / task_id
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / artifact_name
                    target_path.write_bytes(response.content)
                    local_path = str(target_path)
                except _RECOVERABLE_ERRORS:
                    local_path = None
        if not local_path:
            raise PlatformAdapterError(f'Artifact {artifact_name} not found on ClearML task {task_id}.')
    path = Path(local_path)
    if not path.exists():
        raise PlatformAdapterError(f'Artifact {artifact_name} local copy does not exist: {path}')
    return path
def resolve_clearml_task_url(cfg: Any, task_id: str) -> str | None:
    """Resolve a ClearML task URL when possible; returns None when unavailable."""
    if not is_clearml_enabled(cfg) or not task_id:
        return None
    try:
        task = _get_clearml_task(task_id)
    except PlatformAdapterError:
        return None
    for getter_name in ('get_output_log_web_page', 'get_task_output_log_web_page'):
        getter = getattr(task, getter_name, None)
        if callable(getter):
            try:
                url = getter()
            except (RuntimeError, TypeError, ValueError, AttributeError):
                url = None
            if url:
                return str(url)
    url = getattr(task, 'output_log_web_page', None)
    if url:
        return str(url)
    return None
__all__ = ['clearml_task_exists', 'create_clearml_task', 'list_clearml_tasks_by_tags', 'clearml_task_id', 'clearml_task_tags', 'clearml_task_script', 'clearml_task_status_from_obj', 'clearml_task_type_from_obj', 'clearml_task_project_name', 'find_clearml_task_id_by_tags', 'get_clearml_task_tags', 'get_clearml_task_script', 'get_clearml_task_args', 'clone_clearml_task', 'set_clearml_task_entry_point', 'set_clearml_task_parameters', 'replace_clearml_task_parameter_sections', 'set_clearml_task_configuration', 'get_clearml_task_configuration', 'set_clearml_task_project', 'upload_clearml_task_artifact', 'set_clearml_task_runtime_properties', 'enqueue_clearml_task', 'reset_clearml_task', 'get_clearml_task_status', 'ensure_clearml_task_tags', 'replace_clearml_task_tags', 'update_clearml_task_tags', 'ensure_clearml_task_requirements', 'ensure_clearml_task_properties', 'ensure_clearml_task_args', 'reset_clearml_task_args', 'apply_clearml_task_overrides', 'ensure_clearml_task_script', 'get_task_artifact_local_copy', 'resolve_clearml_task_url']
