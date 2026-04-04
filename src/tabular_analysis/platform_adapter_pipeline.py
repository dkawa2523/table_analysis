from __future__ import annotations
import os
import sys
from typing import Any, Iterable, Mapping
from .platform_adapter_task_ops import apply_clearml_task_overrides
from .platform_adapter_core import PlatformAdapterError, _apply_clearml_pipeline_args, _apply_clearml_system_tags, _apply_clearml_tags, _apply_clearml_task_requirements, _apply_clearml_task_script_override, _apply_clearml_task_type, _cfg_value, _ensure_clearml_project_system_tags, _existing_user_properties, _load_clearml_module, _load_clearml_pipeline_utils, _resolve_clearml_pipeline_requirements, _resolve_clearml_task, clearml_task_type_controller
def _load_visible_pipeline_controller_class() -> Any:
    try:
        from clearml import PipelineController
    except ImportError:
        try:
            from clearml.automation.controller import PipelineController
        except ImportError as exc:
            raise PlatformAdapterError('ClearML PipelineController is not available.') from exc
    PipelineController._pipeline_as_sub_project_cached = False
    return PipelineController


def create_pipeline_controller(cfg: Any, *, name: str | None=None, tags: Iterable[str] | None=None, properties: Mapping[str, Any] | None=None, default_queue: str | None=None) -> Any:
    pipeline_utils = _load_clearml_pipeline_utils(clearml_enabled=True)
    if pipeline_utils is None:
        raise PlatformAdapterError('pipeline_utils is not available.')
    _load_visible_pipeline_controller_class()
    controller_project = _cfg_value(cfg, 'run.clearml.pipeline.project_name')
    if not controller_project:
        controller_project = _cfg_value(cfg, 'run.clearml.project_name')
    tag_list: list[str] = []
    if tags:
        tag_list = [str(tag) for tag in tags if tag]
    if 'pipeline' not in tag_list:
        tag_list.append('pipeline')
    controller = pipeline_utils.create_controller(cfg, name=name, project=str(controller_project) if controller_project else None, tags=tag_list, default_queue=default_queue)
    task = _resolve_clearml_task(controller)
    _apply_clearml_task_type(task, clearml_task_type_controller())
    _apply_clearml_system_tags(task, ['pipeline'])
    if tag_list:
        _apply_clearml_tags(task, tag_list)
    project_name = controller_project or _cfg_value(cfg, 'run.clearml.project_name') or getattr(task, 'project', None)
    _ensure_clearml_project_system_tags(project_name, ['pipeline'], remove_tags=['hidden'])
    if properties:
        platform_clearml = _load_clearml_module(clearml_enabled=True)
        setter = getattr(platform_clearml, 'set_user_properties', None)
        if setter is None:
            raise PlatformAdapterError('ml_platform.integrations.clearml.set_user_properties not found.')
        existing = _existing_user_properties(task)
        merged = {**existing, **dict(properties)}
        setter(task, merged)
    _apply_clearml_task_script_override(controller, cfg)
    _apply_clearml_pipeline_args(controller, cfg)
    _apply_clearml_task_requirements(_resolve_clearml_task(controller), _resolve_clearml_pipeline_requirements(cfg))
    return controller


def load_pipeline_controller_from_task(*, source_task_id: str | None=None, source_task: Any | None=None) -> Any:
    if not source_task_id and source_task is None:
        raise PlatformAdapterError('source_task_id or source_task is required to load a pipeline controller.')
    PipelineController = _load_visible_pipeline_controller_class()
    task = source_task
    if task is None:
        try:
            from clearml import Task as ClearMLTask
        except ImportError as exc:
            raise PlatformAdapterError('clearml is required to load a pipeline controller task.') from exc
        try:
            task = ClearMLTask.get_task(task_id=str(source_task_id))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise PlatformAdapterError(f'Failed to load pipeline controller task: {source_task_id}') from exc
    loader = getattr(PipelineController, '_create_pipeline_controller_from_task', None)
    if not callable(loader):
        raise PlatformAdapterError('PipelineController._create_pipeline_controller_from_task is not available.')
    return loader(task)


def create_pipeline_draft_controller(*, project_name: str, task_name: str, module: str | None=None, script: str | None=None, args: Iterable[tuple[str, str]] | None=None, repo: str | None=None, branch: str | None=None, commit: str | None=None, working_dir: str | None=None) -> Any:
    PipelineController = _load_visible_pipeline_controller_class()
    kwargs: dict[str, Any] = {
        'project_name': str(project_name),
        'task_name': str(task_name),
        'repo': repo,
        'branch': branch,
        'commit': commit,
        'working_directory': working_dir,
        'argparse_args': list(args or []),
        'add_run_number': False,
        'detect_repository': repo is None,
    }
    if module:
        kwargs['module'] = module
    else:
        kwargs['script'] = script
    try:
        controller = PipelineController.create(**kwargs)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PlatformAdapterError(f'Failed to create visible pipeline draft: {exc}') from exc
    task = _resolve_clearml_task(controller)
    _apply_clearml_task_type(task, clearml_task_type_controller())
    _apply_clearml_system_tags(task, ['pipeline'])
    _ensure_clearml_project_system_tags(project_name, ['pipeline'], remove_tags=['hidden'])
    return controller


def clone_pipeline_controller(*, source_task_id: str, task_name: str | None=None, project_name: str | None=None, parent_task_id: str | None=None) -> Any:
    PipelineController = _load_visible_pipeline_controller_class()
    try:
        controller = PipelineController.clone(
            str(source_task_id),
            name=str(task_name) if task_name else None,
            parent=str(parent_task_id) if parent_task_id else None,
            project=str(project_name) if project_name else None,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PlatformAdapterError(f'Failed to clone pipeline controller: {exc}') from exc
    task = _resolve_clearml_task(controller)
    if project_name:
        _ensure_clearml_project_system_tags(project_name, ['pipeline'], remove_tags=['hidden'])
        setter = getattr(task, 'set_project', None)
        if callable(setter):
            setter(project_name=str(project_name))
    _apply_clearml_task_type(task, clearml_task_type_controller())
    _apply_clearml_system_tags(task, ['pipeline'])
    return controller


def enqueue_pipeline_controller(target: Any, queue_name: str, *, force: bool=False) -> Any:
    if not queue_name:
        raise PlatformAdapterError('queue_name is required to enqueue a pipeline controller.')
    PipelineController = _load_visible_pipeline_controller_class()
    try:
        return PipelineController.enqueue(target, queue_name=queue_name, force=bool(force))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise PlatformAdapterError(f'Failed to enqueue pipeline controller: {exc}') from exc


def pipeline_require_clearml_agent(queue_name: str | None=None) -> None:
    if os.getenv('CLEARML_TASK_ID') or os.getenv('TRAINS_TASK_ID'):
        return
    pipeline_utils = _load_clearml_pipeline_utils(clearml_enabled=True)
    if pipeline_utils is None:
        raise PlatformAdapterError('pipeline_utils is not available.')
    try:
        pipeline_utils.require_clearml_agent(queue_name)
    except RuntimeError as exc:
        print(f'[warn] {exc}', file=sys.stderr)
def pipeline_step_task_id_ref(step_name: str) -> str:
    pipeline_utils = _load_clearml_pipeline_utils(clearml_enabled=True)
    if pipeline_utils is None:
        raise PlatformAdapterError('pipeline_utils is not available.')
    return pipeline_utils.step_task_id_ref(step_name)
__all__ = ['apply_clearml_task_overrides', 'create_pipeline_controller', 'create_pipeline_draft_controller', 'clone_pipeline_controller', 'enqueue_pipeline_controller', 'load_pipeline_controller_from_task', 'pipeline_require_clearml_agent', 'pipeline_step_task_id_ref']
