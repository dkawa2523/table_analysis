from __future__ import annotations
import os
import sys
from typing import Any, Iterable, Mapping
from .platform_adapter_task_ops import apply_clearml_task_overrides
from .platform_adapter_core import PlatformAdapterError, _apply_clearml_pipeline_args, _apply_clearml_system_tags, _apply_clearml_tags, _apply_clearml_task_requirements, _apply_clearml_task_script_override, _apply_clearml_task_type, _cfg_value, _ensure_clearml_project_system_tags, _existing_user_properties, _load_clearml_module, _load_clearml_pipeline_utils, _resolve_clearml_pipeline_requirements, _resolve_clearml_task, clearml_task_type_controller
def create_pipeline_controller(cfg: Any, *, name: str | None=None, tags: Iterable[str] | None=None, properties: Mapping[str, Any] | None=None, default_queue: str | None=None) -> Any:
    pipeline_utils = _load_clearml_pipeline_utils(clearml_enabled=True)
    if pipeline_utils is None:
        raise PlatformAdapterError('pipeline_utils is not available.')
    project_mode = str(_cfg_value(cfg, 'run.clearml.pipeline.project_mode', 'subproject')).lower()
    tag_pipeline_project = bool(_cfg_value(cfg, 'run.clearml.pipeline.project_tag_pipeline', project_mode == 'visible'))
    unhide_pipeline_project = bool(_cfg_value(cfg, 'run.clearml.pipeline.project_unhide', project_mode == 'visible'))
    controller_project = _cfg_value(cfg, 'run.clearml.pipeline.project_name')
    if not controller_project:
        controller_project = _cfg_value(cfg, 'run.clearml.project_name')
    if project_mode == 'visible':
        try:
            from clearml.automation import PipelineController
        except ImportError:
            PipelineController = None
        if PipelineController is not None:
            PipelineController._pipeline_as_sub_project_cached = False
    elif project_mode == 'subproject':
        try:
            from clearml.automation import PipelineController
        except ImportError:
            PipelineController = None
        if PipelineController is not None:
            PipelineController._pipeline_as_sub_project_cached = True
    tag_list: list[str] = []
    if tags:
        tag_list = [str(tag) for tag in tags if tag]
    if 'pipeline' not in tag_list:
        tag_list.append('pipeline')
    controller = pipeline_utils.create_controller(cfg, name=name, project=str(controller_project) if controller_project else None, tags=tag_list, default_queue=default_queue)
    try:
        if hasattr(controller, '_target_project'):
            controller._target_project = False
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    task = _resolve_clearml_task(controller)
    _apply_clearml_task_type(task, clearml_task_type_controller())
    _apply_clearml_system_tags(task, ['pipeline'])
    if tag_list:
        _apply_clearml_tags(task, tag_list)
    execution_project = _cfg_value(cfg, 'run.clearml.project_name') or controller_project
    if execution_project:
        mover = getattr(task, 'move_to_project', None)
        if callable(mover):
            try:
                current_project = None
                getter = getattr(task, 'get_project_name', None)
                if callable(getter):
                    current_project = getter()
                if not current_project:
                    current_project = getattr(task, 'project', None)
                if str(current_project or '') != str(execution_project):
                    mover(new_project_name=str(execution_project))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    if tag_pipeline_project or execution_project:
        if execution_project:
            project_name = execution_project
        elif project_mode == 'subproject':
            project_name = getattr(task, 'project', None) or controller_project
        else:
            project_name = controller_project or getattr(task, 'project', None)
        remove_tags = ['hidden'] if unhide_pipeline_project else []
        _ensure_clearml_project_system_tags(project_name, ['pipeline'], remove_tags=remove_tags)
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
__all__ = ['apply_clearml_task_overrides', 'create_pipeline_controller', 'pipeline_require_clearml_agent', 'pipeline_step_task_id_ref']
