from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
@dataclass
class TaskContext:
    """Execution context (ClearML Task and output directory)."""
    task: Any | None
    project_name: str
    task_name: str
    output_dir: Path
def _core():
    from . import platform_adapter_core as core
    return core
def _capture_env_snapshot(ctx: TaskContext) -> None:
    core = _core()
    try:
        from .ops.env_snapshot import capture_env_snapshot
    except ImportError as exc:
        raise core.PlatformAdapterError('tabular_analysis.ops.env_snapshot is not available.') from exc
    try:
        (env_path, freeze_path) = capture_env_snapshot(ctx.output_dir)
        upload_artifact(ctx, env_path.name, env_path)
        upload_artifact(ctx, freeze_path.name, freeze_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise core.PlatformAdapterError(f'Failed to capture env snapshot: {exc}') from exc
def init_task_context(cfg, *, stage: str, task_name: str, tags: Optional[list[str]]=None, properties: Optional[dict]=None, task_type: str | None=None, system_tags: Optional[Iterable[str]]=None) -> TaskContext:
    """Create task context via ml_platform ClearML integration."""
    core = _core()
    project_root = core._cfg_value(cfg, 'run.clearml.project_root') or 'MFG'
    usecase_id = core._cfg_value(cfg, 'run.usecase_id') or 'unknown'
    project_name = core._cfg_value(cfg, 'run.clearml.project_name') or core._cfg_value(cfg, 'task.project_name') or f'{project_root}/TabularAnalysis/{usecase_id}/{stage}'
    task_name_value = core._cfg_value(cfg, 'run.clearml.task_name') or task_name
    output_dir = core.resolve_output_dir(cfg, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .clearml.ui_logger import configure_reporting
        configure_reporting(cfg)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    if not core.is_clearml_enabled(cfg):
        ctx = TaskContext(task=None, project_name=str(project_name), task_name=str(task_name_value), output_dir=output_dir)
        _capture_env_snapshot(ctx)
        return ctx
    try:
        core._ensure_clearml_names(cfg, project_name=str(project_name), task_name=str(task_name_value), clearml_enabled=True)
        platform_clearml = core._load_clearml_module(clearml_enabled=True)
        task_factory = getattr(platform_clearml, 'task_factory', None)
        if task_factory is None:
            raise core.PlatformAdapterError('ml_platform.integrations.clearml.task_factory not found.')
        merged_props = core.build_clearml_properties(cfg, stage=stage, task_name=str(task_name_value), extra=properties, clearml_enabled=True)
        process = str(merged_props.get('process') or task_name_value or stage)
        merged_tags = core.build_clearml_tags(cfg, process=process, schema_version=str(merged_props.get('schema_version') or 'unknown'), grid_run_id=merged_props.get('grid_run_id'), retrain_run_id=merged_props.get('retrain_run_id'), extra_tags=core._cfg_value(cfg, 'run.clearml.extra_tags') or [], tags=tags)
        reuse_last_task_id = None
        if not os.getenv('CLEARML_TASK_ID') and (not os.getenv('TRAINS_TASK_ID')):
            reuse_last_task_id = False
        try:
            task = task_factory(cfg, tags=merged_tags, task_type=task_type, reuse_last_task_id=reuse_last_task_id)
        except TypeError:
            task = task_factory(cfg, tags=merged_tags, reuse_last_task_id=reuse_last_task_id)
        core._apply_clearml_task_type(task, task_type)
        core._apply_clearml_task_script_override(task, cfg)
        core._apply_clearml_system_tags(task, system_tags)
        setter = getattr(platform_clearml, 'set_user_properties', None)
        if setter is None:
            raise core.PlatformAdapterError('ml_platform.integrations.clearml.set_user_properties not found.')
        existing = core._existing_user_properties(task)
        merged_props = {**existing, **merged_props}
        setter(task, merged_props)
        ctx = TaskContext(task=task, project_name=str(project_name), task_name=str(task_name_value), output_dir=output_dir)
    except (core.PlatformAdapterError, ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise core.PlatformAdapterError(f'Failed to init ClearML task via ml-platform: {exc}') from exc
    _capture_env_snapshot(ctx)
    return ctx
def save_config_resolved(ctx: TaskContext, cfg) -> Path:
    """Save Hydra-resolved config, delegating to ml_platform when possible."""
    core = _core()
    if ctx.task is not None:
        try:
            from ml_platform.config import export_config_artifact
        except ImportError as exc:
            raise core.PlatformAdapterError('ml_platform.config.export_config_artifact not available for ClearML runs.') from exc
        try:
            return export_config_artifact(cfg, output_dir=ctx.output_dir, task=ctx.task, artifact_name='config_resolved.yaml')
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            raise core.PlatformAdapterError(f'Failed to write config_resolved.yaml via ml_platform: {exc}') from exc
    path = ctx.output_dir / 'config_resolved.yaml'
    try:
        from omegaconf import OmegaConf
        path.write_text(OmegaConf.to_yaml(cfg), encoding='utf-8')
    except (ImportError, AttributeError, TypeError, ValueError):
        path.write_text(str(cfg), encoding='utf-8')
    return path
def write_out_json(ctx: TaskContext, out: dict[str, Any]) -> Path:
    core = _core()
    path = ctx.output_dir / 'out.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    if ctx.task is not None:
        platform_clearml = core._load_clearml_module(clearml_enabled=True)
        uploader = getattr(platform_clearml, 'upload_artifact', None)
        if uploader is None:
            raise core.PlatformAdapterError('ml_platform.integrations.clearml.upload_artifact not found.')
        try:
            uploader(ctx.task, 'out.json', path)
        except core._RECOVERABLE_ERRORS as exc:
            raise core.PlatformAdapterError(f'Failed to upload out.json via ml_platform: {exc}') from exc
    return path
def upload_artifact(ctx: TaskContext, name: str, path: Path) -> None:
    """Upload an artifact to ClearML when enabled."""
    core = _core()
    if ctx.task is None:
        return
    platform_clearml = core._load_clearml_module(clearml_enabled=True)
    uploader = getattr(platform_clearml, 'upload_artifact', None)
    if uploader is None:
        raise core.PlatformAdapterError('ml_platform.integrations.clearml.upload_artifact not found.')
    try:
        uploader(ctx.task, name, path)
    except core._RECOVERABLE_ERRORS as exc:
        raise core.PlatformAdapterError(f'Failed to upload artifact {name} via ml_platform: {exc}') from exc
def update_task_properties(ctx: TaskContext, props: Mapping[str, Any]) -> None:
    """Merge and update task user properties (ClearML only)."""
    core = _core()
    if ctx.task is None:
        return
    platform_clearml = core._load_clearml_module(clearml_enabled=True)
    setter = getattr(platform_clearml, 'set_user_properties', None)
    if setter is None:
        raise core.PlatformAdapterError('ml_platform.integrations.clearml.set_user_properties not found.')
    existing = core._existing_user_properties(ctx.task)
    merged = {**existing, **dict(props)}
    try:
        setter(ctx.task, merged)
    except core._RECOVERABLE_ERRORS as exc:
        raise core.PlatformAdapterError(f'Failed to update user properties via ml_platform: {exc}') from exc
def connect_hyperparameters(ctx: TaskContext, hparams: Mapping[str, Any], *, name: str | None=None) -> None:
    """Connect minimal HyperParameters to ClearML task."""
    core = _core()
    if ctx.task is None:
        return
    connector = getattr(ctx.task, 'connect', None)
    if not callable(connector):
        raise core.PlatformAdapterError('ClearML Task.connect is not available.')
    payload = dict(hparams)
    try:
        if name:
            connector(payload, name=name)
        else:
            connector(payload)
    except core._RECOVERABLE_ERRORS as exc:
        raise core.PlatformAdapterError(f'Failed to connect HyperParameters via ClearML: {exc}') from exc
def connect_configuration(ctx: TaskContext, config: Mapping[str, Any], *, name: str='effective') -> None:
    """Attach a minimal configuration snapshot to ClearML task."""
    core = _core()
    if ctx.task is None:
        return
    connector = getattr(ctx.task, 'connect_configuration', None)
    if not callable(connector):
        raise core.PlatformAdapterError('ClearML Task.connect_configuration is not available.')
    try:
        connector(dict(config), name=name)
    except core._RECOVERABLE_ERRORS as exc:
        raise core.PlatformAdapterError(f'Failed to connect configuration via ClearML: {exc}') from exc
def report_markdown(ctx: TaskContext, *, title: str, markdown: str) -> bool:
    """Publish markdown text to ClearML when reporting is available."""
    core = _core()
    if ctx.task is None:
        return False
    if not markdown:
        return False
    getter = getattr(ctx.task, 'get_logger', None)
    if not callable(getter):
        return False
    try:
        logger = getter()
    except core._RECOVERABLE_ERRORS:
        return False
    if logger is None:
        return False
    reporter = getattr(logger, 'report_text', None)
    if not callable(reporter):
        return False
    try:
        payload = markdown.strip()
        if title:
            payload = f'# {title}\n\n{payload}'
        reporter(payload, print_console=False)
        return True
    except core._RECOVERABLE_ERRORS:
        return False
def add_task_tags(ctx: TaskContext, tags: Iterable[str]) -> None:
    """Add tags to a ClearML task when enabled."""
    core = _core()
    if ctx.task is None:
        return
    tag_list = core._dedupe_tags(tags)
    if not tag_list:
        return
    task = ctx.task
    adder = getattr(task, 'add_tags', None)
    if callable(adder):
        try:
            adder(tag_list)
            return
        except core._RECOVERABLE_ERRORS as exc:
            raise core.PlatformAdapterError(f'Failed to add task tags via ClearML: {exc}') from exc
    getter = getattr(task, 'get_tags', None)
    setter = getattr(task, 'set_tags', None)
    if callable(setter):
        existing: list[str] = []
        if callable(getter):
            try:
                current = getter() or []
                if isinstance(current, (str, bytes)):
                    existing = [str(current)]
                elif isinstance(current, Iterable):
                    existing = [str(item) for item in current]
            except core._RECOVERABLE_ERRORS:
                existing = []
        merged = core._dedupe_tags([*existing, *tag_list])
        try:
            setter(merged)
            return
        except core._RECOVERABLE_ERRORS as exc:
            raise core.PlatformAdapterError(f'Failed to set task tags via ClearML: {exc}') from exc
    raise core.PlatformAdapterError('ClearML task does not support tag updates.')
__all__ = ['TaskContext', 'init_task_context', 'save_config_resolved', 'write_out_json', 'upload_artifact', 'update_task_properties', 'report_markdown', 'connect_hyperparameters', 'connect_configuration', 'add_task_tags']
