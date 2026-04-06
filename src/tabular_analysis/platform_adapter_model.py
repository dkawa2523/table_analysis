from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from .platform_adapter_artifacts import get_task_artifact_local_copy
from .platform_adapter_common import PlatformAdapterError, RECOVERABLE_CLEARML_ERRORS, dedupe_tags
from .platform_adapter_task_context import TaskContext
@dataclass(frozen=True)
class ResolvedModelReference:
    kind: str
    value: str
    model_bundle_path: Path
    model_id: str | None
    train_task_id: str | None
    registry_model_id: str | None
    local_path: Path | None
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
def _resolve_model_bundle_candidate(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        candidate = candidate / 'model_bundle.joblib'
    if not candidate.exists():
        raise PlatformAdapterError(f'model_bundle.joblib not found: {candidate}')
    return candidate.resolve()
def register_model_artifact(ctx: TaskContext, *, model_path: Path, model_name: str, tags: Iterable[str] | None=None, metadata: Mapping[str, Any] | None=None, comment: str | None=None, framework: str | None=None) -> str:
    """Register a model artifact in ClearML model registry and tag it."""
    if ctx.task is None:
        raise PlatformAdapterError('ClearML is disabled; cannot register model.')
    try:
        from clearml import OutputModel
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.OutputModel is required for model promotion.') from exc
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise PlatformAdapterError(f'Model file does not exist: {model_path}')
    try:
        output_model = OutputModel(task=ctx.task, name=str(model_name), tags=dedupe_tags(tags or []), comment=comment, framework=framework)
        output_model.update_weights(weights_filename=str(model_path), auto_delete_file=False, async_enable=False)
        if metadata:
            for (key, value) in metadata.items():
                if value is None:
                    continue
                if isinstance(value, (dict, list, tuple)):
                    text = json.dumps(value, ensure_ascii=True)
                else:
                    text = str(value)
                if not text:
                    continue
                output_model.set_metadata(str(key), text)
        return str(output_model.id)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to register model via ClearML: {exc}') from exc
def register_promoted_model(ctx: TaskContext, *, model_path: Path, model_name: str, tags: Iterable[str] | None=None, metadata: Mapping[str, Any] | None=None, comment: str | None=None, framework: str | None=None) -> str:
    """Register a promoted model artifact in ClearML model registry and tag it."""
    return register_model_artifact(ctx, model_path=model_path, model_name=model_name, tags=tags, metadata=metadata, comment=comment, framework=framework)
def get_clearml_model_local_copy(model_id: str) -> Path:
    """Download a ClearML registry model artifact and return the local path."""
    try:
        from clearml import Model
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.Model is required for registry model downloads.') from exc
    try:
        model = Model(model_id=str(model_id))
        local_path = model.get_local_copy(raise_on_error=True)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to download ClearML model {model_id}: {exc}') from exc
    if not local_path:
        raise PlatformAdapterError(f'ClearML model {model_id} did not return a local copy path.')
    path = Path(local_path)
    if not path.exists():
        raise PlatformAdapterError(f'ClearML model local copy does not exist: {path}')
    return path.resolve()
def resolve_model_reference(*, cfg: Any | None=None, model_id: str | None=None, train_task_id: str | None=None) -> ResolvedModelReference:
    model_id_value = _normalize_optional_str(model_id)
    train_task_id_value = _normalize_optional_str(train_task_id)
    if train_task_id_value:
        try:
            artifact_path = get_task_artifact_local_copy(cfg, train_task_id_value, 'model_bundle.joblib')
        except (PlatformAdapterError, AttributeError, ImportError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, FloatingPointError, json.JSONDecodeError) as exc:
            raise PlatformAdapterError(f'Failed to fetch model_bundle.joblib from task {train_task_id_value}: {exc}') from exc
        path = _resolve_model_bundle_candidate(artifact_path)
        return ResolvedModelReference(kind='train_task_artifact', value=train_task_id_value, model_bundle_path=path, model_id=model_id_value, train_task_id=train_task_id_value, registry_model_id=None, local_path=path, source='train_task_artifact', metadata={'artifact_name': 'model_bundle.joblib'})
    if not model_id_value:
        raise PlatformAdapterError('infer.model_id or infer.train_task_id is required.')
    try:
        path = _resolve_model_bundle_candidate(model_id_value)
        return ResolvedModelReference(kind='local_bundle', value=model_id_value, model_bundle_path=path, model_id=model_id_value, train_task_id=None, registry_model_id=None, local_path=path, source='local_path', metadata={})
    except PlatformAdapterError:
        pass
    local_copy = get_clearml_model_local_copy(model_id_value)
    path = _resolve_model_bundle_candidate(local_copy)
    return ResolvedModelReference(kind='registry_model', value=model_id_value, model_bundle_path=path, model_id=model_id_value, train_task_id=None, registry_model_id=model_id_value, local_path=path, source='registry_model', metadata={'local_copy_path': str(local_copy)})
def resolve_registry_model_bundle_by_stage(*, stage: str, usecase_id: str | None=None) -> tuple[str, Path]:
    """Resolve a ClearML registry model bundle by stage tags."""
    try:
        from clearml import Model
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.Model is required for registry queries.') from exc
    tags = ['__$all', f'stage:{stage}']
    if usecase_id:
        tags.append(f'usecase:{usecase_id}')
    try:
        models = Model.query_models(tags=tags, only_published=False, include_archived=True, max_results=1)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to query ClearML registry for stage {stage}: {exc}') from exc
    if not models:
        suffix = f' usecase={usecase_id}' if usecase_id else ''
        raise PlatformAdapterError(f'No ClearML registry model found for stage={stage}.{suffix}')
    model = models[0]
    model_id = getattr(model, 'id', None) or getattr(model, 'model_id', None)
    model_id_str = str(model_id) if model_id is not None else 'unknown'
    try:
        local_path = model.get_local_copy(raise_on_error=True)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to download ClearML model {model_id_str}: {exc}') from exc
    if not local_path:
        raise PlatformAdapterError(f'ClearML model {model_id_str} did not return a local copy path.')
    path = Path(local_path)
    if not path.exists():
        raise PlatformAdapterError(f'ClearML model local copy does not exist: {path}')
    return (model_id_str, path.resolve())
def _model_tags(model: Any) -> list[str]:
    tags = getattr(model, 'tags', None)
    if tags is None:
        return []
    if isinstance(tags, (str, bytes)):
        return [str(tags)]
    try:
        return [str(item) for item in tags]
    except RECOVERABLE_CLEARML_ERRORS:
        return []
def _set_model_tags(model: Any, tags: Iterable[str]) -> None:
    try:
        model.tags = list(tags)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to update model tags via ClearML: {exc}') from exc
def _strip_tag_prefixes(tags: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    prefix_list = [str(prefix) for prefix in prefixes if prefix is not None]
    if not prefix_list:
        return [str(tag) for tag in tags if tag is not None]
    cleaned: list[str] = []
    for tag in tags:
        if tag is None:
            continue
        text = str(tag)
        if any((text.startswith(prefix) for prefix in prefix_list)):
            continue
        cleaned.append(text)
    return cleaned
def update_registry_model_tags(*, model_id: str, add_tags: Iterable[str] | None=None, remove_prefixes: Iterable[str] | None=None) -> list[str]:
    """Update tags on a ClearML registry model and return the final tags."""
    try:
        from clearml import Model
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.Model is required for registry tag updates.') from exc
    model = Model(model_id=str(model_id))
    tags = _model_tags(model)
    if remove_prefixes:
        tags = _strip_tag_prefixes(tags, remove_prefixes)
    if add_tags:
        tags = dedupe_tags([*tags, *list(add_tags)])
    _set_model_tags(model, tags)
    return tags
def update_recommended_registry_model_tags(*, usecase_id: str, processed_dataset_id: str, recommended_model_id: str, tags_to_add: Iterable[str], remove_prefixes: Iterable[str]) -> dict[str, Any]:
    """Mark the recommended model and clear old recommendation tags for the same dataset."""
    try:
        from clearml import Model
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.Model is required for registry queries.') from exc
    base_tags = ['__$all', f'usecase:{usecase_id}', f'dataset:{processed_dataset_id}']
    try:
        models = Model.query_models(tags=base_tags, only_published=False, include_archived=True, max_results=200)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to query ClearML registry for dataset={processed_dataset_id}: {exc}') from exc
    updated = 0
    for model in models:
        model_id = getattr(model, 'id', None) or getattr(model, 'model_id', None)
        model_id_str = str(model_id) if model_id is not None else ''
        tags = _strip_tag_prefixes(_model_tags(model), remove_prefixes)
        if model_id_str == recommended_model_id:
            tags = dedupe_tags([*tags, *list(tags_to_add)])
        _set_model_tags(model, tags)
        updated += 1
    return {'updated_models': updated, 'matched_models': len(models)}
def update_recommended_registry_model_tags_multi(*, usecase_id: str, processed_dataset_id: str, split_hash: str | None=None, recipe_hash: str | None=None, recommendations: Iterable[tuple[str, Iterable[str], Mapping[str, Any] | None]], remove_prefixes: Iterable[str]) -> dict[str, Any]:
    """Update recommendation tags for multiple models (latest-only per dataset)."""
    try:
        from clearml import Model
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.Model is required for registry queries.') from exc
    rec_map = {str(model_id): {'tags': list(tags), 'metadata': metadata or {}} for (model_id, tags, metadata) in recommendations if model_id}
    base_tags = ['__$all', f'usecase:{usecase_id}', f'dataset:{processed_dataset_id}']
    if split_hash:
        base_tags.append(f'split:{split_hash}')
    if recipe_hash:
        base_tags.append(f'recipe:{recipe_hash}')
    try:
        models = Model.query_models(tags=base_tags, only_published=False, include_archived=True, max_results=200)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to query ClearML registry for dataset={processed_dataset_id}: {exc}') from exc
    updated = 0
    for model in models:
        model_id = getattr(model, 'id', None) or getattr(model, 'model_id', None)
        model_id_str = str(model_id) if model_id is not None else ''
        tags = _strip_tag_prefixes(_model_tags(model), remove_prefixes)
        if model_id_str in rec_map:
            payload = rec_map[model_id_str]
            tags = dedupe_tags([*tags, *payload.get('tags', [])])
            metadata = payload.get('metadata') or {}
            if metadata:
                _update_model_metadata(model, metadata)
        _set_model_tags(model, tags)
        updated += 1
    return {'updated_models': updated, 'matched_models': len(models), 'recommended_models': len(rec_map)}
def _update_model_metadata(model: Any, metadata: Mapping[str, Any]) -> None:
    setter = getattr(model, 'set_metadata', None)
    if not callable(setter):
        return
    for (key, value) in metadata.items():
        if value is None:
            continue
        try:
            setter(str(key), str(value))
        except RECOVERABLE_CLEARML_ERRORS as exc:
            raise PlatformAdapterError(f'Failed to update model metadata via ClearML: {exc}') from exc
def _model_snapshot(model: Any | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {'registry_model_id': getattr(model, 'id', None), 'name': getattr(model, 'name', None), 'tags': _model_tags(model)}
def rollback_registry_stage(*, usecase_id: str, stage: str, target_model_id: str | None=None, reason: str | None=None) -> dict[str, Any]:
    """Rollback ClearML model registry stage to previous production (best-effort)."""
    try:
        from clearml import Model
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError('clearml.Model is required for model registry rollback.') from exc
    tags = ['__$all', f'usecase:{usecase_id}', f'stage:{stage}']
    models = Model.query_models(tags=tags, only_published=False, include_archived=True, max_results=20)
    current = models[0] if models else None
    target = None
    if target_model_id:
        target = Model(model_id=str(target_model_id))
    elif len(models) > 1:
        target = models[1]
    if target is None:
        raise PlatformAdapterError('No previous model found for rollback.')
    target_tags = dedupe_tags([*_model_tags(target), f'usecase:{usecase_id}', f'stage:{stage}', 'rollback:true'])
    _set_model_tags(target, target_tags)
    _update_model_metadata(target, {'promotion_stage': stage, 'rollback_reason': reason, 'rollback_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
    if current is not None and getattr(current, 'id', None) != getattr(target, 'id', None):
        current_tags = [tag for tag in _model_tags(current) if tag != f'stage:{stage}']
        _set_model_tags(current, dedupe_tags(current_tags))
    return {'before': _model_snapshot(current), 'after': _model_snapshot(target)}
__all__ = ['ResolvedModelReference', 'register_model_artifact', 'register_promoted_model', 'get_clearml_model_local_copy', 'resolve_model_reference', 'resolve_registry_model_bundle_by_stage', 'update_registry_model_tags', 'update_recommended_registry_model_tags', 'update_recommended_registry_model_tags_multi', 'rollback_registry_stage']
