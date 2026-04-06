from __future__ import annotations
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional
from .platform_adapter_clearml_env import load_clearml_dataset, is_clearml_enabled
from .platform_adapter_common import (
    PlatformAdapterError,
    RECOVERABLE_CLEARML_ERRORS,
    apply_clearml_files_host_substitution,
)
def _connect_dataset_task_sections(dataset: Any, sections: Mapping[str, Mapping[str, Any]] | None, order: Iterable[str] | None) -> None:
    if not sections:
        return
    task = getattr(dataset, '_task', None)
    if task is None:
        return
    connector = getattr(task, 'connect', None)
    if not callable(connector):
        return
    def _connect(name: str, payload: Mapping[str, Any]) -> None:
        cleaned = {key: value for (key, value) in payload.items() if value is not None}
        if not cleaned:
            return
        try:
            connector(dict(cleaned), name=name)
        except RECOVERABLE_CLEARML_ERRORS as exc:
            print(f'[warn] Failed to connect dataset HyperParameters ({name}): {exc}', file=sys.stderr)
    seen: set[str] = set()
    if order:
        for name in order:
            payload = sections.get(name)
            if not payload:
                continue
            _connect(name, payload)
            seen.add(name)
    for (name, payload) in sections.items():
        if name in seen:
            continue
        _connect(name, payload)
def register_dataset(cfg: Any, *, dataset_path: Path, dataset_name: str, dataset_project: str | None=None, dataset_tags: Optional[Iterable[str]]=None, dataset_version: str | None=None, description: str | None=None, parent_dataset_ids: Optional[Iterable[str]]=None, task_sections: Optional[Mapping[str, Mapping[str, Any]]]=None, task_section_order: Optional[Iterable[str]]=None) -> str:
    if not is_clearml_enabled(cfg):
        raise PlatformAdapterError('ClearML is disabled; cannot register dataset.')
    ClearMLDataset = load_clearml_dataset(clearml_enabled=True)
    try:
        parents = [str(parent) for parent in parent_dataset_ids or [] if parent]
        dataset = ClearMLDataset.create(dataset_name=dataset_name, dataset_project=dataset_project, dataset_tags=list(dataset_tags) if dataset_tags else None, dataset_version=dataset_version, description=description, parent_datasets=parents or None)
        _connect_dataset_task_sections(dataset, task_sections, task_section_order)
        dataset.add_files(path=str(dataset_path))
        dataset.upload()
        dataset.finalize()
        return str(dataset.id)
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to register dataset via ClearML: {exc}') from exc
def get_dataset_local_copy(cfg: Any, dataset_id: str) -> Path:
    if not is_clearml_enabled(cfg):
        raise PlatformAdapterError('ClearML is disabled; cannot fetch dataset.')
    ClearMLDataset = load_clearml_dataset(clearml_enabled=True)
    apply_clearml_files_host_substitution()
    try:
        dataset = ClearMLDataset.get(dataset_id=str(dataset_id))
        local_path = dataset.get_local_copy()
    except RECOVERABLE_CLEARML_ERRORS:
        apply_clearml_files_host_substitution()
        try:
            dataset = ClearMLDataset.get(dataset_id=str(dataset_id))
            local_path = dataset.get_local_copy()
        except RECOVERABLE_CLEARML_ERRORS as exc_retry:
            raise PlatformAdapterError(f'Failed to fetch dataset via ClearML: {exc_retry}') from exc_retry
    if not local_path:
        raise PlatformAdapterError('ClearML Dataset.get_local_copy returned an empty path.')
    return Path(local_path)
def get_dataset_info(cfg: Any, dataset_id: str) -> dict[str, Any]:
    if not is_clearml_enabled(cfg):
        raise PlatformAdapterError('ClearML is disabled; cannot fetch dataset info.')
    ClearMLDataset = load_clearml_dataset(clearml_enabled=True)
    try:
        dataset = ClearMLDataset.get(dataset_id=str(dataset_id))
    except RECOVERABLE_CLEARML_ERRORS as exc:
        raise PlatformAdapterError(f'Failed to fetch dataset info via ClearML: {exc}') from exc
    info: dict[str, Any] = {'dataset_id': str(getattr(dataset, 'id', dataset_id))}
    version = getattr(dataset, 'version', None)
    if version is None:
        version = getattr(dataset, 'dataset_version', None)
    if version is not None:
        info['dataset_version'] = str(version)
    name = getattr(dataset, 'name', None)
    if name:
        info['dataset_name'] = str(name)
    project = getattr(dataset, 'project', None)
    if project:
        info['dataset_project'] = str(project)
    return info
__all__ = ['register_dataset', 'get_dataset_local_copy', 'get_dataset_info']
