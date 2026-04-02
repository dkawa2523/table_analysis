from __future__ import annotations
from pathlib import Path
from typing import Iterable, Mapping
from ..platform_adapter_artifacts import (
    get_dataset_info,
    get_dataset_local_copy,
    register_dataset,
)
def create_raw_dataset(
    cfg: object,
    *,
    dataset_path: Path,
    dataset_name: str,
    dataset_project: str | None = None,
    dataset_tags: Iterable[str] | None = None,
    dataset_version: str | None = None,
    description: str | None = None,
    task_sections: Mapping[str, Mapping[str, object]] | None = None,
    task_section_order: Iterable[str] | None = None,
) -> str:
    return register_dataset(
        cfg,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_tags=dataset_tags,
        dataset_version=dataset_version,
        description=description,
        task_sections=task_sections,
        task_section_order=task_section_order,
    )
def create_processed_dataset(
    cfg: object,
    *,
    dataset_dir: Path,
    dataset_name: str,
    dataset_project: str | None = None,
    dataset_tags: Iterable[str] | None = None,
    dataset_version: str | None = None,
    description: str | None = None,
    parent_dataset_ids: Iterable[str] | None = None,
    task_sections: Mapping[str, Mapping[str, object]] | None = None,
    task_section_order: Iterable[str] | None = None,
) -> str:
    return register_dataset(
        cfg,
        dataset_path=dataset_dir,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_tags=dataset_tags,
        dataset_version=dataset_version,
        description=description,
        parent_dataset_ids=parent_dataset_ids,
        task_sections=task_sections,
        task_section_order=task_section_order,
    )
def get_raw_dataset_local_copy(cfg: object, raw_dataset_id: str) -> Path:
    return get_dataset_local_copy(cfg, raw_dataset_id)
def get_processed_dataset_local_copy(cfg: object, processed_dataset_id: str) -> Path:
    return get_dataset_local_copy(cfg, processed_dataset_id)
def resolve_dataset_version(cfg: object, dataset_id: str) -> str | None:
    info = get_dataset_info(cfg, dataset_id)
    version = info.get("dataset_version")
    return str(version) if version is not None else None
