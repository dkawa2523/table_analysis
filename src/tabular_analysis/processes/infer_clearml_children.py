from __future__ import annotations

"""ClearML-backed child-task executor for infer batch/optimize flows.

Keep this module as the only place that knows how infer orchestration clones,
sanitizes, and enqueues ClearML child tasks so it can be replaced later
without rewriting infer core runtime code.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..common.model_reference import build_infer_reference
from ..ops.clearml_identity import build_project_name
from ..platform_adapter_common import PlatformAdapterError
from ..platform_adapter_task_ops import (
    clone_clearml_task,
    enqueue_clearml_task,
    get_clearml_task_status,
    replace_clearml_task_hyperparameters,
    set_clearml_task_entry_point,
    set_clearml_task_project,
    update_clearml_task_tags,
)
from .pipeline_support import DEFAULT_PIPELINE_CHILD_QUEUE as _DEFAULT_PIPELINE_CHILD_QUEUE


@dataclass(frozen=True)
class ClearMLInferChildContext:
    queue_name: str
    child_project_name: str | None
    source_task_id: str
    child_model_id: str | None
    child_train_task_id: str | None


def _serialize_override_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _mapping_to_cli_args(overrides: Mapping[str, Any]) -> list[str]:
    return [f"{str(key)}={_serialize_override_value(value)}" for (key, value) in overrides.items()]


def resolve_clearml_child_queue(cfg: Any) -> str:
    queue = _normalize_str(_cfg_value(cfg, "exec_policy.queues.infer"))
    if not queue:
        queue = _normalize_str(_cfg_value(cfg, "exec_policy.queues.default"))
    return queue or _DEFAULT_PIPELINE_CHILD_QUEUE


def build_clearml_child_context(
    cfg: Any,
    ctx: Any,
    *,
    infer_cfg: Any,
    meta: Mapping[str, Any],
    context_name: str,
) -> ClearMLInferChildContext:
    queue_name = resolve_clearml_child_queue(cfg)
    child_project_name = None
    project_root = _normalize_str(_cfg_value(cfg, "run.clearml.project_root"))
    usecase_id = _normalize_str(_cfg_value(cfg, "run.usecase_id"))
    if project_root and usecase_id:
        child_project_name = build_project_name(
            project_root,
            usecase_id,
            stage="infer",
            process="infer_child",
            cfg=cfg,
        )
    source_task_id = _normalize_str(_cfg_value(cfg, "run.clearml.clone_from_task_id")) or str(
        getattr(ctx.task, "id", "")
    )
    if not source_task_id:
        raise PlatformAdapterError(f"ClearML task id is required to clone {context_name} child tasks.")
    explicit_model_id = _normalize_str(getattr(infer_cfg, "model_id", None))
    child_reference = build_infer_reference(
        model_id=explicit_model_id or _normalize_str(meta.get("model_id")),
        registry_model_id=None if explicit_model_id else _normalize_str(meta.get("registry_model_id")),
        train_task_id=_normalize_str(getattr(infer_cfg, "train_task_id", None))
        or _normalize_str(meta.get("train_task_id")),
    )
    child_model_id = child_reference.get("infer_model_id")
    child_train_task_id = child_reference.get("infer_train_task_id")
    if not child_model_id and not child_train_task_id:
        raise ValueError(f"infer.model_id or infer.train_task_id is required for {context_name} child tasks.")
    return ClearMLInferChildContext(
        queue_name=queue_name,
        child_project_name=child_project_name,
        source_task_id=source_task_id,
        child_model_id=child_model_id,
        child_train_task_id=child_train_task_id,
    )


def clone_and_enqueue_clearml_infer_child(
    *,
    source_task_id: str,
    child_name: str,
    queue_name: str,
    overrides: Mapping[str, Any],
    child_project_name: str | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    child_task_id = clone_clearml_task(source_task_id=source_task_id, task_name=child_name)
    try:
        set_clearml_task_entry_point(child_task_id, "tools/clearml_entrypoint.py")
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    if child_project_name:
        set_clearml_task_project(child_task_id, child_project_name)
    if tags:
        update_clearml_task_tags(child_task_id, add=[str(tag) for tag in tags if str(tag).strip()])
    replace_clearml_task_hyperparameters(child_task_id, args=_mapping_to_cli_args(overrides))
    enqueue_clearml_task(child_task_id, queue_name)
    return child_task_id


def wait_for_clearml_child_tasks(
    task_ids: Sequence[str],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    remaining = set(task_ids)
    deadline = time.monotonic() + timeout_sec
    terminal = {"completed", "failed", "stopped", "closed", "aborted"}
    while remaining and time.monotonic() < deadline:
        for task_id in list(remaining):
            status = get_clearml_task_status(task_id)
            if status:
                status_value = str(status).lower()
                if status_value in terminal:
                    statuses[task_id] = status_value
                    remaining.remove(task_id)
        if remaining:
            time.sleep(poll_interval_sec)
    for task_id in remaining:
        statuses[task_id] = "timeout"
    return statuses


__all__ = [
    "ClearMLInferChildContext",
    "build_clearml_child_context",
    "clone_and_enqueue_clearml_infer_child",
    "resolve_clearml_child_queue",
    "wait_for_clearml_child_tasks",
]
