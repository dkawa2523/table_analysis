from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, set_cfg_value as _set_cfg_value, normalize_str as _normalize_str
import re
from pathlib import Path
from typing import Any, Iterable
def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized.strip("-_") or "unknown"
def _extract_dataset_token(value: Any) -> str | None:
    text = _normalize_str(value)
    if not text:
        return None
    if text.startswith("local:"):
        text = text.split(":", 1)[1] or "local"
    if "/" in text or "\\" in text:
        name = Path(text).name
        if name:
            text = Path(name).stem or name
    return _sanitize_identifier(text)
def _shorten_token(token: str, max_len: int = 12) -> str:
    if max_len <= 0:
        return token
    if len(token) <= max_len:
        return token
    if max_len < 6:
        return token[:max_len]
    tail = 3
    head = max_len - tail - 1
    return f"{token[:head]}~{token[-tail:]}"
def _resolve_model_abbr(cfg: Any) -> str:
    value = _normalize_str(_cfg_value(cfg, "model_variant.name"))
    if value is None:
        value = _normalize_str(_cfg_value(cfg, "train.model"))
    return _sanitize_identifier(value or "unknown")
def _resolve_preprocess_variant(cfg: Any) -> str:
    value = _normalize_str(_cfg_value(cfg, "preprocess_variant.name"))
    if value is None:
        value = _normalize_str(_cfg_value(cfg, "preprocess.variant"))
    if value is None:
        value = _normalize_str(_cfg_value(cfg, "group.preprocess.preprocess_variant.name"))
    if value is None:
        run_dir = _normalize_str(_cfg_value(cfg, "train.inputs.preprocess_run_dir"))
        if run_dir:
            name = Path(run_dir).name
            if name.startswith("preprocess__"):
                value = name.split("__", 1)[-1]
    return _sanitize_identifier(value or "unknown")
def _resolve_raw_dataset_id(cfg: Any) -> str:
    for path in ("data.raw_dataset_id", "data.dataset_path", "data.processed_dataset_id"):
        token = _extract_dataset_token(_cfg_value(cfg, path))
        if token:
            return token
    return "unknown"
def _merge_extra_tags(cfg: Any, tags: Iterable[str]) -> None:
    existing = _cfg_value(cfg, "run.clearml.extra_tags") or []
    if isinstance(existing, str):
        existing_list = [existing]
    elif isinstance(existing, Iterable):
        existing_list = [str(item) for item in existing if item is not None]
    else:
        existing_list = []
    merged = existing_list[:]
    for tag in tags:
        if tag and tag not in merged:
            merged.append(tag)
    _set_cfg_value(cfg, "run.clearml.extra_tags", merged)
def apply_train_model_naming(cfg: Any) -> dict[str, Any]:
    """Apply train_model naming/tagging policy to config."""
    model_abbr = _resolve_model_abbr(cfg)
    preprocess_variant = _resolve_preprocess_variant(cfg)
    raw_dataset_id = _resolve_raw_dataset_id(cfg)
    raw_short = _shorten_token(raw_dataset_id)
    task_name = f"train__{model_abbr}__pp={preprocess_variant}__ds={raw_short}"
    tags = [
        f"model:{model_abbr}",
        f"preprocess:{preprocess_variant}",
        f"dataset:{raw_dataset_id}",
    ]
    _set_cfg_value(cfg, "run.clearml.task_name", task_name)
    _merge_extra_tags(cfg, tags)
    return {"task_name": task_name, "tags": tags}
def apply_train_ensemble_naming(cfg: Any) -> dict[str, Any]:
    """Apply train_ensemble naming/tagging policy to config."""
    method = _normalize_str(_cfg_value(cfg, "ensemble.method")) or "mean_topk"
    top_k = _cfg_value(cfg, "ensemble.top_k")
    try:
        top_k = int(top_k) if top_k is not None else None
    except (TypeError, ValueError):
        top_k = None
    preprocess_variant = _resolve_preprocess_variant(cfg)
    method_token = _sanitize_identifier(method)
    model_abbr = f"ensemble_{method_token}"
    if top_k is not None:
        task_name = f"train_ensemble/{method_token}(k={top_k})"
    else:
        task_name = f"train_ensemble/{method_token}"
    tags = [
        f"model:{model_abbr}",
        f"ensemble:{method_token}",
        f"preprocess:{preprocess_variant}",
    ]
    if top_k is not None:
        tags.append(f"topk:{top_k}")
    _set_cfg_value(cfg, "run.clearml.task_name", task_name)
    _merge_extra_tags(cfg, tags)
    return {"task_name": task_name, "tags": tags}
