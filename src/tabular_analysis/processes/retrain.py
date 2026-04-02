from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, set_cfg_value as _set_cfg_value, normalize_str as _normalize_str
from ..common.collection_utils import to_mapping as _to_mapping
from ..common.dataset_utils import derive_local_raw_dataset_id
from ..common.json_utils import load_json as _load_json
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import uuid
from ..ops.clearml_identity import apply_clearml_identity, build_project_name
from ..platform_adapter_artifacts import resolve_output_dir, upload_artifact
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..platform_adapter_task import get_task_artifact_local_copy, report_markdown, update_task_properties
from .lifecycle import emit_outputs_and_manifest, start_runtime
from . import pipeline as pipeline_process
def _ensure_run_id(cfg: Any, path: str) -> str:
    existing = _normalize_str(_cfg_value(cfg, path))
    if existing:
        return existing
    new_id = uuid.uuid4().hex
    _set_cfg_value(cfg, path, new_id)
    return new_id
def _project_name(cfg: Any, stage: str) -> str:
    project_root = _normalize_str(_cfg_value(cfg, 'run.clearml.project_root')) or 'MFG'
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'
    return build_project_name(project_root, usecase_id, stage, cfg=cfg)
def _resolve_ref_out(cfg: Any, ref: Mapping[str, Any], *, clearml_enabled: bool, label: str) -> tuple[dict[str, Any], Path]:
    run_dir = _normalize_str(ref.get('run_dir'))
    task_id = _normalize_str(ref.get('task_id'))
    out_path = None
    if run_dir:
        candidate = Path(run_dir).expanduser() / 'out.json'
        if candidate.exists():
            out_path = candidate
    if out_path is None and clearml_enabled and task_id:
        out_path = get_task_artifact_local_copy(cfg, task_id, 'out.json')
    if out_path is None or not out_path.exists():
        raise FileNotFoundError(f'{label} out.json not found for ref: {ref}')
    payload = _load_json(out_path)
    if not isinstance(payload, dict):
        raise ValueError(f'{label} out.json must contain an object.')
    return (payload, out_path)
def _resolve_challenger_ref(leaderboard_out: Mapping[str, Any], *, clearml_enabled: bool) -> str | None:
    ordered_keys = ['recommended_train_task_id', 'recommended_train_task_ref', 'recommended_model_id']
    if not clearml_enabled:
        ordered_keys = ['recommended_train_task_ref', 'recommended_train_task_id', 'recommended_model_id']
    for key in ordered_keys:
        value = _normalize_str(leaderboard_out.get(key))
        if value:
            return value
    return None
def run(cfg: Any) -> None:
    retrain_run_id = _ensure_run_id(cfg, 'run.retrain_run_id')
    grid_run_id = _ensure_run_id(cfg, 'run.grid_run_id')
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='retrain', tags=[*identity.tags, f'retrain:{retrain_run_id}'], properties={**identity.user_properties, 'retrain_run_id': retrain_run_id})
    clearml_enabled = is_clearml_enabled(cfg)
    dataset_path = _normalize_str(_cfg_value(cfg, 'retrain.dataset_path')) or _normalize_str(_cfg_value(cfg, 'data.dataset_path'))
    dataset_id = _normalize_str(_cfg_value(cfg, 'retrain.dataset_id')) or _normalize_str(_cfg_value(cfg, 'data.raw_dataset_id'))
    if not dataset_id and dataset_path:
        dataset_id = derive_local_raw_dataset_id(dataset_path)
        _set_cfg_value(cfg, 'data.raw_dataset_id', dataset_id)
        _set_cfg_value(cfg, 'retrain.dataset_id', dataset_id)
    pipeline_cfg = copy.deepcopy(cfg)
    _set_cfg_value(pipeline_cfg, 'task.name', 'pipeline')
    _set_cfg_value(pipeline_cfg, 'task.stage', '99_pipeline')
    _set_cfg_value(pipeline_cfg, 'task.project_name', _project_name(cfg, '99_pipeline'))
    _set_cfg_value(pipeline_cfg, 'run.grid_run_id', grid_run_id)
    _set_cfg_value(pipeline_cfg, 'run.retrain_run_id', retrain_run_id)
    if dataset_path:
        _set_cfg_value(pipeline_cfg, 'data.dataset_path', dataset_path)
    if dataset_id:
        _set_cfg_value(pipeline_cfg, 'data.raw_dataset_id', dataset_id)
    pipeline_process.run(pipeline_cfg)
    pipeline_output_dir = resolve_output_dir(pipeline_cfg, getattr(pipeline_cfg.task, 'stage', '99_pipeline'))
    pipeline_run_path = pipeline_output_dir / 'pipeline_run.json'
    if not pipeline_run_path.exists():
        raise FileNotFoundError(f'pipeline_run.json not found: {pipeline_run_path}')
    pipeline_run = _load_json(pipeline_run_path)
    leaderboard_ref = _to_mapping(pipeline_run.get('leaderboard_ref'))
    if not leaderboard_ref:
        raise ValueError('pipeline_run is missing leaderboard_ref.')
    (leaderboard_out, leaderboard_out_path) = _resolve_ref_out(cfg, leaderboard_ref, clearml_enabled=clearml_enabled, label='leaderboard')
    challenger_ref = _resolve_challenger_ref(leaderboard_out, clearml_enabled=clearml_enabled)
    if not challenger_ref:
        raise ValueError('leaderboard did not provide a challenger reference.')
    decision_payload: dict[str, Any] = {'retrain_run_id': retrain_run_id, 'grid_run_id': grid_run_id, 'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'dataset_path': dataset_path, 'dataset_id': dataset_id, 'challenger_model_ref': challenger_ref, 'leaderboard_out': {'recommended_model_id': leaderboard_out.get('recommended_model_id'), 'recommended_train_task_ref': leaderboard_out.get('recommended_train_task_ref'), 'recommended_train_task_id': leaderboard_out.get('recommended_train_task_id'), 'recommended_best_score': leaderboard_out.get('recommended_best_score'), 'recommended_primary_metric': leaderboard_out.get('recommended_primary_metric'), 'recommendation_count': leaderboard_out.get('recommendation_count'), 'recommended_models': leaderboard_out.get('recommended_models')}}
    decision_payload['decision'] = {'action': 'select_model', 'reason': 'user_select_at_infer'}
    decision_path = ctx.output_dir / 'retrain_decision.json'
    decision_path.write_text(json.dumps(decision_payload, ensure_ascii=True, indent=2), encoding='utf-8')
    retrain_run_payload = {'retrain_run_id': retrain_run_id, 'grid_run_id': grid_run_id, 'pipeline_run_path': str(pipeline_run_path), 'leaderboard_ref': leaderboard_ref, 'leaderboard_out_path': str(leaderboard_out_path)}
    retrain_run_path = ctx.output_dir / 'retrain_run.json'
    retrain_run_path.write_text(json.dumps(retrain_run_payload, ensure_ascii=True, indent=2), encoding='utf-8')
    summary_lines = ['# Retrain Summary', '', f'- retrain_run_id: {retrain_run_id}', f'- grid_run_id: {grid_run_id}', f"- dataset_path: {dataset_path or 'n/a'}", f"- dataset_id: {dataset_id or 'n/a'}", f'- challenger_model_ref: {challenger_ref}']
    summary_lines.extend(['', '## Decision', '- action: select_model (choose at infer time)'])
    summary_path = ctx.output_dir / 'retrain_summary.md'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    if clearml_enabled:
        for (name, path) in [('retrain_decision.json', decision_path), ('retrain_run.json', retrain_run_path), ('retrain_summary.md', summary_path)]:
            upload_artifact(ctx, name, path)
        report_markdown(ctx, title='', markdown='\n'.join(summary_lines))
        update_task_properties(ctx, {'retrain_run_id': retrain_run_id, 'grid_run_id': grid_run_id, 'decision': 'select_model'})
    out = {'retrain_run_id': retrain_run_id, 'grid_run_id': grid_run_id, 'pipeline_run_path': str(pipeline_run_path), 'leaderboard_ref': leaderboard_ref, 'decision_json': str(decision_path), 'summary_md': str(summary_path), 'retrain_run_json': str(retrain_run_path)}
    emit_outputs_and_manifest(ctx, cfg, process='retrain', out=out, inputs={'dataset_path': dataset_path, 'dataset_id': dataset_id, 'challenger_model_ref': challenger_ref}, outputs={'retrain_decision_json': str(decision_path), 'retrain_run_json': str(retrain_run_path), 'retrain_summary_md': str(summary_path)}, hash_payloads={'config_hash': ('config', cfg), 'split_hash': ('split', {}), 'recipe_hash': ('recipe', {})}, clearml_enabled=clearml_enabled)
