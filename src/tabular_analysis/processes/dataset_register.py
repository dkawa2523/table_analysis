from __future__ import annotations
from ..common.config_utils import normalize_str as _normalize_str, normalize_task_type as _normalize_task_type
from ..common.dataset_utils import hash_file as _hash_file, load_tabular_frame as _load_dataframe, select_tabular_file as _select_tabular_file
from ..common.feature_types import infer_tabular_feature_types
from ..common.schema_version import build_schema_tag, normalize_schema_version
import json
from pathlib import Path
from typing import Any
from ..clearml.datasets import create_raw_dataset, get_raw_dataset_local_copy
from ..clearml.hparams import build_dataset_register_sections, connect_dataset_register
from ..clearml.ui_logger import log_scalar, report_plotly
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..ops.clearml_identity import apply_clearml_identity
from ..ops.data_quality import raise_on_quality_fail, run_data_quality_gate
from .lifecycle import emit_outputs_and_manifest, start_runtime
from ..viz.data_profile import build_categorical_topk_bars, build_head_table, build_missingness_bar, build_numeric_histograms, build_target_distribution, summarize_dataframe
_PROFILE_SETTINGS = {'max_numeric': 4, 'max_categorical': 4, 'max_categories': 10, 'max_columns': 30, 'sample_rows': 5000, 'table_rows': 5, 'table_columns': 20}
def _series_name(prefix: str, name: Any, max_len: int=48) -> str:
    text = str(name).replace('/', '_').replace(' ', '_')
    if len(text) > max_len:
        text = text[:max_len - 3] + '...'
    return f'{prefix}_{text}'
def _build_schema(df) -> dict[str, Any]:
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    null_count = df.isna().sum()
    fields: dict[str, Any] = {}
    for col in df.columns:
        key = str(col)
        count = int(null_count[col])
        rate = float(count / rows) if rows else 0.0
        fields[key] = {'dtype': str(df[col].dtype), 'null_count': count, 'null_rate': rate}
    return {'rows': rows, 'columns': cols, 'fields': fields}
def _infer_schema(path: Path, output_dir: Path):
    df = _load_dataframe(path)
    schema = _build_schema(df)
    preview_path = output_dir / 'preview.csv'
    df.head(5).to_csv(preview_path, index=False)
    schema_path = output_dir / 'schema.json'
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding='utf-8')
    return (df, schema)
def _log_data_profile(ctx: Any, df, *, target_column: str | None, id_columns: list[Any]) -> None:
    summary = summarize_dataframe(df)
    log_scalar(ctx.task, 'dataset_register', 'rows', summary.get('rows'), step=0)
    log_scalar(ctx.task, 'dataset_register', 'columns', summary.get('columns'), step=0)
    log_scalar(ctx.task, 'dataset_register', 'missing_rate', summary.get('missing_rate'), step=0)
    exclude = {str(value) for value in id_columns if value is not None}
    if target_column:
        exclude.add(str(target_column))
    columns = list(getattr(df, 'columns', []))
    feature_columns = [col for col in columns if str(col) not in exclude]
    if not feature_columns:
        feature_columns = columns
    (numeric_features, categorical_features) = infer_tabular_feature_types(df, feature_columns)
    head_fig = build_head_table(df, max_rows=_PROFILE_SETTINGS['table_rows'], max_columns=_PROFILE_SETTINGS['table_columns'], output_dir=ctx.output_dir)
    report_plotly(ctx.task, 'dataset_register', 'head_table', head_fig, step=0)
    missing_fig = build_missingness_bar(df, columns=columns, max_columns=_PROFILE_SETTINGS['max_columns'], output_dir=ctx.output_dir)
    report_plotly(ctx.task, 'dataset_register', 'missingness', missing_fig, step=0)
    for (col, fig) in build_numeric_histograms(df, numeric_features, max_columns=_PROFILE_SETTINGS['max_numeric'], sample_rows=_PROFILE_SETTINGS['sample_rows'], output_dir=ctx.output_dir, title_prefix='Numeric Histogram'):
        report_plotly(ctx.task, 'dataset_register', _series_name('numeric_hist', col), fig, step=0)
    for (col, fig) in build_categorical_topk_bars(df, categorical_features, max_columns=_PROFILE_SETTINGS['max_categorical'], top_k=_PROFILE_SETTINGS['max_categories'], sample_rows=_PROFILE_SETTINGS['sample_rows'], output_dir=ctx.output_dir, title_prefix='Top Categories'):
        report_plotly(ctx.task, 'dataset_register', _series_name('categorical_topk', col), fig, step=0)
    target_fig = build_target_distribution(df, str(target_column) if target_column else '', bins=30, top_k=_PROFILE_SETTINGS['max_categories'], sample_rows=_PROFILE_SETTINGS['sample_rows'], output_dir=ctx.output_dir)
    report_plotly(ctx.task, 'dataset_register', 'target_distribution', target_fig, step=0)
def run(cfg: Any) -> None:
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='dataset_register', tags=identity.tags, properties=identity.user_properties)
    clearml_enabled = is_clearml_enabled(cfg)
    dataset_path_value = _normalize_str(getattr(cfg.data, 'dataset_path', None))
    raw_dataset_id_input = _normalize_str(getattr(cfg.data, 'raw_dataset_id', None))
    target_column = _normalize_str(getattr(getattr(cfg, 'data', None), 'target_column', None))
    connect_dataset_register(ctx, cfg, dataset_path=dataset_path_value, target_column=target_column, raw_dataset_id=raw_dataset_id_input)
    raw_dataset_id: str | None = None
    raw_dataset_hash: str | None = None
    raw_schema: dict[str, Any] | None = None
    raw_df = None
    dataset_path: Path | None = None
    if dataset_path_value:
        dataset_path = Path(dataset_path_value).expanduser().resolve()
    elif raw_dataset_id_input and (not clearml_enabled):
        if raw_dataset_id_input.startswith('local:'):
            dataset_path = None
        else:
            candidate = Path(raw_dataset_id_input).expanduser()
            if candidate.exists():
                dataset_path = candidate.resolve()
    if dataset_path is not None:
        dataset_file = _select_tabular_file(dataset_path)
        raw_dataset_hash = _hash_file(dataset_file)
        (raw_df, raw_schema) = _infer_schema(dataset_file, ctx.output_dir)
        if clearml_enabled:
            usecase_id = _normalize_str(getattr(getattr(cfg, 'run', None), 'usecase_id', None)) or 'unknown'
            schema_version = normalize_schema_version(getattr(getattr(cfg, 'run', None), 'schema_version', None), default='unknown')
            dataset_name = f'{usecase_id}__raw__{dataset_file.stem}'
            dataset_project = _normalize_str(getattr(getattr(cfg, 'task', None), 'project_name', None))
            dataset_tags = [f'usecase:{usecase_id}', 'process:dataset_register', build_schema_tag(schema_version)]
            (dataset_sections, dataset_order) = build_dataset_register_sections(cfg, dataset_path=dataset_path_value or str(dataset_file), target_column=target_column, raw_dataset_id=None)
            raw_dataset_id = create_raw_dataset(cfg, dataset_path=dataset_file, dataset_name=dataset_name, dataset_project=dataset_project, dataset_tags=dataset_tags, description=f'raw_dataset_hash={raw_dataset_hash}', task_sections=dataset_sections, task_section_order=dataset_order)
        else:
            raw_dataset_id = f'local:{raw_dataset_hash}'
    elif raw_dataset_id_input:
        if raw_dataset_id_input.startswith('local:'):
            raw_dataset_id = raw_dataset_id_input
            raw_dataset_hash = raw_dataset_id_input.split(':', 1)[1]
            if not raw_dataset_hash:
                raise ValueError('raw_dataset_id local:<hash> must include a hash.')
            raw_schema = {'rows': None, 'columns': None, 'fields': {}}
        elif clearml_enabled:
            local_copy = get_raw_dataset_local_copy(cfg, raw_dataset_id_input)
            dataset_file = _select_tabular_file(local_copy)
            raw_dataset_hash = _hash_file(dataset_file)
            (raw_df, raw_schema) = _infer_schema(dataset_file, ctx.output_dir)
            raw_dataset_id = raw_dataset_id_input
        else:
            raise ValueError('data.dataset_path is required when ClearML is disabled and raw_dataset_id is not local.')
    else:
        raise ValueError('Either data.dataset_path or data.raw_dataset_id is required.')
    if raw_dataset_id is None or raw_dataset_hash is None or raw_schema is None:
        raise RuntimeError('dataset_register failed to resolve raw_dataset outputs.')
    task_type = _normalize_task_type(getattr(getattr(cfg, 'eval', None), 'task_type', None))
    target_column = _normalize_str(getattr(getattr(cfg, 'data', None), 'target_column', None))
    id_columns = getattr(getattr(cfg, 'data', None), 'id_columns', []) or []
    quality_result = run_data_quality_gate(cfg=cfg, ctx=ctx, df=raw_df, target_column=target_column, task_type=task_type, id_columns=id_columns, output_dir=ctx.output_dir, schema=raw_schema if isinstance(raw_schema, dict) else None)
    data_quality = quality_result['payload']
    quality_summary = quality_result['summary']
    gate = quality_result['gate']
    data_quality_path = quality_result['paths']['json']
    if clearml_enabled and raw_df is not None:
        _log_data_profile(ctx, raw_df, target_column=target_column, id_columns=id_columns)
    out = {'raw_dataset_id': raw_dataset_id, 'raw_dataset_hash': raw_dataset_hash, 'raw_schema': raw_schema, 'data_quality_summary': quality_summary}
    inputs: dict[str, Any] = {}
    if dataset_path_value:
        inputs['dataset_path'] = dataset_path_value
    if raw_dataset_id_input:
        inputs['raw_dataset_id'] = raw_dataset_id_input
    emit_outputs_and_manifest(ctx, cfg, process='dataset_register', out=out, inputs=inputs, outputs={'raw_dataset_id': raw_dataset_id, 'raw_dataset_hash': raw_dataset_hash}, hash_payloads={'config_hash': ('config', cfg), 'split_hash': ('split', {}), 'recipe_hash': ('recipe', {})}, clearml_enabled=clearml_enabled)
    raise_on_quality_fail(cfg=cfg, ctx=ctx, gate=gate, payload=data_quality, json_path=data_quality_path)
