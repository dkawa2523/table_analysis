from __future__ import annotations

from ..common.collection_utils import to_container as _to_container
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str, normalize_task_type as _normalize_task_type
from ..common.dataset_utils import hash_file as _hash_file, load_tabular_frame as _load_dataframe, select_tabular_file as _select_tabular_file
from ..common.feature_types import infer_tabular_feature_types
from ..common.omegaconf_utils import ensure_config_alias
from ..common.schema_version import build_schema_tag, normalize_schema_version
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from ..clearml.datasets import create_processed_dataset, get_raw_dataset_local_copy, resolve_dataset_version
from ..clearml.hparams import build_preprocess_sections, connect_preprocess
from ..clearml.ui_logger import report_plotly
from ..feature_engineering.categorical import build_categorical_encoding_report, build_tabular_preprocessor, encode_target_for_mean, normalize_encoding
from ..io.bundle_io import save_bundle
from ..io.schema import infer_schema
from ..ops.clearml_identity import apply_clearml_identity
from ..ops.data_quality import raise_on_quality_fail, run_data_quality_gate
from ..platform_adapter_artifacts import hash_recipe, hash_split, upload_artifact
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..platform_adapter_task_context import update_task_properties
from ..viz.data_profile import build_missing_rate_comparison_bar, build_profile_comparison_table, build_profile_summary
from .artifact_writers import write_json_artifact, write_split_artifacts, write_text_artifact
from .contracts import ArtifactBundle, ExecutionResult, ReferenceInfo, ResolvedInputs, RuntimeSettings
from .lifecycle import emit_outputs_and_manifest, start_runtime

_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_PREPROCESS_RECOVERABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return bool(value)


def _hash_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _ensure_columns(df, columns: Iterable[str], *, label: str) -> list[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns in dataset: {missing}")
    return list(columns)


def _missing_stats(df, columns: Iterable[str]) -> dict[str, Any]:
    cols = list(columns)
    rows = int(df.shape[0])
    if not cols or rows <= 0:
        return {"missing_total": 0, "missing_rate": 0.0, "missing_columns": 0}
    subset = df[cols]
    missing_counts = subset.isna().sum()
    missing_total = int(missing_counts.sum())
    missing_columns = int((missing_counts > 0).sum())
    denom = rows * len(cols)
    missing_rate = float(missing_total / denom) if denom else 0.0
    return {
        "missing_total": missing_total,
        "missing_rate": missing_rate,
        "missing_columns": missing_columns,
    }


def _log_preprocess_profile(
    ctx: Any,
    *,
    df,
    processed_df,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    processed_feature_columns: list[str],
) -> None:
    (processed_numeric, processed_categorical) = infer_tabular_feature_types(processed_df, processed_feature_columns)
    raw_summary = build_profile_summary(
        df,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    processed_summary = build_profile_summary(
        processed_df,
        feature_columns=processed_feature_columns,
        numeric_features=processed_numeric,
        categorical_features=processed_categorical,
    )
    table_fig = build_profile_comparison_table(
        raw_summary,
        processed_summary,
        output_dir=ctx.output_dir,
        title="Raw vs Processed Summary",
    )
    report_plotly(ctx.task, "preprocess", "raw_vs_processed_summary", table_fig, step=0)
    missing_fig = build_missing_rate_comparison_bar(
        float(raw_summary.get("missing_rate", 0.0)),
        float(processed_summary.get("missing_rate", 0.0)),
        output_dir=ctx.output_dir,
        title="Missing Rate (raw vs processed)",
    )
    report_plotly(ctx.task, "preprocess", "missing_rate", missing_fig, step=0)


def _split_indices(
    df,
    *,
    strategy: str,
    test_size: float,
    seed: int,
    group_column: str | None,
    time_column: str | None,
    stratify_labels: Any | None = None,
) -> tuple[list[int], list[int]]:
    import math
    import numpy as np

    n_samples = int(df.shape[0])
    if n_samples < 2:
        raise ValueError("Dataset must contain at least 2 rows for splitting.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"data.split.test_size must be between 0 and 1. Got {test_size}.")

    indices = np.arange(n_samples)
    strategy_lower = str(strategy).strip().lower()
    if strategy_lower == "random":
        from sklearn.model_selection import train_test_split

        (train_idx, val_idx) = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
    elif strategy_lower == "group":
        if not group_column:
            raise ValueError("data.split.group_column is required for group strategy.")
        if group_column not in df.columns:
            raise ValueError(f"group_column not found in dataset: {group_column}")
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (train_idx, val_idx) = next(splitter.split(indices, groups=df[group_column]))
    elif strategy_lower == "stratified":
        if stratify_labels is None:
            raise ValueError("stratified split requires target labels for stratification.")
        labels = np.asarray(stratify_labels).reshape(-1)
        if labels.shape[0] != n_samples:
            raise ValueError("stratified split labels size does not match dataset rows.")
        if len(np.unique(labels)) < 2:
            raise ValueError("stratified split requires at least 2 classes in target.")
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (train_idx, val_idx) = next(splitter.split(indices, labels))
    elif strategy_lower == "time":
        if not time_column:
            raise ValueError("data.split.time_column is required for time strategy.")
        if time_column not in df.columns:
            raise ValueError(f"time_column not found in dataset: {time_column}")
        order = np.argsort(df[time_column].to_numpy(), kind="mergesort")
        n_val = int(math.ceil(n_samples * test_size))
        if n_val <= 0 or n_val >= n_samples:
            raise ValueError("data.split.test_size produces an invalid split size for time strategy.")
        val_idx = order[-n_val:]
        train_idx = order[:-n_val]
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    return (sorted(train_idx.tolist()), sorted(val_idx.tolist()))


def _resolve_preprocess_inputs(cfg: Any, *, clearml_enabled: bool) -> ResolvedInputs:
    dataset_path_value = _normalize_str(getattr(cfg.data, "dataset_path", None))
    raw_dataset_id_input = _normalize_str(getattr(cfg.data, "raw_dataset_id", None))

    dataset_path: Path | None = None
    if dataset_path_value:
        dataset_path = Path(dataset_path_value).expanduser().resolve()
    elif raw_dataset_id_input and (not clearml_enabled):
        if raw_dataset_id_input.startswith("local:"):
            dataset_path = None
        else:
            candidate = Path(raw_dataset_id_input).expanduser()
            if candidate.exists():
                dataset_path = candidate.resolve()

    dataset_file: Path | None = None
    if dataset_path is not None:
        dataset_file = _select_tabular_file(dataset_path)
    elif raw_dataset_id_input:
        if raw_dataset_id_input.startswith("local:"):
            if not dataset_path_value:
                raise ValueError("data.dataset_path is required when raw_dataset_id is local in local mode.")
        elif clearml_enabled:
            local_copy = get_raw_dataset_local_copy(cfg, raw_dataset_id_input)
            dataset_file = _select_tabular_file(local_copy)
        else:
            raise ValueError("data.dataset_path is required when ClearML is disabled and raw_dataset_id is not local.")
    else:
        raise ValueError("Either data.dataset_path or data.raw_dataset_id is required.")

    if dataset_file is None:
        raise RuntimeError("preprocess failed to resolve dataset file.")

    return ResolvedInputs(
        dataset_path=dataset_path,
        input_path=dataset_file,
        references=(
            ReferenceInfo(
                name="raw_dataset",
                identifier=raw_dataset_id_input,
                path=dataset_file,
                metadata={"dataset_path": dataset_path_value},
            ),
        ),
        metadata={
            "dataset_path_value": dataset_path_value,
            "raw_dataset_id_input": raw_dataset_id_input,
        },
    )


def _resolve_preprocess_runtime(cfg: Any, *, clearml_enabled: bool, df) -> RuntimeSettings:
    target_column = _normalize_str(getattr(cfg.data, "target_column", None))
    if not target_column or target_column not in df.columns:
        raise ValueError(f"target_column not found in dataset: {target_column}")

    id_columns = _ensure_columns(df, getattr(cfg.data, "id_columns", []) or [], label="id")
    drop_columns = _ensure_columns(df, getattr(cfg.data, "drop_columns", []) or [], label="drop")
    feature_columns = [col for col in df.columns if col not in set([target_column, *id_columns, *drop_columns])]
    if not feature_columns:
        raise ValueError("No feature columns remain after applying target/id/drop exclusions.")

    split_cfg = getattr(cfg.data, "split", None)
    split_strategy = _normalize_str(getattr(split_cfg, "strategy", None)) or "random"
    split_test_size = float(getattr(split_cfg, "test_size", 0.2))
    split_seed = int(getattr(split_cfg, "seed", 42))
    split_group_column = _normalize_str(getattr(split_cfg, "group_column", None))
    split_time_column = _normalize_str(getattr(split_cfg, "time_column", None))

    task_type = _normalize_task_type(getattr(getattr(cfg, "eval", None), "task_type", None))
    if split_strategy.lower() == "stratified" and task_type != "classification":
        raise ValueError("data.split.strategy=stratified requires eval.task_type=classification.")

    preprocess_variant = _to_container(getattr(cfg, "preprocess_variant", {})) or {}
    preprocess_variant_name = (
        _normalize_str(preprocess_variant.get("name"))
        or _normalize_str(getattr(getattr(cfg, "preprocess", None), "variant", None))
        or "unknown"
    )
    store_features = _normalize_bool(_cfg_value(cfg, "ops.processed_dataset.store_features", True), default=True)
    numeric_impute = _normalize_str(getattr(getattr(cfg, "preprocess", None), "numeric_impute", None)) or "mean"
    categorical_impute = _normalize_str(getattr(getattr(cfg, "preprocess", None), "categorical_impute", None)) or "most_frequent"

    categorical_cfg = getattr(getattr(cfg, "preprocess", None), "categorical", None)
    categorical_encoding_raw = _normalize_str(getattr(categorical_cfg, "encoding", None)) or preprocess_variant.get("categorical_encoder", None)
    categorical_encoding = normalize_encoding(categorical_encoding_raw or "onehot") or "onehot"
    auto_onehot_max_categories = int(getattr(categorical_cfg, "auto_onehot_max_categories", 50) or 50)
    hashing_cfg = getattr(categorical_cfg, "hashing", None)
    hashing_n_features = int(getattr(hashing_cfg, "n_features", 128) or 128)
    target_mean_cfg = getattr(categorical_cfg, "target_mean_oof", None)
    target_mean_folds = int(getattr(target_mean_cfg, "folds", 5) or 5)
    target_mean_smoothing = getattr(target_mean_cfg, "smoothing", 10.0)
    if target_mean_smoothing is None:
        target_mean_smoothing = 0.0
    target_mean_smoothing = float(target_mean_smoothing)

    return RuntimeSettings(
        task_name="preprocess",
        stage=str(cfg.task.stage),
        clearml_enabled=clearml_enabled,
        metadata={
            "task_type": task_type,
            "target_column": target_column,
            "id_columns": id_columns,
            "drop_columns": drop_columns,
            "feature_columns": feature_columns,
            "split_strategy": split_strategy,
            "split_test_size": split_test_size,
            "split_seed": split_seed,
            "split_group_column": split_group_column,
            "split_time_column": split_time_column,
            "preprocess_variant": preprocess_variant,
            "preprocess_variant_name": preprocess_variant_name,
            "store_features": store_features,
            "numeric_impute": numeric_impute,
            "categorical_impute": categorical_impute,
            "categorical_encoding": categorical_encoding,
            "auto_onehot_max_categories": auto_onehot_max_categories,
            "hashing_n_features": hashing_n_features,
            "target_mean_folds": target_mean_folds,
            "target_mean_smoothing": target_mean_smoothing,
        },
    )


def _fit_preprocess_pipeline(
    cfg: Any,
    *,
    ctx: Any,
    resolved_inputs: ResolvedInputs,
    runtime: RuntimeSettings,
    df,
) -> dict[str, Any]:
    dataset_path_value = resolved_inputs.metadata.get("dataset_path_value")
    raw_dataset_id_input = resolved_inputs.metadata.get("raw_dataset_id_input")
    feature_columns = list(runtime.metadata["feature_columns"])
    target_column = str(runtime.metadata["target_column"])
    split_strategy = str(runtime.metadata["split_strategy"])
    split_test_size = float(runtime.metadata["split_test_size"])
    split_seed = int(runtime.metadata["split_seed"])
    split_group_column = runtime.metadata.get("split_group_column")
    split_time_column = runtime.metadata.get("split_time_column")
    task_type = str(runtime.metadata["task_type"])
    store_features = bool(runtime.metadata["store_features"])
    preprocess_variant = dict(runtime.metadata["preprocess_variant"])
    preprocess_variant_name = str(runtime.metadata["preprocess_variant_name"])
    numeric_impute = str(runtime.metadata["numeric_impute"])
    categorical_impute = str(runtime.metadata["categorical_impute"])
    categorical_encoding = str(runtime.metadata["categorical_encoding"])
    auto_onehot_max_categories = int(runtime.metadata["auto_onehot_max_categories"])
    hashing_n_features = int(runtime.metadata["hashing_n_features"])
    target_mean_folds = int(runtime.metadata["target_mean_folds"])
    target_mean_smoothing = float(runtime.metadata["target_mean_smoothing"])
    id_columns = list(runtime.metadata["id_columns"])
    drop_columns = list(runtime.metadata["drop_columns"])

    connect_preprocess(
        ctx,
        cfg,
        raw_dataset_id=raw_dataset_id_input,
        dataset_path=dataset_path_value if not raw_dataset_id_input else None,
        preprocess_variant=preprocess_variant_name,
        split_strategy=split_strategy,
        split_seed=split_seed,
        store_features=store_features,
    )

    (numeric_features, categorical_features) = infer_tabular_feature_types(df, feature_columns)
    (train_idx, val_idx) = _split_indices(
        df,
        strategy=split_strategy,
        test_size=split_test_size,
        seed=split_seed,
        group_column=split_group_column,
        time_column=split_time_column,
        stratify_labels=df[target_column] if split_strategy.lower() == "stratified" else None,
    )
    split_payload = {"train_index": train_idx, "val_index": val_idx}
    split_hash = hash_split(split_payload)

    train_df = df.iloc[train_idx]
    cat_unique_counts = {col: int(train_df[col].nunique(dropna=True)) for col in categorical_features}
    (preprocessor, encoding_by_column, categorical_encoding) = build_tabular_preprocessor(
        preprocess_variant,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        numeric_impute=numeric_impute,
        categorical_impute=categorical_impute,
        categorical_encoding=categorical_encoding,
        auto_onehot_max_categories=auto_onehot_max_categories,
        hashing_n_features=hashing_n_features,
        target_mean_smoothing=target_mean_smoothing,
        unique_counts=cat_unique_counts,
    )
    if categorical_encoding == "target_mean_oof" and categorical_features:
        (y_encoded, _) = encode_target_for_mean(
            train_values=train_df[target_column],
            all_values=df[target_column],
            task_type=task_type,
        )
        preprocessor.fit(train_df[feature_columns], y_encoded[train_idx])
        transformed = preprocessor.transform_with_oof(
            df[feature_columns],
            y_encoded,
            train_idx=train_idx,
            folds=target_mean_folds,
            seed=split_seed,
            task_type=task_type,
        )
    else:
        preprocessor.fit(train_df[feature_columns])
        transformed = preprocessor.transform(df[feature_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except _PREPROCESS_RECOVERABLE_ERRORS:
        feature_names = [f"f{i}" for i in range(int(getattr(transformed, "shape", [0, 0])[1]))]

    try:
        import pandas as pd
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("pandas is required for preprocess output.") from exc

    unique_counts_imputed: dict[str, int] = {}
    if categorical_features and getattr(preprocessor, "categorical_imputer", None) is not None:
        cat_imputed = preprocessor.categorical_imputer.transform(train_df[categorical_features])
        cat_imputed_df = pd.DataFrame(cat_imputed, columns=categorical_features)
        unique_counts_imputed = {col: int(cat_imputed_df[col].nunique(dropna=True)) for col in categorical_features}

    encoding_report = build_categorical_encoding_report(
        categorical_features,
        encoding=categorical_encoding,
        encoding_by_column=encoding_by_column,
        unique_counts=unique_counts_imputed,
        hashing_n_features=hashing_n_features,
    )
    processed_df = pd.DataFrame(transformed, columns=feature_names)
    processed_df[target_column] = df[target_column].to_numpy()
    processed_feature_columns = [col for col in processed_df.columns if col != target_column]
    n_rows = int(df.shape[0])
    n_features = len(processed_feature_columns)
    missing_before = _missing_stats(df, feature_columns)
    missing_after = _missing_stats(processed_df, processed_feature_columns)
    dropped_columns = [str(col) for col in dict.fromkeys([*id_columns, *drop_columns])]
    quality_after = {
        "rows": n_rows,
        "columns_before": len(feature_columns),
        "columns_after": len(processed_feature_columns),
        "dropped_columns": {"count": len(dropped_columns), "columns": dropped_columns},
        "missing": {
            "before": missing_before,
            "after": missing_after,
            "reduced_total": missing_before["missing_total"] - missing_after["missing_total"],
        },
    }

    processed_path = ctx.output_dir / "processed_dataset.parquet"
    processed_df.to_parquet(processed_path, index=False)

    categorical_encoding_config = {
        "encoding": categorical_encoding,
        "auto_onehot_max_categories": auto_onehot_max_categories,
        "hashing": {"n_features": hashing_n_features},
        "target_mean_oof": {"folds": target_mean_folds, "smoothing": target_mean_smoothing},
    }
    recipe_payload = {
        "variant": preprocess_variant,
        "impute": {"numeric": numeric_impute, "categorical": categorical_impute},
        "categorical_encoding": categorical_encoding_config,
        "columns": {
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_column": target_column,
            "id_columns": id_columns,
            "drop_columns": drop_columns,
        },
    }
    recipe_hash = hash_recipe(recipe_payload)
    processed_id_payload = {
        "raw_dataset_hash": _hash_file(resolved_inputs.input_path),
        "recipe_hash": recipe_hash,
        "split_hash": split_hash,
    }
    processed_dataset_hash = _hash_payload(processed_id_payload)

    schema = infer_schema(df[feature_columns])
    schema["target_column"] = target_column
    schema["id_columns"] = id_columns
    schema["drop_columns"] = drop_columns
    schema_hash = _hash_payload(schema)
    usecase_id = _normalize_str(getattr(getattr(cfg, "run", None), "usecase_id", None)) or "unknown"
    schema_version = normalize_schema_version(getattr(getattr(cfg, "run", None), "schema_version", None), default="unknown")

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "split_payload": split_payload,
        "split_hash": split_hash,
        "processed_df": processed_df,
        "processed_path": processed_path,
        "feature_names": feature_names,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "processed_feature_columns": processed_feature_columns,
        "preprocessor": preprocessor,
        "encoding_report": encoding_report,
        "quality_after": quality_after,
        "recipe_payload": recipe_payload,
        "recipe_hash": recipe_hash,
        "processed_dataset_hash": processed_dataset_hash,
        "schema": schema,
        "schema_hash": schema_hash,
        "usecase_id": usecase_id,
        "schema_version": schema_version,
        "categorical_encoding": categorical_encoding,
        "categorical_encoding_config": categorical_encoding_config,
        "n_rows": n_rows,
        "n_features": n_features,
    }


def _build_encoding_note(
    *,
    categorical_encoding: str,
    auto_onehot_max_categories: int,
    hashing_n_features: int,
    target_mean_folds: int,
    target_mean_smoothing: float,
) -> str:
    if categorical_encoding == "auto":
        return f"auto(onehot_max={auto_onehot_max_categories}, hash_n_features={hashing_n_features})"
    if categorical_encoding == "hashing":
        return f"hashing(n_features={hashing_n_features})"
    if categorical_encoding == "target_mean_oof":
        return f"target_mean_oof(folds={target_mean_folds}, smoothing={target_mean_smoothing})"
    return categorical_encoding


def _write_preprocess_artifacts(
    *,
    ctx: Any,
    resolved_inputs: ResolvedInputs,
    runtime: RuntimeSettings,
    state: dict[str, Any],
) -> tuple[dict[str, ArtifactBundle], Path]:
    target_column = str(runtime.metadata["target_column"])
    id_columns = list(runtime.metadata["id_columns"])
    drop_columns = list(runtime.metadata["drop_columns"])
    split_strategy = str(runtime.metadata["split_strategy"])
    split_test_size = float(runtime.metadata["split_test_size"])
    split_seed = int(runtime.metadata["split_seed"])
    split_group_column = runtime.metadata.get("split_group_column")
    split_time_column = runtime.metadata.get("split_time_column")
    preprocess_variant_name = str(runtime.metadata["preprocess_variant_name"])
    task_type = str(runtime.metadata["task_type"])
    store_features = bool(runtime.metadata["store_features"])
    auto_onehot_max_categories = int(runtime.metadata["auto_onehot_max_categories"])
    hashing_n_features = int(runtime.metadata["hashing_n_features"])
    target_mean_folds = int(runtime.metadata["target_mean_folds"])
    target_mean_smoothing = float(runtime.metadata["target_mean_smoothing"])

    encoding_report_path = write_json_artifact(ctx.output_dir / "categorical_encoding_report.json", state["encoding_report"])
    quality_after_path = write_json_artifact(ctx.output_dir / "quality_after_preprocess.json", state["quality_after"])
    schema_path = write_json_artifact(ctx.output_dir / "schema.json", state["schema"])
    split_bundle = write_split_artifacts(
        ctx.output_dir,
        {
            "strategy": split_strategy,
            "seed": split_seed,
            "test_size": split_test_size,
            "group_column": split_group_column,
            "time_column": split_time_column,
            **state["split_payload"],
        },
    )
    recipe_path = write_json_artifact(ctx.output_dir / "recipe.json", state["recipe_payload"])
    feature_names_path = write_json_artifact(ctx.output_dir / "feature_names.json", state["feature_names"])

    bundle = {
        "preprocess_variant": preprocess_variant_name,
        "pipeline": state["preprocessor"],
        "feature_names": state["feature_names"],
        "columns": {
            "feature_columns": state["feature_columns"],
            "numeric_features": state["numeric_features"],
            "categorical_features": state["categorical_features"],
            "target_column": target_column,
            "id_columns": id_columns,
            "drop_columns": drop_columns,
        },
        "schema": state["schema"],
        "impute": {
            "numeric": runtime.metadata["numeric_impute"],
            "categorical": runtime.metadata["categorical_impute"],
        },
    }
    bundle_path = ctx.output_dir / "preprocess_bundle.joblib"
    save_bundle(bundle_path, bundle)
    bundle_hash = _hash_file(bundle_path)

    meta_payload = {
        "processed_dataset_hash": state["processed_dataset_hash"],
        "raw_dataset_hash": _hash_file(resolved_inputs.input_path),
        "raw_dataset_id": resolved_inputs.metadata.get("raw_dataset_id_input"),
        "recipe_hash": state["recipe_hash"],
        "split_hash": state["split_hash"],
        "schema_hash": state["schema_hash"],
        "bundle_hash": bundle_hash,
        "preprocess_variant": preprocess_variant_name,
        "task_type": task_type,
        "target_column": target_column,
        "n_rows": state["n_rows"],
        "n_features": state["n_features"],
        "store_features": store_features,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": state["schema_version"],
        "usecase_id": state["usecase_id"],
    }
    meta_path = write_json_artifact(ctx.output_dir / "meta.json", meta_payload)

    encoding_note = _build_encoding_note(
        categorical_encoding=state["categorical_encoding"],
        auto_onehot_max_categories=auto_onehot_max_categories,
        hashing_n_features=hashing_n_features,
        target_mean_folds=target_mean_folds,
        target_mean_smoothing=target_mean_smoothing,
    )
    summary_lines = [
        "# Preprocess Summary",
        "",
        f"- variant: {preprocess_variant_name}",
        f"- categorical_encoding: {encoding_note} (high-card handling)",
        f"- rows: {state['n_rows']}",
        f"- features: {len(state['feature_columns'])} (numeric={len(state['numeric_features'])}, categorical={len(state['categorical_features'])})",
        f"- split: train={len(state['train_idx'])} val={len(state['val_idx'])} strategy={split_strategy}",
        "- processed_dataset_id: pending",
        f"- store_features: {store_features}",
        f"- split_hash: {state['split_hash']}",
        f"- recipe_hash: {state['recipe_hash']}",
        f"- schema_hash: {state['schema_hash']}",
    ]
    summary_path = write_text_artifact(ctx.output_dir / "summary.md", "\n".join(summary_lines) + "\n")

    dataset_dir = ctx.output_dir / "processed_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_path, dataset_dir / "schema.json")
    shutil.copy2(split_bundle.primary_path, dataset_dir / "split.json")
    shutil.copy2(recipe_path, dataset_dir / "recipe.json")
    shutil.copy2(bundle_path, dataset_dir / "preprocess_bundle.joblib")
    shutil.copy2(meta_path, dataset_dir / "meta.json")
    shutil.copy2(feature_names_path, dataset_dir / "feature_names.json")
    if store_features:
        x_path = dataset_dir / "X.parquet"
        y_path = dataset_dir / "y.parquet"
        state["processed_df"][state["processed_feature_columns"]].to_parquet(x_path, index=False)
        state["processed_df"][[target_column]].to_parquet(y_path, index=False)
    else:
        for name in ("X.parquet", "y.parquet"):
            candidate = dataset_dir / name
            if candidate.exists():
                candidate.unlink()

    artifacts = {
        "split": split_bundle,
        "schema": ArtifactBundle(primary_path=schema_path, paths={"schema.json": schema_path}),
        "recipe": ArtifactBundle(primary_path=recipe_path, paths={"recipe.json": recipe_path}),
        "feature_names": ArtifactBundle(primary_path=feature_names_path, paths={"feature_names.json": feature_names_path}),
        "meta": ArtifactBundle(primary_path=meta_path, paths={"meta.json": meta_path}),
        "summary": ArtifactBundle(primary_path=summary_path, paths={"summary.md": summary_path}),
        "encoding_report": ArtifactBundle(primary_path=encoding_report_path, paths={"categorical_encoding_report.json": encoding_report_path}),
        "quality_after": ArtifactBundle(primary_path=quality_after_path, paths={"quality_after_preprocess.json": quality_after_path}),
        "bundle": ArtifactBundle(primary_path=bundle_path, paths={"preprocess_bundle.joblib": bundle_path}),
        "processed_dataset": ArtifactBundle(primary_path=state["processed_path"], paths={"processed_dataset.parquet": state["processed_path"]}),
    }
    return (artifacts, dataset_dir)


def _register_processed_dataset_if_needed(
    cfg: Any,
    *,
    runtime: RuntimeSettings,
    resolved_inputs: ResolvedInputs,
    state: dict[str, Any],
    dataset_dir: Path,
) -> tuple[str, str | None]:
    if not runtime.clearml_enabled:
        return (f"local:{state['processed_dataset_hash']}", None)

    usecase_id = state["usecase_id"]
    schema_version = state["schema_version"]
    preprocess_variant_name = runtime.metadata["preprocess_variant_name"]
    dataset_name = f"processed__{usecase_id}__{preprocess_variant_name}__{state['split_hash']}__{schema_version}"
    dataset_project = _normalize_str(getattr(getattr(cfg, "task", None), "project_name", None))
    dataset_tags = [f"usecase:{usecase_id}", "process:preprocess", "type:processed", build_schema_tag(schema_version)]
    parent_ids = None
    raw_dataset_id_input = resolved_inputs.metadata.get("raw_dataset_id_input")
    if raw_dataset_id_input and (not str(raw_dataset_id_input).startswith("local:")):
        parent_ids = [raw_dataset_id_input]
    (dataset_sections, dataset_order) = build_preprocess_sections(
        cfg,
        raw_dataset_id=raw_dataset_id_input,
        dataset_path=resolved_inputs.metadata.get("dataset_path_value") if not raw_dataset_id_input else None,
        preprocess_variant=preprocess_variant_name,
        split_strategy=runtime.metadata["split_strategy"],
        split_seed=runtime.metadata["split_seed"],
        store_features=runtime.metadata["store_features"],
    )
    processed_dataset_id = create_processed_dataset(
        cfg,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_tags=dataset_tags,
        description=(
            f"raw_hash={_hash_file(resolved_inputs.input_path)} "
            f"recipe_hash={state['recipe_hash']} split_hash={state['split_hash']} "
            f"schema_hash={state['schema_hash']} store_features={runtime.metadata['store_features']}"
        ),
        parent_dataset_ids=parent_ids,
        task_sections=dataset_sections,
        task_section_order=dataset_order,
    )
    return (processed_dataset_id, resolve_dataset_version(cfg, processed_dataset_id))


def _build_execution_result(
    *,
    resolved_inputs: ResolvedInputs,
    runtime: RuntimeSettings,
    state: dict[str, Any],
    processed_dataset_id: str,
    processed_dataset_version: str | None,
    artifacts: dict[str, ArtifactBundle],
) -> ExecutionResult:
    raw_dataset_hash = _hash_file(resolved_inputs.input_path)
    out = {
        "processed_dataset_id": processed_dataset_id,
        "processed_dataset_version": processed_dataset_version,
        "split_hash": state["split_hash"],
        "recipe_hash": state["recipe_hash"],
        "schema_hash": state["schema_hash"],
        "processed_dataset_hash": state["processed_dataset_hash"],
        "n_rows": state["n_rows"],
        "n_features": state["n_features"],
        "preprocess_variant": runtime.metadata["preprocess_variant_name"],
    }
    if not runtime.clearml_enabled:
        out["processed_dataset_path"] = str(state["processed_path"])

    raw_dataset_id_value = resolved_inputs.metadata.get("raw_dataset_id_input")
    if not raw_dataset_id_value and resolved_inputs.metadata.get("dataset_path_value"):
        raw_dataset_id_value = f"local:{raw_dataset_hash}"
    inputs: dict[str, Any] = {
        "raw_dataset_id": raw_dataset_id_value,
        "upstream_task_ids": [],
        "preprocess_variant": runtime.metadata["preprocess_variant_name"],
        "target_column": runtime.metadata["target_column"],
        "categorical_encoding": state["categorical_encoding_config"],
        "store_features": runtime.metadata["store_features"],
    }
    if resolved_inputs.metadata.get("dataset_path_value"):
        inputs["dataset_path"] = resolved_inputs.metadata["dataset_path_value"]
    outputs = {
        "processed_dataset_id": processed_dataset_id,
        "processed_dataset_version": processed_dataset_version,
        "split_hash": state["split_hash"],
        "recipe_hash": state["recipe_hash"],
        "schema_hash": state["schema_hash"],
        "n_rows": state["n_rows"],
        "n_features": state["n_features"],
    }
    references = (
        *resolved_inputs.references,
        ReferenceInfo(
            name="processed_dataset",
            identifier=processed_dataset_id,
            path=state["processed_path"],
            metadata={
                "split_hash": state["split_hash"],
                "recipe_hash": state["recipe_hash"],
                "schema_hash": state["schema_hash"],
            },
        ),
    )
    return ExecutionResult(out=out, inputs=inputs, outputs=outputs, references=references, artifacts=artifacts)


def _upload_preprocess_artifacts(*, ctx: Any, runtime: RuntimeSettings, execution: ExecutionResult) -> None:
    if not runtime.clearml_enabled:
        return
    for bundle in execution.artifacts.values():
        for (name, path) in bundle.paths.items():
            upload_artifact(ctx, name, path)


def run(cfg: Any) -> None:
    ensure_config_alias(cfg, "group.preprocess.preprocess_variant", "preprocess_variant")
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    ctx = start_runtime(
        cfg,
        stage=cfg.task.stage,
        task_name="preprocess",
        tags=identity.tags,
        properties=identity.user_properties,
    )
    clearml_enabled = is_clearml_enabled(cfg)
    resolved_inputs = _resolve_preprocess_inputs(cfg, clearml_enabled=clearml_enabled)
    df = _load_dataframe(resolved_inputs.input_path)
    runtime = _resolve_preprocess_runtime(cfg, clearml_enabled=clearml_enabled, df=df)

    quality_result = run_data_quality_gate(
        cfg=cfg,
        ctx=ctx,
        df=df,
        target_column=runtime.metadata["target_column"],
        task_type=runtime.metadata["task_type"],
        id_columns=runtime.metadata["id_columns"],
        output_dir=ctx.output_dir,
    )
    raise_on_quality_fail(
        cfg=cfg,
        ctx=ctx,
        gate=quality_result["gate"],
        payload=quality_result["payload"],
        json_path=quality_result["paths"]["json"],
    )

    state = _fit_preprocess_pipeline(cfg, ctx=ctx, resolved_inputs=resolved_inputs, runtime=runtime, df=df)
    if clearml_enabled:
        _log_preprocess_profile(
            ctx,
            df=df,
            processed_df=state["processed_df"],
            feature_columns=state["feature_columns"],
            numeric_features=state["numeric_features"],
            categorical_features=state["categorical_features"],
            processed_feature_columns=state["processed_feature_columns"],
        )

    (artifacts, dataset_dir) = _write_preprocess_artifacts(
        ctx=ctx,
        resolved_inputs=resolved_inputs,
        runtime=runtime,
        state=state,
    )
    (processed_dataset_id, processed_dataset_version) = _register_processed_dataset_if_needed(
        cfg,
        runtime=runtime,
        resolved_inputs=resolved_inputs,
        state=state,
        dataset_dir=dataset_dir,
    )
    summary_path = artifacts["summary"].primary_path
    if summary_path is not None:
        updated_lines = []
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- processed_dataset_id: "):
                updated_lines.append(f"- processed_dataset_id: {processed_dataset_id}")
            else:
                updated_lines.append(line)
        write_text_artifact(summary_path, "\n".join(updated_lines) + "\n")

    if clearml_enabled:
        update_task_properties(
            ctx,
            {
                "processed_dataset_id": processed_dataset_id,
                "split_hash": state["split_hash"],
                "recipe_hash": state["recipe_hash"],
                "schema_hash": state["schema_hash"],
                "processed_dataset_version": processed_dataset_version,
            },
        )

    execution = _build_execution_result(
        resolved_inputs=resolved_inputs,
        runtime=runtime,
        state=state,
        processed_dataset_id=processed_dataset_id,
        processed_dataset_version=processed_dataset_version,
        artifacts=artifacts,
    )
    _upload_preprocess_artifacts(ctx=ctx, runtime=runtime, execution=execution)
    emit_outputs_and_manifest(
        ctx,
        cfg,
        process="preprocess",
        out=execution.out,
        inputs=execution.inputs,
        outputs=execution.outputs,
        hash_payloads={
            "config_hash": ("config", cfg),
            "split_hash": state["split_hash"],
            "recipe_hash": state["recipe_hash"],
            "schema_hash": state["schema_hash"],
        },
        clearml_enabled=clearml_enabled,
    )
