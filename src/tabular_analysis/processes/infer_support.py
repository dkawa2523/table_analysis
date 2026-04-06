from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..clearml.hparams import connect_infer
from ..common.collection_utils import to_container as _to_container
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..common.dataset_utils import load_tabular_frame as _load_dataframe, select_tabular_file as _select_tabular_file
from ..common.json_utils import load_json as _load_json
from ..common.model_reference import build_infer_reference
from ..platform_adapter_task import PlatformAdapterError, get_task_artifact_local_copy
from ..platform_adapter_artifacts import upload_artifact
from .lifecycle import emit_outputs_and_manifest

_BOOL_TRUTHY = {"true", "1", "yes", "y", "t"}
_BOOL_FALSY = {"false", "0", "no", "n", "f"}


def parse_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(int(value))
    text = str(value).strip().lower()
    if not text:
        return False
    if text in _BOOL_TRUTHY:
        return True
    if text in _BOOL_FALSY:
        return False
    return False


def to_int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _to_positive_int_or_none(value: Any) -> int | None:
    num = to_int_or_none(value)
    return num if num is not None and num > 0 else None


def _to_positive_float(value: Any, *, default: float) -> float:
    try:
        num = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        num = None
    return float(num) if num is not None and num > 0 else default


def _normalize_optimize_type(value: Any) -> str:
    key = (_normalize_str(value) or "").lower()
    if key in {"int", "integer"}:
        return "int"
    if key in {"categorical", "category", "choice", "choices", "discrete"}:
        return "categorical"
    return "float"


def _normalize_optimize_search_space(space: Any) -> list[dict[str, Any]]:
    space = _to_container(space)
    entries: list[dict[str, Any]] = []
    if not space:
        return entries
    if isinstance(space, str):
        text = space.strip()
        if not text:
            return entries
        try:
            space = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("infer.optimize.search_space must be valid JSON or a mapping/list.") from exc
    if isinstance(space, Mapping):
        for (name, payload) in space.items():
            entry: dict[str, Any] = {"name": str(name)}
            if isinstance(payload, Mapping):
                entry.update(dict(payload))
            elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                entry["choices"] = list(payload)
            else:
                entry["choices"] = [payload]
            entries.append(entry)
        return entries
    if isinstance(space, Sequence) and not isinstance(space, (str, bytes)):
        for item in space:
            if not isinstance(item, Mapping):
                continue
            entry = dict(item)
            name = entry.get("name") or entry.get("param")
            if name is None:
                continue
            entry["name"] = str(name)
            entries.append(entry)
    return entries


def _resolve_optimize_objective_keys(cfg: Any) -> list[str]:
    payload = _to_container(_cfg_value(cfg, "infer.optimize.objective"))
    keys_value: Any = None
    if isinstance(payload, Mapping):
        keys_value = payload.get("keys") if "keys" in payload else payload.get("key")
    elif payload is not None:
        keys_value = payload
    if keys_value is None:
        return ["prediction"]
    if isinstance(keys_value, Sequence) and not isinstance(keys_value, (str, bytes)):
        keys = [str(item).strip() for item in keys_value if str(item).strip()]
        return keys or ["prediction"]
    key_text = str(keys_value).strip()
    return [key_text] if key_text else ["prediction"]


def _format_optimize_search_space(space: Sequence[Mapping[str, Any]]) -> str | None:
    if not space:
        return None
    parts: list[str] = []
    for entry in space:
        name = entry.get("name")
        if not name:
            continue
        type_name = _normalize_optimize_type(entry.get("type"))
        if type_name == "categorical":
            choices = entry.get("choices") or entry.get("values")
            if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)):
                parts.append(f"{name}:cat[{len(choices)}]")
            else:
                parts.append(f"{name}:cat")
            continue
        low = entry.get("low")
        high = entry.get("high")
        step = entry.get("step")
        log_scale = parse_bool(entry.get("log")) if "log" in entry else False
        suffix = ""
        if step is not None:
            suffix = f",step={step}"
        if log_scale:
            suffix = f"{suffix},log"
        parts.append(f"{name}:{type_name}[{low},{high}{suffix}]")
    return "; ".join(parts) if parts else None


def frame_from_payload(payload: Any):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for infer.") from exc
    if isinstance(payload, Mapping):
        return pd.DataFrame([dict(payload)])
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if not payload:
            return pd.DataFrame()
        if all(isinstance(item, Mapping) for item in payload):
            return pd.DataFrame([dict(item) for item in payload])
    raise ValueError("infer.input_json must be a JSON object or list of objects.")


def _iter_csv_chunks(path: Path, *, chunk_size: int):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for infer.") from exc
    return pd.read_csv(path, chunksize=chunk_size)


def _iter_parquet_chunks(path: Path, *, chunk_size: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet chunked infer.") from exc
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


def iter_tabular_chunks(path: Path, *, chunk_size: int, max_rows: int | None = None):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        iterator = _iter_csv_chunks(path, chunk_size=chunk_size)
    elif suffix in (".parquet", ".pq"):
        iterator = _iter_parquet_chunks(path, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    row_offset = 0
    for chunk in iterator:
        if max_rows is not None and row_offset >= max_rows:
            break
        if max_rows is not None:
            remaining = max_rows - row_offset
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        if len(chunk) == 0:
            break
        yield (chunk, row_offset)
        row_offset += len(chunk)


def resolve_batch_execution_mode(cfg: Any, *, clearml_enabled: bool) -> str:
    execution = (_normalize_str(_cfg_value(cfg, "infer.batch.execution")) or "inline").lower()
    if execution not in {"inline", "clearml_children"}:
        raise ValueError("infer.batch.execution must be inline or clearml_children.")
    if execution == "clearml_children" and not clearml_enabled:
        raise ValueError("infer.batch.execution=clearml_children requires run.clearml.enabled=true.")
    return execution


def resolve_batch_settings(cfg: Any) -> dict[str, Any]:
    chunk_size = _to_positive_int_or_none(_cfg_value(cfg, "infer.batch.chunk_size", None))
    output_format = (_normalize_str(_cfg_value(cfg, "infer.batch.output_format", None)) or "csv").lower()
    if output_format not in {"csv", "parquet"}:
        raise ValueError("infer.batch.output_format must be csv or parquet.")
    write_mode = (_normalize_str(_cfg_value(cfg, "infer.batch.write_mode", None)) or "overwrite").lower()
    if write_mode not in {"overwrite", "append"}:
        raise ValueError("infer.batch.write_mode must be overwrite or append.")
    max_rows = _to_positive_int_or_none(_cfg_value(cfg, "infer.batch.max_rows", None))
    return {
        "chunk_size": chunk_size,
        "output_format": output_format,
        "write_mode": write_mode,
        "max_rows": max_rows,
    }


def resolve_batch_children_settings(cfg: Any) -> dict[str, Any]:
    return {
        "max_children": _to_positive_int_or_none(_cfg_value(cfg, "infer.batch.max_children", None)),
        "wait_timeout_sec": _to_positive_float(
            _cfg_value(cfg, "infer.batch.wait_timeout_sec", None),
            default=3600.0,
        ),
        "poll_interval_sec": _to_positive_float(
            _cfg_value(cfg, "infer.batch.poll_interval_sec", None),
            default=10.0,
        ),
    }


def resolve_optimize_settings(cfg: Any) -> dict[str, Any]:
    n_trials = _to_positive_int_or_none(_cfg_value(cfg, "infer.optimize.n_trials", None)) or 20
    direction = (_normalize_str(_cfg_value(cfg, "infer.optimize.direction", None)) or "maximize").lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("infer.optimize.direction must be maximize or minimize.")
    sampler_cfg = _to_container(_cfg_value(cfg, "infer.optimize.sampler", None))
    sampler_name = None
    sampler_seed = None
    if isinstance(sampler_cfg, Mapping):
        sampler_name = _normalize_str(sampler_cfg.get("name"))
        sampler_seed = sampler_cfg.get("seed")
    else:
        sampler_name = _normalize_str(sampler_cfg)
    if sampler_name is None:
        sampler_name = _normalize_str(_cfg_value(cfg, "infer.optimize.sampler_name", None))
    if sampler_seed is None:
        sampler_seed = _cfg_value(cfg, "infer.optimize.seed", None)
    wait_timeout_sec = _cfg_value(cfg, "infer.optimize.wait_timeout_sec", None)
    if wait_timeout_sec is None:
        wait_timeout_sec = _cfg_value(cfg, "infer.batch.wait_timeout_sec", None)
    poll_interval_sec = _cfg_value(cfg, "infer.optimize.poll_interval_sec", None)
    if poll_interval_sec is None:
        poll_interval_sec = _cfg_value(cfg, "infer.batch.poll_interval_sec", None)
    contour_params = _to_container(_cfg_value(cfg, "infer.optimize.plots.contour_params", None))
    if isinstance(contour_params, str):
        contour_params = [contour_params]
    if isinstance(contour_params, Sequence) and not isinstance(contour_params, (str, bytes)):
        contour_params = [str(item) for item in contour_params if str(item).strip()]
    else:
        contour_params = None
    return {
        "n_trials": n_trials,
        "direction": direction,
        "sampler_name": sampler_name or "tpe",
        "sampler_seed": to_int_or_none(sampler_seed),
        "search_space": _normalize_optimize_search_space(_cfg_value(cfg, "infer.optimize.search_space")),
        "objective_keys": _resolve_optimize_objective_keys(cfg),
        "top_k": _to_positive_int_or_none(_cfg_value(cfg, "infer.optimize.top_k", None)) or 5,
        "wait_timeout_sec": _to_positive_float(wait_timeout_sec, default=3600.0),
        "poll_interval_sec": _to_positive_float(poll_interval_sec, default=10.0),
        "history_log_scale": parse_bool(_cfg_value(cfg, "infer.optimize.plots.history_log_scale", False)),
        "contour_params": contour_params,
    }


def build_optimize_hparams(optimize_settings: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not optimize_settings:
        return None
    objective_keys = optimize_settings.get("objective_keys") or []
    if isinstance(objective_keys, Sequence) and not isinstance(objective_keys, (str, bytes)):
        objective_label = ",".join(str(item) for item in objective_keys if str(item).strip())
    else:
        objective_label = str(objective_keys) if objective_keys else None
    return {
        "infer.optimize.n_trials": optimize_settings.get("n_trials"),
        "infer.optimize.direction": optimize_settings.get("direction"),
        "infer.optimize.sampler": optimize_settings.get("sampler_name"),
        "infer.optimize.objective": objective_label,
        "infer.optimize.search_space": _format_optimize_search_space(
            optimize_settings.get("search_space") or []
        ),
        "infer.optimize.top_k": optimize_settings.get("top_k"),
    }


def load_batch_inputs(
    payload: Any | None,
    path: str | None,
    *,
    max_rows: int | None = None,
) -> tuple[Any | None, str | None]:
    if payload is not None:
        df = frame_from_payload(payload)
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows)
        return (df, None)
    if not path:
        return (None, None)
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        payload = _load_json(resolved)
        df = frame_from_payload(payload)
        if max_rows is not None and max_rows > 0:
            df = df.head(max_rows)
        return (df, str(resolved))
    data_path = _select_tabular_file(resolved)
    if data_path.suffix.lower() == ".csv" and max_rows is not None and max_rows > 0:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required for infer.") from exc
        df = pd.read_csv(data_path, nrows=max_rows)
        return (df, str(data_path))
    df = _load_dataframe(data_path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    return (df, str(data_path))


def load_train_profile(
    cfg: Any,
    bundle: Mapping[str, Any],
    model_bundle_path: Path,
    *,
    train_task_id: str | None,
    clearml_enabled: bool,
) -> tuple[dict[str, Any] | None, Path | None]:
    candidate = bundle.get("train_profile") if isinstance(bundle, Mapping) else None
    profile = dict(candidate) if isinstance(candidate, Mapping) else None
    if profile is not None:
        return (profile, None)
    profile_path = model_bundle_path.parent / "train_profile.json"
    if profile_path.exists():
        return (_load_json(profile_path), profile_path)
    if clearml_enabled and train_task_id:
        try:
            artifact_path = get_task_artifact_local_copy(cfg, train_task_id, "train_profile.json")
            return (_load_json(artifact_path), artifact_path)
        except PlatformAdapterError:
            pass
    return (None, None)


def ensure_drift_frame(data: Any, feature_names: Sequence[str] | None):
    try:
        import pandas as pd
    except (AttributeError, ImportError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("pandas is required for drift reporting.") from exc
    if isinstance(data, pd.DataFrame):
        return data
    values = data.toarray() if hasattr(data, "toarray") else data
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        return pd.DataFrame(values)
    n_cols = int(shape[1])
    columns = list(feature_names) if feature_names and len(feature_names) == n_cols else [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(values, columns=columns)


def _coerce_threshold(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return num if 0.0 <= num <= 1.0 else None


def resolve_threshold_used(bundle: Mapping[str, Any], *, n_classes: int | None) -> float | None:
    if n_classes != 2:
        return None
    postprocess = bundle.get("postprocess")
    candidates = (
        postprocess.get("threshold") if isinstance(postprocess, Mapping) else None,
        postprocess.get("best_threshold") if isinstance(postprocess, Mapping) else None,
        bundle.get("best_threshold"),
    )
    return next((threshold for threshold in map(_coerce_threshold, candidates) if threshold is not None), None)


def resolve_calibration_info(bundle: Mapping[str, Any]) -> dict[str, Any] | None:
    postprocess = bundle.get("postprocess")
    metrics = bundle.get("metrics")
    for candidate in (
        (postprocess or {}).get("calibration") if isinstance(postprocess, Mapping) else None,
        bundle.get("calibration"),
        (metrics or {}).get("calibration") if isinstance(metrics, Mapping) else None,
    ):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return None


def resolve_class_labels(bundle: Mapping[str, Any], model: Any) -> list[Any] | None:
    labels = bundle.get("class_labels")
    if isinstance(labels, (list, tuple)):
        return list(labels)
    for candidate in (bundle.get("label_encoder"), model):
        if (classes := getattr(candidate, "classes_", None)) is not None:
            return list(classes)
    return None


def resolve_preprocess_columns(
    preprocess_bundle: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str], str | None]:
    columns = preprocess_bundle.get("columns") or {}
    feature_columns = list(columns.get("feature_columns") or [])
    if not feature_columns:
        schema = preprocess_bundle.get("schema") or {}
        fields = schema.get("fields")
        if isinstance(fields, Mapping):
            feature_columns = [str(name) for name in fields.keys()]
    numeric_features = [str(name) for name in columns.get("numeric_features") or []]
    categorical_features = [str(name) for name in columns.get("categorical_features") or []]
    target_column = _normalize_str(columns.get("target_column")) or _normalize_str(
        preprocess_bundle.get("target_column")
    )
    return (feature_columns, numeric_features, categorical_features, target_column)


def build_model_reference_payload(meta: Mapping[str, Any], *, model_bundle_path: Path) -> dict[str, str | None]:
    legacy_model_id = _normalize_str(meta.get("model_id")) or str(model_bundle_path)
    reference_kind = _normalize_str(meta.get("reference_kind"))
    explicit_model_id = _normalize_str(meta.get("model_id"))
    if reference_kind == "train_task_artifact" and _normalize_str(meta.get("train_task_id")):
        reference = build_infer_reference(
            model_id=explicit_model_id,
            registry_model_id=meta.get("registry_model_id"),
            train_task_id=meta.get("train_task_id"),
        )
        if reference.get("registry_model_id") is None:
            reference = build_infer_reference(train_task_id=meta.get("train_task_id"))
    else:
        reference = build_infer_reference(
            model_id=explicit_model_id,
            registry_model_id=meta.get("registry_model_id"),
            train_task_id=meta.get("train_task_id"),
        )
        if reference.get("infer_model_id") is None and reference.get("infer_train_task_id") is None:
            reference = build_infer_reference(model_id=legacy_model_id)
    return {
        "model_id": legacy_model_id,
        "train_task_id": reference.get("train_task_id"),
        "registry_model_id": reference.get("registry_model_id"),
        "infer_model_id": reference.get("infer_model_id"),
        "infer_train_task_id": reference.get("infer_train_task_id"),
        "reference_kind": reference.get("reference_kind"),
    }


def handle_infer_dry_run(
    ctx: Any,
    cfg: Any,
    *,
    clearml_enabled: bool,
    infer_cfg: Any,
    mode: str,
    validation_mode: str,
    input_source: str,
    input_path_value: str | None,
    input_json_label: str | None,
    include_dataset: bool,
    include_execution: bool,
    optimize_settings: Mapping[str, Any] | None,
    optimize_hparams: Mapping[str, Any] | None,
) -> bool:
    dry_run = bool(getattr(infer_cfg, "dry_run", False))
    if not dry_run:
        return False
    model_id = _normalize_str(getattr(infer_cfg, "model_id", None))
    train_task_id = _normalize_str(getattr(infer_cfg, "train_task_id", None))
    reference = build_infer_reference(model_id=model_id or "dry_run", train_task_id=train_task_id)
    connect_infer(
        ctx,
        cfg,
        model_id=reference.get("infer_model_id") or reference.get("infer_train_task_id") or "dry_run",
        model_abbr=None,
        infer_mode=mode,
        schema_policy=validation_mode,
        input_source=input_source,
        input_path=input_path_value,
        input_json=input_json_label,
        provenance={"train_task_id": train_task_id},
        optimize_payload=optimize_hparams,
        include_dataset=include_dataset,
        include_execution=include_execution,
    )
    if mode == "optimize":
        trials_path = ctx.output_dir / "optimize_trials.csv"
        trials_path.write_text("", encoding="utf-8")
        summary_path = ctx.output_dir / "optimize_summary.json"
        summary_payload = {
            "n_trials": optimize_settings.get("n_trials") if optimize_settings else None,
            "direction": optimize_settings.get("direction") if optimize_settings else None,
            "objective_keys": optimize_settings.get("objective_keys") if optimize_settings else None,
            "search_space": optimize_settings.get("search_space") if optimize_settings else None,
        }
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if clearml_enabled:
            upload_artifact(ctx, trials_path.name, trials_path)
            upload_artifact(ctx, summary_path.name, summary_path)
        out = {
            "mode": mode,
            "model_id": model_id or "dry_run",
            "train_task_id": train_task_id,
            "infer_model_id": reference.get("infer_model_id") or "dry_run",
            "infer_train_task_id": reference.get("infer_train_task_id"),
            "reference_kind": reference.get("reference_kind"),
            "schema_validation": {"mode": validation_mode, "ok": True, "warnings_count": 0, "errors_count": 0},
            "optimize": {
                "n_trials": optimize_settings.get("n_trials") if optimize_settings else None,
                "direction": optimize_settings.get("direction") if optimize_settings else None,
                "objective_keys": optimize_settings.get("objective_keys") if optimize_settings else None,
            },
            "optimize_trials_path": str(trials_path),
            "optimize_summary_path": str(summary_path),
            "dry_run": True,
        }
        inputs = {
            "model_id": model_id,
            "train_task_id": train_task_id,
            "infer_model_id": reference.get("infer_model_id"),
            "infer_train_task_id": reference.get("infer_train_task_id"),
            "mode": mode,
            "input_path": input_path_value,
            "dry_run": True,
        }
        outputs = {"optimize_trials_path": str(trials_path), "optimize_summary_path": str(summary_path)}
        emit_outputs_and_manifest(
            ctx,
            cfg,
            process="infer",
            out=out,
            inputs=inputs,
            outputs=outputs,
            hash_payloads={"config_hash": ("config", cfg), "split_hash": "dry_run", "recipe_hash": "dry_run"},
            clearml_enabled=clearml_enabled,
        )
        return True
    if mode == "single":
        predictions_path = ctx.output_dir / "prediction.json"
        predictions_path.write_text(json.dumps({"prediction": None}, ensure_ascii=False, indent=2), encoding="utf-8")
        input_preview_path = ctx.output_dir / "input_preview.json"
        input_preview_path.write_text("{}", encoding="utf-8")
    else:
        predictions_path = ctx.output_dir / "predictions.csv"
        predictions_path.write_text("", encoding="utf-8")
        input_preview_path = ctx.output_dir / "input_preview.csv"
        input_preview_path.write_text("", encoding="utf-8")
    if clearml_enabled:
        upload_artifact(ctx, predictions_path.name, predictions_path)
        upload_artifact(ctx, input_preview_path.name, input_preview_path)
    out = {
        "predictions_path": str(predictions_path),
        "input_preview_path": str(input_preview_path),
        "mode": mode,
        "model_id": model_id or "dry_run",
        "train_task_id": train_task_id,
        "infer_model_id": reference.get("infer_model_id") or "dry_run",
        "infer_train_task_id": reference.get("infer_train_task_id"),
        "reference_kind": reference.get("reference_kind"),
        "schema_validation": {"mode": validation_mode, "ok": True, "warnings_count": 0, "errors_count": 0},
        "dry_run": True,
    }
    inputs = {
        "model_id": model_id,
        "train_task_id": train_task_id,
        "infer_model_id": reference.get("infer_model_id"),
        "infer_train_task_id": reference.get("infer_train_task_id"),
        "mode": mode,
        "input_path": input_path_value,
        "dry_run": True,
    }
    outputs = {"predictions_path": str(predictions_path), "input_preview_path": str(input_preview_path)}
    emit_outputs_and_manifest(
        ctx,
        cfg,
        process="infer",
        out=out,
        inputs=inputs,
        outputs=outputs,
        hash_payloads={"config_hash": ("config", cfg), "split_hash": "dry_run", "recipe_hash": "dry_run"},
        clearml_enabled=clearml_enabled,
    )
    return True


__all__ = [
    "build_model_reference_payload",
    "build_optimize_hparams",
    "ensure_drift_frame",
    "frame_from_payload",
    "handle_infer_dry_run",
    "iter_tabular_chunks",
    "load_batch_inputs",
    "load_train_profile",
    "parse_bool",
    "resolve_batch_children_settings",
    "resolve_batch_execution_mode",
    "resolve_batch_settings",
    "resolve_calibration_info",
    "resolve_class_labels",
    "resolve_optimize_settings",
    "resolve_preprocess_columns",
    "resolve_threshold_used",
    "to_int_or_none",
]
