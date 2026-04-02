from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..common.probability_utils import extract_positive_class_proba
from ..io.bundle_io import load_bundle
from ..processes import infer as infer_process
from .auth import resolve_principal, verify_api_key
from .model_loader import ModelResolution, resolve_model_bundle
from .settings import ServingSettings

_FASTAPI_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)


def _sanitize(value: Any) -> Any:
    return infer_process._sanitize_json_value(value)


def _normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "records" in payload:
            records = payload.get("records")
            if isinstance(records, list):
                return [dict(item) for item in records if isinstance(item, dict)]
            if isinstance(records, dict):
                return [dict(records)]
        if "inputs" in payload:
            records = payload.get("inputs")
            if isinstance(records, list):
                return [dict(item) for item in records if isinstance(item, dict)]
            if isinstance(records, dict):
                return [dict(records)]
        return [dict(payload)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise ValueError("request body must be an object, a list of objects, or a payload with records/inputs.")


def _load_runtime_model(
    *,
    model_bundle_path: str | Path | None,
    settings: ServingSettings,
) -> tuple[ModelResolution | None, dict[str, Any] | None, str | None]:
    try:
        resolution = resolve_model_bundle(model_bundle_path, settings=settings)
        bundle = load_bundle(resolution.model_bundle_path)
        if not isinstance(bundle, dict):
            raise ValueError("model_bundle.joblib must deserialize to a mapping.")
        return (resolution, bundle, None)
    except Exception as exc:  # noqa: BLE001
        return (None, None, str(exc))


def _resolve_prediction_context(bundle: dict[str, Any]) -> tuple[Any, dict[str, Any], str]:
    preprocess_bundle = bundle.get("preprocess_bundle")
    if not isinstance(preprocess_bundle, dict):
        raise ValueError("model_bundle is missing preprocess_bundle.")
    model = bundle.get("calibrated_model") if bundle.get("calibrated_model") is not None else bundle.get("model")
    if model is None:
        raise ValueError("model_bundle is missing model.")
    task_type = str(bundle.get("task_type") or "regression").strip().lower()
    return (model, preprocess_bundle, task_type)


def _predict_rows(bundle: dict[str, Any], *, records: list[dict[str, Any]], validation_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pandas as pd
    except _FASTAPI_IMPORT_ERRORS as exc:
        raise RuntimeError("pandas is required for serving.") from exc

    (model, preprocess_bundle, task_type) = _resolve_prediction_context(bundle)
    df = pd.DataFrame(records)
    (validated_df, validation) = infer_process._validate_inputs(df, preprocess_bundle, validation_mode=validation_mode)
    if validation_mode == "strict" and (not validation.get("ok")):
        return ([], validation)

    pipeline = preprocess_bundle.get("pipeline")
    if pipeline is None:
        raise ValueError("preprocess_bundle.pipeline is missing.")
    transformed = pipeline.transform(validated_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if task_type == "classification":
        predictor = model
        n_classes = bundle.get("n_classes")
        try:
            n_classes = int(n_classes) if n_classes is not None else None
        except (TypeError, ValueError, OverflowError):
            n_classes = None
        if not hasattr(predictor, "predict_proba"):
            preds = predictor.predict(transformed)
            proba = None
        else:
            proba = predictor.predict_proba(transformed)
            threshold_used = infer_process._resolve_threshold_used(bundle, n_classes=n_classes)
            if threshold_used is not None:
                preds = (extract_positive_class_proba(proba) >= threshold_used).astype(int)
            else:
                preds = predictor.predict(transformed)

        class_labels = infer_process._resolve_class_labels(bundle, bundle.get("model"))
        label_encoder = bundle.get("label_encoder")
        if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
            pred_labels = label_encoder.inverse_transform(preds)
        else:
            pred_labels = preds
            if class_labels:
                try:
                    pred_labels = [class_labels[int(idx)] for idx in preds]
                except (TypeError, ValueError, IndexError):
                    pred_labels = preds
        if hasattr(pred_labels, "tolist"):
            pred_labels = pred_labels.tolist()
        if not isinstance(pred_labels, list):
            pred_labels = list(pred_labels)

        rows: list[dict[str, Any]] = []
        if proba is not None:
            try:
                import numpy as np
            except _FASTAPI_IMPORT_ERRORS as exc:
                raise RuntimeError("numpy is required for classification serving.") from exc
            proba_arr = np.asarray(proba)
            if proba_arr.ndim == 1:
                proba_arr = np.stack([1.0 - proba_arr, proba_arr], axis=1)
            if class_labels is None or len(class_labels) != int(proba_arr.shape[1]):
                class_labels = [str(idx) for idx in range(int(proba_arr.shape[1]))]
            for idx, pred_label in enumerate(pred_labels):
                row = {"prediction": _sanitize(pred_label), "predicted_label": _sanitize(pred_label)}
                row["predicted_proba"] = infer_process._build_proba_payload(proba_arr[idx], class_labels)
                rows.append(row)
        else:
            rows = [{"prediction": _sanitize(pred_label), "predicted_label": _sanitize(pred_label)} for pred_label in pred_labels]
        return (rows, validation)

    preds = model.predict(transformed)
    rows = infer_process._preds_to_rows(preds)
    payload_rows: list[dict[str, Any]] = []
    for row in rows:
        if len(row) == 1:
            payload_rows.append({"prediction": _sanitize(row[0])})
        else:
            payload_rows.append({"prediction": [_sanitize(value) for value in row]})
    return (payload_rows, validation)


def _write_audit_entry(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_app(model_bundle_path: str | Path | None = None, *, strict_schema: bool | None = None):
    try:
        from fastapi import Body, FastAPI, Header, HTTPException
    except _FASTAPI_IMPORT_ERRORS as exc:
        raise RuntimeError("fastapi is required for serving.") from exc

    settings = ServingSettings.from_env()
    if strict_schema is True:
        settings = ServingSettings(
            model_ref=settings.model_ref,
            model_stage=settings.model_stage,
            schema_mode="strict",
            api_keys=settings.api_keys,
            audit_log_path=settings.audit_log_path,
            usecase_id=settings.usecase_id,
            model_registry_state_path=settings.model_registry_state_path,
        )
    elif strict_schema is False and settings.schema_mode == "strict":
        settings = ServingSettings(
            model_ref=settings.model_ref,
            model_stage=settings.model_stage,
            schema_mode="warn",
            api_keys=settings.api_keys,
            audit_log_path=settings.audit_log_path,
            usecase_id=settings.usecase_id,
            model_registry_state_path=settings.model_registry_state_path,
        )

    (resolution, bundle, load_error) = _load_runtime_model(model_bundle_path=model_bundle_path, settings=settings)
    app = FastAPI(title="Tabular Analysis Serving", version="1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if bundle is not None and load_error is None else "error",
            "model_ref": resolution.model_ref if resolution is not None else settings.model_ref,
            "model_bundle_path": str(resolution.model_bundle_path) if resolution is not None else None,
            "schema_mode": settings.schema_mode,
            "error": load_error,
        }

    @app.post("/predict")
    def predict(payload: Any = Body(...), x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = uuid4().hex
        principal = resolve_principal(x_api_key, settings.api_keys)
        status = "ok"
        error_summary = None
        http_status = 200
        row_count = None
        validation: dict[str, Any] = {"ok": True, "warnings_count": 0, "errors_count": 0, "issues": {}}
        try:
            if not verify_api_key(x_api_key, settings.api_keys):
                status = "unauthorized"
                http_status = 401
                error_summary = "invalid_api_key"
                raise HTTPException(status_code=401, detail="X-API-Key is required.")
            if bundle is None:
                status = "error"
                http_status = 503
                error_summary = load_error or "model_not_loaded"
                raise HTTPException(status_code=503, detail=error_summary)

            records = _normalize_records(payload)
            row_count = len(records)
            (predictions, validation) = _predict_rows(bundle, records=records, validation_mode=settings.schema_mode)
            if settings.schema_mode == "strict" and (not validation.get("ok")):
                status = "schema_error"
                http_status = 422
                error_summary = "schema_validation_failed"
                raise HTTPException(
                    status_code=422,
                    detail={"request_id": request_id, "schema_validation": validation},
                )
            return {
                "request_id": request_id,
                "model_ref": resolution.model_ref if resolution is not None else settings.model_ref,
                "predictions": predictions,
                "schema_validation": validation,
            }
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            status = "error"
            http_status = 500
            error_summary = str(exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            _write_audit_entry(
                settings.audit_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "request_id": request_id,
                    "model_ref": resolution.model_ref if resolution is not None else settings.model_ref,
                    "status": status,
                    "http_status": http_status,
                    "latency_ms": latency_ms,
                    "error_summary": error_summary,
                    "schema_mode": settings.schema_mode,
                    "principal": principal,
                    "rows": row_count,
                    "schema_validation_ok": bool(validation.get("ok")),
                },
            )

    return app


try:
    app = create_app()
except Exception:  # noqa: BLE001
    app = None
