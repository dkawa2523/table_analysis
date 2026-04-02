#!/usr/bin/env python3
"""Serving API tests (optional FastAPI)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tabular_analysis.io.bundle_io import load_bundle
from tabular_analysis.processes import infer as infer_process


def _fastapi_available() -> bool:
    return importlib.util.find_spec("fastapi") is not None


@contextmanager
def _patched_env(overrides: Mapping[str, str | None]):
    old = os.environ.copy()
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


@contextmanager
def _chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _find_model_bundle(repo: Path) -> Path:
    primary = repo / "work" / "_smoke_classification" / "out" / "03_train_model" / "model_bundle.joblib"
    if primary.exists():
        return primary
    fallback = repo / "work" / "_smoke_regression_model" / "out" / "03_train_model" / "model_bundle.joblib"
    if fallback.exists():
        return fallback
    work_root = repo / "work"
    if work_root.exists():
        for candidate in work_root.rglob("model_bundle.joblib"):
            return candidate
    raise FileNotFoundError("model_bundle.joblib not found under work/")


def _dummy_records(model_bundle_path: Path) -> list[dict[str, Any]]:
    bundle = load_bundle(model_bundle_path)
    preprocess_bundle = bundle.get("preprocess_bundle") if isinstance(bundle, dict) else None
    if not isinstance(preprocess_bundle, dict):
        raise RuntimeError("preprocess_bundle missing from model_bundle.joblib")
    df = infer_process._build_dummy_input(preprocess_bundle)
    records = df.to_dict(orient="records")
    if not records:
        raise RuntimeError("dummy input generation failed")
    return records


def _write_registry_state(path: Path, *, model_bundle_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": None,
        "usecases": {
            "demo": {
                "stages": {
                    "production": {
                        "current": {
                            "model_id": str(model_bundle_path),
                            "stage": "production",
                            "source": "local_test",
                        }
                    }
                }
            }
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _assert_audit_log(path: Path) -> None:
    if not path.exists():
        raise AssertionError("audit log was not created")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise AssertionError("audit log is empty")
    payload = json.loads(lines[-1])
    for key in (
        "timestamp",
        "request_id",
        "model_ref",
        "status",
        "latency_ms",
        "error_summary",
    ):
        if key not in payload:
            raise AssertionError(f"audit log missing field: {key}")


def main() -> int:
    if not _fastapi_available():
        print("SKIP: fastapi is not installed")
        return 0

    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        raise RuntimeError(f"fastapi is installed but TestClient failed: {exc}") from exc

    repo = Path(__file__).resolve().parents[2]
    model_bundle_path = _find_model_bundle(repo)
    records = _dummy_records(model_bundle_path)
    good_record = records[0]
    bad_record = dict(good_record)
    bad_record.pop(next(iter(bad_record)))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        registry_path = tmp_path / "model_registry_state.json"
        audit_path = tmp_path / "audit.jsonl"
        _write_registry_state(registry_path, model_bundle_path=model_bundle_path)

        base_env = {
            "MODEL_STAGE": "production",
            "MODEL_REF": None,
            "TABULAR_MODEL_BUNDLE": None,
            "TABULAR_MODEL_BUNDLE_PATH": None,
            "MODEL_BUNDLE_PATH": None,
            "API_KEY": "secret",
            "AUDIT_LOG_PATH": str(audit_path),
        }

        with _patched_env({**base_env, "SCHEMA_MODE": "strict"}), _chdir(tmp_path):
            serving_app = importlib.import_module("tabular_analysis.serve.app")
            app = serving_app.create_app()
            client = TestClient(app)

            health = client.get("/health")
            if health.status_code != 200:
                raise AssertionError(f"/health failed: {health.status_code} {health.text}")
            if health.json().get("status") != "ok":
                raise AssertionError("/health did not report ok")

            unauth = client.post("/predict", json={"records": [good_record]})
            if unauth.status_code != 401:
                raise AssertionError("/predict should require API key")

            invalid = client.post(
                "/predict",
                json={"records": [bad_record]},
                headers={"X-API-Key": "secret"},
            )
            if invalid.status_code != 422:
                raise AssertionError("strict schema should return 422")

            ok = client.post(
                "/predict",
                json={"records": [good_record]},
                headers={"X-API-Key": "secret"},
            )
            if ok.status_code != 200:
                raise AssertionError(f"/predict failed: {ok.status_code} {ok.text}")
            body = ok.json()
            if "schema_validation" not in body:
                raise AssertionError("schema_validation missing in response")
            if "request_id" not in body:
                raise AssertionError("request_id missing in response")

            _assert_audit_log(audit_path)

        with _patched_env({**base_env, "SCHEMA_MODE": "warn"}), _chdir(tmp_path):
            app_warn = serving_app.create_app()
            client_warn = TestClient(app_warn)
            warn_resp = client_warn.post(
                "/predict",
                json={"records": [bad_record]},
                headers={"X-API-Key": "secret"},
            )
            if warn_resp.status_code != 200:
                raise AssertionError("warn schema should return 200")
            warn_body = warn_resp.json()
            if warn_body.get("schema_validation", {}).get("ok") is True:
                raise AssertionError("warn schema should report ok=false for invalid input")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
