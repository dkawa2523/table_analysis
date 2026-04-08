#!/usr/bin/env python3
"""Lightweight regression checks for refactor foundations."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PLATFORM_SRC = _REPO.parent / "ml_platform_v1-master" / "src"
for candidate in (_SRC, _PLATFORM_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tabular_analysis.common.drift_utils import collect_top_drift_features, normalize_drift_metric_names, select_primary_drift_metric
from tabular_analysis.common.feature_types import infer_tabular_feature_types
from tabular_analysis.common.model_reference import (
    build_infer_reference,
    resolve_infer_selection_assignment,
    resolve_infer_selection_key,
    resolve_infer_selection_kind,
    resolve_infer_selection_value,
    resolve_preferred_infer_reference,
)
from tabular_analysis.common.probability_utils import extract_positive_class_proba
from tabular_analysis.processes.artifact_writers import write_split_artifacts
from tabular_analysis.processes.contracts import ArtifactBundle, ExecutionResult, ReferenceInfo, ResolvedInputs, RuntimeSettings
from tabular_analysis.processes.infer_support import build_model_reference_payload


def _assert_feature_type_inference() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "flag": [True, False],
            "cat": ["a", "b"],
        }
    )
    (numeric, categorical) = infer_tabular_feature_types(df, ["num", "flag", "cat"])
    if numeric != ["num"]:
        raise AssertionError(f"unexpected numeric inference: {numeric}")
    if categorical != ["flag", "cat"]:
        raise AssertionError(f"unexpected categorical inference: {categorical}")


def _assert_probability_helper() -> None:
    import numpy as np

    arr = np.asarray([[0.2, 0.8], [0.9, 0.1]])
    values = extract_positive_class_proba(arr).tolist()
    if values != [0.8, 0.1]:
        raise AssertionError(f"unexpected positive-class probabilities: {values}")


def _assert_drift_helpers() -> None:
    metrics = normalize_drift_metric_names({"PSI": 1, "ks": 2}, default_metrics=("psi",))
    if metrics != ["psi", "ks"]:
        raise AssertionError(f"unexpected normalized metrics: {metrics}")

    report = {
        "metrics": ["ks", "psi"],
        "numeric": {
            "a": {"psi": 0.1, "ks": 0.2, "status": "ok"},
            "b": {"psi": 0.4, "ks": 0.1, "status": "warn"},
        },
        "categorical": {
            "c": {"psi": 0.3, "status": "warn"},
        },
    }
    primary = select_primary_drift_metric(report)
    if primary != "psi":
        raise AssertionError(f"unexpected primary metric: {primary}")
    top = collect_top_drift_features(report, metric="psi", limit=2, include_metrics=("ks",))
    if [item["feature"] for item in top] != ["b", "c"]:
        raise AssertionError(f"unexpected top features: {top}")


def _assert_split_artifact_writer() -> None:
    payload = {"train_index": [0, 1], "val_index": [2]}
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = write_split_artifacts(Path(tmpdir), payload)
        split_path = bundle.paths["split.json"]
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        if split_payload != payload:
            raise AssertionError("split artifact payload mismatch")


def _assert_model_reference_helpers() -> None:
    ref = build_infer_reference(model_id="local_bundle.joblib", registry_model_id="reg123", train_task_id="task123")
    if ref["infer_model_id"] != "reg123":
        raise AssertionError(f"unexpected preferred infer_model_id: {ref}")
    if ref["infer_train_task_id"] is not None:
        raise AssertionError(f"registry reference should not keep infer_train_task_id: {ref}")
    train_ref = build_infer_reference(
        model_id="C:/runs/train/01/model_bundle.joblib",
        train_task_id="task123",
    )
    if train_ref["infer_train_task_id"] != "task123":
        raise AssertionError(f"train_task_id should beat local bundle paths for infer portability: {train_ref}")
    if train_ref["infer_model_id"] is not None:
        raise AssertionError(f"local bundle path should not leak as infer_model_id when train_task_id exists: {train_ref}")
    merged = resolve_preferred_infer_reference({"recommended_model_id": "local_bundle.joblib"}, {"registry_model_id": "reg123"})
    if merged["infer_model_id"] != "reg123":
        raise AssertionError(f"unexpected recommendation precedence: {merged}")
    payload = build_model_reference_payload({"model_id": "local_bundle.joblib", "registry_model_id": "reg123", "train_task_id": "task123"}, model_bundle_path=Path("model_bundle.joblib"))
    if payload["infer_model_id"] != "reg123":
        raise AssertionError(f"unexpected manifest model reference payload: {payload}")
    path_fallback_payload = build_model_reference_payload({}, model_bundle_path=Path("model_bundle.joblib"))
    if path_fallback_payload["model_id"] is not None:
        raise AssertionError(f"model bundle path should not leak into infer reference payloads: {path_fallback_payload}")
    if resolve_infer_selection_key({"infer_model_id": "reg123"}) != "infer.model_id":
        raise AssertionError("infer selection key should prefer registry model ids")
    if resolve_infer_selection_value({"infer_train_task_id": "task123"}) != "task123":
        raise AssertionError("infer selection value should expose train task ids directly")
    if resolve_infer_selection_kind({"infer_train_task_id": "task123"}) != "train_task_id":
        raise AssertionError("infer selection kind should collapse to operator-facing train_task_id")
    if resolve_infer_selection_assignment({"infer_model_id": "reg123"}) != "infer.model_id=reg123":
        raise AssertionError("infer selection assignment should be copy-paste ready")


def _assert_contract_dataclasses() -> None:
    ref = ReferenceInfo(name="raw_dataset", identifier="local:abc")
    inputs = ResolvedInputs(references=(ref,))
    runtime = RuntimeSettings(task_name="preprocess", stage="dev", clearml_enabled=False)
    artifact = ArtifactBundle(paths={})
    result = ExecutionResult(out={"ok": True}, inputs={}, outputs={}, references=inputs.references, artifacts={"split": artifact})
    if result.references[0].identifier != "local:abc":
        raise AssertionError("ExecutionResult did not preserve ReferenceInfo")
    if runtime.task_name != "preprocess":
        raise AssertionError("RuntimeSettings contract mismatch")


def main() -> int:
    _assert_feature_type_inference()
    _assert_probability_helper()
    _assert_drift_helpers()
    _assert_split_artifact_writer()
    _assert_model_reference_helpers()
    _assert_contract_dataclasses()
    print("OK: refactor foundation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
