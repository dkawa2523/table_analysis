#!/usr/bin/env python3
"""Regression checks for leaderboard UI inference columns."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PLATFORM_SRC = _REPO.parent / "ml_platform_v1-master" / "src"
for candidate in (_SRC, _PLATFORM_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tabular_analysis.processes import leaderboard as leaderboard_process
from tabular_analysis.viz import leaderboard_plots


class _FakeTable:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeFigure:
    def __init__(self, data=None):
        self.data = data or []
        self.layout = {}

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)


class _FakeGo:
    Table = _FakeTable
    Figure = _FakeFigure


def _assert_build_leaderboard_rows_includes_infer_targets() -> None:
    entries = [
        {
            "model_family": "single",
            "ensemble_method": None,
            "n_base_models": None,
            "composite_score": 0.9,
            "best_score": 0.1,
            "primary_metric_ci_low": None,
            "primary_metric_ci_mid": None,
            "primary_metric_ci_high": None,
            "primary_metric": "rmse",
            "primary_metric_source": "valid",
            "task_type": "regression",
            "model_id": "model-a",
            "infer_model_id": "model-a",
            "infer_train_task_id": None,
            "reference_kind": "model_id",
            "train_task_id": "train-a",
            "preprocess_variant": "stdscaler_ohe",
            "model_variant": "ridge",
            "train_task_ref": "train-a",
            "processed_dataset_id": "processed-a",
            "split_hash": "split-a",
            "rmse": 0.1,
        },
        {
            "model_family": "single",
            "ensemble_method": None,
            "n_base_models": None,
            "composite_score": 0.8,
            "best_score": 0.2,
            "primary_metric_ci_low": None,
            "primary_metric_ci_mid": None,
            "primary_metric_ci_high": None,
            "primary_metric": "rmse",
            "primary_metric_source": "valid",
            "task_type": "regression",
            "model_id": "local_bundle.joblib",
            "infer_model_id": None,
            "infer_train_task_id": "train-b",
            "reference_kind": "train_task_id",
            "train_task_id": "train-b",
            "preprocess_variant": "stdscaler_ohe",
            "model_variant": "lasso",
            "train_task_ref": "train-b",
            "processed_dataset_id": "processed-b",
            "split_hash": "split-b",
            "rmse": 0.2,
        },
    ]
    (rows, effective_top_k) = leaderboard_process._build_leaderboard_rows(entries, top_k=10, scoring_metrics=["rmse"])
    if effective_top_k != 10:
        raise AssertionError(f"unexpected top_k passthrough: {effective_top_k}")
    first = rows[0]
    second = rows[1]
    if first.get("infer_selector") != "infer.model_id" or first.get("infer_target") != "model-a":
        raise AssertionError(f"unexpected infer selector payload for model id row: {first}")
    if second.get("infer_selector") != "infer.train_task_id" or second.get("infer_target") != "train-b":
        raise AssertionError(f"unexpected infer selector payload for train task row: {second}")


def _assert_build_leaderboard_table_exposes_inference_columns() -> None:
    original_plotly_go = leaderboard_plots.plotly_go
    try:
        leaderboard_plots.plotly_go = lambda: _FakeGo
        rows = [
            {
                "rank": 1,
                "model_variant": "ridge",
                "preprocess_variant": "stdscaler_ohe",
                "reference_kind": "model_id",
                "infer_selector": "infer.model_id",
                "infer_target": "model-a",
                "rmse": 0.1,
                "composite_score": 0.9,
            }
        ]
        fig = leaderboard_plots.build_leaderboard_table(rows, metric_names=["rmse"], score_key="composite_score", score_label="Composite Score")
        if fig is None:
            raise AssertionError("expected fake plotly figure")
        headers = fig.data[0].kwargs["header"]["values"]
        for required in ("ref_kind", "infer_key", "infer_value"):
            if required not in headers:
                raise AssertionError(f"leaderboard plot table missing inference column {required}: {headers}")
        cells = fig.data[0].kwargs["cells"]["values"]
        header_to_values = dict(zip(headers, cells))
        if header_to_values.get("infer_key") != ["infer.model_id"]:
            raise AssertionError(f"unexpected infer_key cells: {header_to_values.get('infer_key')}")
        if header_to_values.get("infer_value") != ["model-a"]:
            raise AssertionError(f"unexpected infer_value cells: {header_to_values.get('infer_value')}")
    finally:
        leaderboard_plots.plotly_go = original_plotly_go


def main() -> int:
    _assert_build_leaderboard_rows_includes_infer_targets()
    _assert_build_leaderboard_table_exposes_inference_columns()
    print("OK: leaderboard ui contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
