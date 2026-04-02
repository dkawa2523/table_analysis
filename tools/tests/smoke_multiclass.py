#!/usr/bin/env python3
"""Local-mode smoke tests for multiclass classification flow."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], *, cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode})\n$ {' '.join(cmd)}\n\n{p.stdout}")
    return p.stdout


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def _load_json(path: Path) -> Dict[str, Any]:
    _must_exist(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _check_common(stage_dir: Path) -> None:
    _must_exist(stage_dir / "config_resolved.yaml")
    _must_exist(stage_dir / "out.json")
    _must_exist(stage_dir / "manifest.json")


def _make_toy_csv(path: Path, *, target_path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy/scikit-learn are required for smoke tests. "
            "Install requirements/base.txt first.\n"
            + str(exc)
        )

    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    columns = [f"num{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    df.drop(columns=["target"]).to_csv(target_path, index=False)


def _read_csv_header(path: Path) -> List[str]:
    _must_exist(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="logistic_regression")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_multiclass"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_multiclass.csv"
    infer_path = tmp / "toy_multiclass_infer.csv"
    _make_toy_csv(csv_path, target_path=infer_path)

    py = sys.executable

    # 1) dataset_register
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=dataset_register",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.dataset_path={csv_path}",
            "data.target_column=target",
        ],
        cwd=repo,
    )
    ds_dir = out_root / "01_dataset_register"
    _check_common(ds_dir)

    # 2) preprocess
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=preprocess",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.dataset_path={csv_path}",
            "data.target_column=target",
            "eval.task_type=classification",
            "group/preprocess=stdscaler_ohe",
            "data.split.strategy=stratified",
            "data.split.seed=42",
            "group/split=stratified",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    for k in ["processed_dataset_id", "split_hash", "recipe_hash"]:
        if k not in pp_out:
            raise AssertionError(f"preprocess out.json must contain {k}")

    # 3) train_model (multiclass classification)
    processed_ref = pp_out["processed_dataset_id"]
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=train_model",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.processed_dataset_id={processed_ref}",
            "eval.task_type=classification",
            "eval.primary_metric=f1_macro",
            "eval.direction=auto",
            "eval.classification.mode=auto",
            "eval.cv_folds=0",
            "viz.enabled=true",
            f"group/model={args.model}",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    _must_exist(tr_dir / "confusion_matrix.csv")
    _must_exist(tr_dir / "confusion_matrix.png")

    tr_out = _load_json(tr_dir / "out.json")
    if tr_out.get("n_classes") != 3:
        raise AssertionError("train_model out.json must contain n_classes=3")
    class_labels = tr_out.get("class_labels") or []
    if len(class_labels) != 3:
        raise AssertionError("train_model out.json must contain class_labels for 3 classes")

    metrics = _load_json(tr_dir / "metrics.json")
    holdout = metrics.get("holdout") or {}
    for key in ("accuracy", "f1_macro", "logloss"):
        if key not in holdout:
            raise AssertionError(f"metrics.json holdout must contain {key}")

    # 4) infer (batch)
    model_ref = tr_out["model_id"]
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=infer",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"infer.model_id={model_ref}",
            "infer.mode=batch",
            f"infer.input_path={infer_path}",
            "eval.classification.top_k=2",
        ],
        cwd=repo,
    )
    inf_dir = out_root / "04_infer"
    _check_common(inf_dir)
    inf_out = _load_json(inf_dir / "out.json")
    predictions_path = Path(inf_out["predictions_path"])
    header = _read_csv_header(predictions_path)
    if "pred_label" not in header:
        raise AssertionError("predictions.csv must contain pred_label")
    proba_cols = [name for name in header if name.startswith("proba_")]
    if len(proba_cols) < 3:
        raise AssertionError("predictions.csv must contain proba_<class> columns")
    for key in ("top1_label", "top1_proba", "top2_label", "top2_proba"):
        if key not in header:
            raise AssertionError(f"predictions.csv must contain {key}")

    print("OK: multiclass classification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
