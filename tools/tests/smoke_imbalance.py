#!/usr/bin/env python3
"""Local-mode smoke tests for imbalance handling."""

from __future__ import annotations

import argparse
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


def _make_imbalanced_csv(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(3, 1.2, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1]),
        }
    )
    score = 0.8 * df["num1"] - 0.3 * df["num2"] + (df["cat"] == "b").astype(float)
    threshold = float(score.quantile(0.95))
    df["target"] = (score >= threshold).astype(int)
    if df["target"].nunique() < 2:
        raise RuntimeError("failed to generate imbalanced binary target")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_imbalance"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_imbalance.csv"
    _make_imbalanced_csv(csv_path)

    py = sys.executable

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
    for key in ["processed_dataset_id", "split_hash", "recipe_hash"]:
        if key not in pp_out:
            raise AssertionError(f"preprocess out.json must contain {key}")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=train_model",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.processed_dataset_id={pp_out['processed_dataset_id']}",
            "eval.task_type=classification",
            "eval.primary_metric=pr_auc",
            "eval.direction=auto",
            "eval.cv_folds=0",
            "eval.imbalance.enabled=true",
            "eval.imbalance.strategy=class_weight",
            "eval.metrics.fbeta_beta=2.0",
            "group/model=logistic_regression",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)

    tr_out = _load_json(tr_dir / "out.json")
    imbalance = tr_out.get("imbalance") or {}
    if imbalance.get("enabled") is not True:
        raise AssertionError("train_model out.json must include imbalance.enabled=true")
    if imbalance.get("strategy") != "class_weight":
        raise AssertionError("train_model out.json must include imbalance.strategy=class_weight")
    if imbalance.get("applied") is not True:
        raise AssertionError("train_model out.json must include imbalance.applied=true")

    metrics = _load_json(tr_dir / "metrics.json")
    if metrics.get("direction") != "maximize":
        raise AssertionError("metrics.json direction must be maximize for pr_auc")
    holdout = metrics.get("holdout") or {}
    for key in ("pr_auc", "balanced_accuracy", "fbeta"):
        if key not in holdout:
            raise AssertionError(f"metrics.json holdout must contain {key}")

    print("OK: imbalance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
