#!/usr/bin/env python3
"""Local-mode smoke tests for regression train_model variants."""

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


def _make_toy_csv(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(5, 2, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["target"] = (
        0.3 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "b").astype(float)
    ) + rng.normal(0, 0.1, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--split", default="random", choices=["random", "group", "time"])
    ap.add_argument("--expect-feature-importance", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_regression_model"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_regression.csv"
    _make_toy_csv(csv_path)

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
    ds_out = _load_json(ds_dir / "out.json")
    if "raw_dataset_id" not in ds_out:
        raise AssertionError("dataset_register out.json must contain raw_dataset_id")

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
            "eval.task_type=regression",
            "group/preprocess=stdscaler_ohe",
            f"data.split.strategy={args.split}",
            "data.split.seed=42",
            f"group/split={args.split}",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    for k in ["processed_dataset_id", "split_hash", "recipe_hash"]:
        if k not in pp_out:
            raise AssertionError(f"preprocess out.json must contain {k}")

    # 3) train_model (regression)
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
            "eval.task_type=regression",
            "eval.primary_metric=rmse",
            "eval.direction=minimize",
            "eval.cv_folds=0",
            f"group/model={args.model}",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    if args.expect_feature_importance:
        _must_exist(tr_dir / "feature_importance.csv")
    tr_out = _load_json(tr_dir / "out.json")
    for k in ["model_id", "primary_metric", "best_score", "task_type"]:
        if k not in tr_out:
            raise AssertionError(f"train_model out.json must contain {k}")

    print("OK: train_model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
