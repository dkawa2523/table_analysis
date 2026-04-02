#!/usr/bin/env python3
"""Local-mode smoke tests for classification thresholding."""

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

    rng = np.random.default_rng(7)
    n = 240
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(4, 1.5, size=n),
            "cat": rng.choice(["x", "y", "z"], size=n),
        }
    )
    score = 0.4 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "y").astype(float)
    df["target"] = (score > score.median()).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_thresholding"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_thresholding.csv"
    _make_toy_csv(csv_path)

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
    ds_out = _load_json(ds_dir / "out.json")
    if ds_out.get("raw_dataset_id") is None:
        raise AssertionError("dataset_register out.json must contain raw_dataset_id")

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
            "data.split.strategy=random",
            "data.split.seed=42",
            "group/split=random",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    for k in ["processed_dataset_id", "split_hash", "recipe_hash"]:
        if k not in pp_out:
            raise AssertionError(f"preprocess out.json must contain {k}")

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
            "eval.primary_metric=f1",
            "eval.direction=maximize",
            "eval.thresholding.enabled=true",
            "eval.thresholding.metric=f1",
            "group/model=logistic_regression",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    tr_out = _load_json(tr_dir / "out.json")
    best_threshold = tr_out.get("best_threshold")
    if best_threshold is None:
        raise AssertionError("train_model out.json must contain best_threshold")
    best_threshold = float(best_threshold)
    if not (0.0 <= best_threshold <= 1.0):
        raise AssertionError("best_threshold must be between 0 and 1")
    if "threshold_metric" not in tr_out:
        raise AssertionError("train_model out.json must contain threshold_metric")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=infer",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"infer.model_id={tr_out['model_id']}",
            "infer.mode=single",
        ],
        cwd=repo,
    )
    inf_dir = out_root / "04_infer"
    _check_common(inf_dir)
    inf_out = _load_json(inf_dir / "out.json")
    predictions_path = Path(inf_out["predictions_path"])
    pred_payload = _load_json(predictions_path)
    if "threshold_used" not in pred_payload:
        raise AssertionError("prediction.json must contain threshold_used")
    threshold_used = pred_payload["threshold_used"]
    if threshold_used is None:
        raise AssertionError("threshold_used must not be null")
    threshold_used = float(threshold_used)
    if not (0.0 <= threshold_used <= 1.0):
        raise AssertionError("threshold_used must be between 0 and 1")
    if abs(threshold_used - best_threshold) > 1e-8:
        raise AssertionError("threshold_used must match best_threshold")

    print("OK: thresholding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
