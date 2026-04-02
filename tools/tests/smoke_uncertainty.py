#!/usr/bin/env python3
"""Smoke test for regression conformal prediction intervals."""

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
    n = 240
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(3, 1.5, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["target"] = (
        0.2 * df["num1"] - 0.15 * df["num2"] + (df["cat"] == "b").astype(float)
    ) + rng.normal(0, 0.2, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="ridge")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_uncertainty"
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
            "data.split.strategy=random",
            "data.split.seed=42",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    processed_ref = pp_out.get("processed_dataset_id")
    if not processed_ref:
        raise AssertionError("preprocess out.json must contain processed_dataset_id")

    # 3) train_model with uncertainty
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
            "eval.uncertainty.enabled=true",
            "eval.uncertainty.alpha=0.1",
            f"group/model={args.model}",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    tr_out = _load_json(tr_dir / "out.json")
    uncertainty = tr_out.get("uncertainty") or {}
    if not uncertainty.get("q"):
        raise AssertionError("train_model out.json must contain uncertainty.q")

    # 4) infer (batch) with prediction intervals
    model_ref = tr_out.get("model_id")
    if not model_ref:
        raise AssertionError("train_model out.json must contain model_id")
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
            f"infer.input_path={csv_path}",
        ],
        cwd=repo,
    )
    inf_dir = out_root / "04_infer"
    _check_common(inf_dir)
    pred_path = inf_dir / "predictions.csv"
    _must_exist(pred_path)
    header = pred_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    if "pred_lower" not in header or "pred_upper" not in header:
        raise AssertionError("predictions.csv must contain pred_lower and pred_upper")

    print("OK: uncertainty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
