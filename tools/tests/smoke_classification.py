#!/usr/bin/env python3
"""Local-mode smoke tests for classification flow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _python_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str((repo / "src").resolve())
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    return env


def _run(cmd: List[str], *, cwd: Path) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_python_env(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
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
    score = 0.5 * df["num1"] - 0.2 * df["num2"] + (df["cat"] == "b").astype(float)
    df["target"] = (score > score.median()).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument(
        "--until",
        default="dataset_register",
        choices=["dataset_register", "preprocess", "train_model", "infer", "leaderboard"],
    )
    ap.add_argument("--split", default="random", choices=["random", "group", "time", "stratified"])
    ap.add_argument("--model", default="logistic_regression")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_classification"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_classification.csv"
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

    if args.until == "dataset_register":
        print("OK: dataset_register")
        return 0

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

    if args.until == "preprocess":
        print("OK: preprocess")
        return 0

    # 3) train_model (classification)
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
            "eval.primary_metric=accuracy",
            "eval.direction=maximize",
            f"group/model={args.model}",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    tr_out = _load_json(tr_dir / "out.json")
    for k in ["model_id", "primary_metric", "best_score", "task_type", "n_classes"]:
        if k not in tr_out:
            raise AssertionError(f"train_model out.json must contain {k}")

    if args.until == "train_model":
        print("OK: train_model")
        return 0

    # 4) infer (single)
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
            "infer.mode=single",
        ],
        cwd=repo,
    )
    inf_dir = out_root / "04_infer"
    _check_common(inf_dir)
    inf_out = _load_json(inf_dir / "out.json")
    predictions_path = Path(inf_out["predictions_path"])
    pred_payload = _load_json(predictions_path)
    for k in ["predicted_label", "predicted_proba"]:
        if k not in pred_payload:
            raise AssertionError(f"prediction.json must contain {k}")

    if args.until == "infer":
        print("OK: infer")
        return 0

    # 5) leaderboard
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=leaderboard",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            "eval.task_type=classification",
            "eval.primary_metric=accuracy",
            "eval.direction=maximize",
            f"leaderboard.train_task_ids=['{tr_dir}']",
        ],
        cwd=repo,
    )
    lb_dir = out_root / "05_leaderboard"
    _check_common(lb_dir)
    lb_out = _load_json(lb_dir / "out.json")
    for k in ["leaderboard_csv", "recommended_model_id"]:
        if k not in lb_out:
            raise AssertionError(f"leaderboard out.json must contain {k}")

    print("OK: leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
