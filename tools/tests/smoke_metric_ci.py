#!/usr/bin/env python3
"""Smoke test for bootstrap metric confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode})\n$ {' '.join(cmd)}\n\n{proc.stdout}")
    return proc.stdout


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

    rng = np.random.default_rng(11)
    n = 220
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(3, 1.5, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["target"] = (
        0.3 * df["num1"] - 0.2 * df["num2"] + (df["cat"] == "c").astype(float)
    ) + rng.normal(0, 0.3, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_metric_ci"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_ci.csv"
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
            "group/split=random",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    processed_ref = pp_out.get("processed_dataset_id")
    if not processed_ref:
        raise AssertionError("preprocess out.json must contain processed_dataset_id")

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
            "eval.ci.enabled=true",
            "eval.ci.n_boot=40",
            "eval.ci.alpha=0.1",
            "eval.ci.seed=0",
            "group/model=ridge",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    metrics_ci = _load_json(tr_dir / "metrics_ci.json")
    interval = metrics_ci.get("primary_metric") or {}
    for key in ("low", "mid", "high"):
        if interval.get(key) is None:
            raise AssertionError(f"metrics_ci.json missing primary_metric.{key}")
    tr_out = _load_json(tr_dir / "out.json")
    if not tr_out.get("primary_metric_ci"):
        raise AssertionError("train_model out.json must contain primary_metric_ci")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=leaderboard",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            "eval.task_type=regression",
            "eval.primary_metric=rmse",
            "eval.direction=minimize",
            f"leaderboard.train_task_ids=['{tr_dir}']",
        ],
        cwd=repo,
    )
    lb_dir = out_root / "05_leaderboard"
    _check_common(lb_dir)
    leaderboard_path = lb_dir / "leaderboard.csv"
    _must_exist(leaderboard_path)
    with leaderboard_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise AssertionError("leaderboard.csv must contain at least one row")
    for key in ("primary_metric_ci_low", "primary_metric_ci_mid", "primary_metric_ci_high"):
        if key not in row:
            raise AssertionError(f"leaderboard.csv missing column: {key}")
        if row[key] == "":
            raise AssertionError(f"leaderboard.csv {key} must not be empty")

    print("OK: metric CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
