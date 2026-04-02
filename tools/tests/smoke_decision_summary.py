#!/usr/bin/env python3
"""Smoke test for model card and decision summary outputs."""

from __future__ import annotations

import argparse
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

    rng = np.random.default_rng(7)
    n = 180
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(2, 1.2, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["target"] = 0.5 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "a").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_decision_summary"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_decision.csv"
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
            "group/model=ridge",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    model_card_path = tr_dir / "model_card.md"
    _must_exist(model_card_path)
    if "# Model Card" not in model_card_path.read_text(encoding="utf-8"):
        raise AssertionError("model_card.md must include a title")

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
    decision_path = lb_dir / "decision_summary.md"
    _must_exist(decision_path)
    if "# Decision Summary" not in decision_path.read_text(encoding="utf-8"):
        raise AssertionError("decision_summary.md must include a title")
    decision_json = _load_json(lb_dir / "decision_summary.json")
    recommended = decision_json.get("recommended") or {}
    if "infer_model_id" not in recommended:
        raise AssertionError("decision_summary.json must contain infer_model_id")

    print("OK: decision summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
