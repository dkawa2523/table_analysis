#!/usr/bin/env python3
"""Local-mode smoke tests for probability calibration."""

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


def _make_classification_csv(path: Path, *, target_path: Path, n_classes: int, seed: int) -> None:
    try:
        import pandas as pd
        from sklearn.datasets import make_classification
    except Exception as exc:
        raise RuntimeError(
            "pandas/scikit-learn are required for smoke tests. "
            "Install requirements/base.txt first.\n"
            + str(exc)
        )

    X, y = make_classification(
        n_samples=220,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    columns = [f"num{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    df.drop(columns=["target"]).to_csv(target_path, index=False)


def _run_flow(repo: Path, *, name: str, n_classes: int, seed: int) -> None:
    flow_root = repo / "work" / "_smoke_calibration" / name
    tmp = flow_root / "tmp"
    out_root = flow_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / f"toy_{name}.csv"
    infer_path = tmp / f"toy_{name}_infer.csv"
    _make_classification_csv(csv_path, target_path=infer_path, n_classes=n_classes, seed=seed)

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
            "eval.primary_metric=logloss",
            "eval.direction=auto",
            "eval.calibration.enabled=true",
            "eval.calibration.method=sigmoid",
            "eval.calibration.mode=prefit",
            "group/model=logistic_regression",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    _must_exist(tr_dir / "calibration_reliability.png")
    _must_exist(tr_dir / "calibration_report.json")

    tr_out = _load_json(tr_dir / "out.json")
    calibration = tr_out.get("calibration")
    if not isinstance(calibration, dict) or not calibration.get("enabled"):
        raise AssertionError("train_model out.json must contain calibration.enabled=true")
    if calibration.get("method") != "sigmoid":
        raise AssertionError("train_model out.json must contain calibration.method=sigmoid")
    if calibration.get("mode") != "prefit":
        raise AssertionError("train_model out.json must contain calibration.mode=prefit")

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
            f"infer.input_path={infer_path}",
        ],
        cwd=repo,
    )
    inf_dir = out_root / "04_infer"
    _check_common(inf_dir)
    inf_out = _load_json(inf_dir / "out.json")
    inf_calibration = inf_out.get("calibration")
    if not isinstance(inf_calibration, dict) or not inf_calibration.get("calibrated_proba"):
        raise AssertionError("infer out.json must contain calibration.calibrated_proba=true")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    _run_flow(repo, name="binary", n_classes=2, seed=7)
    _run_flow(repo, name="multiclass", n_classes=3, seed=11)

    print("OK: calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
