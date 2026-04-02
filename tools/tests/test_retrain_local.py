#!/usr/bin/env python3
"""Local test for retrain orchestration."""

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
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_python_env(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n" + str(exc)
        )

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(5, 2, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["target"] = 0.2 * df["num1"] - 0.15 * df["num2"] + (df["cat"] == "b").astype(float) + rng.normal(
        0, 0.1, size=n
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    work_root = repo / "work" / "_test_retrain"
    tmp = work_root / "tmp"
    out_root = work_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy.csv"
    _make_toy_csv(csv_path)

    usecase_id = "RetrainLocal"

    py = sys.executable

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=dataset_register",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"run.usecase_id={usecase_id}",
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
            f"run.usecase_id={usecase_id}",
            f"data.dataset_path={csv_path}",
            "data.target_column=target",
            "group/preprocess=stdscaler_ohe",
            "data.split.strategy=random",
            "data.split.seed=42",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    processed_dataset_id = pp_out["processed_dataset_id"]

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=train_model",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"run.usecase_id={usecase_id}",
            f"data.processed_dataset_id={processed_dataset_id}",
            "eval.primary_metric=rmse",
            "group/model=ridge",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    tr_out = _load_json(tr_dir / "out.json")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=retrain",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"run.usecase_id={usecase_id}",
            f"data.dataset_path={csv_path}",
            "data.target_column=target",
        ],
        cwd=repo,
    )
    retrain_dir = out_root / "08_retrain"
    _check_common(retrain_dir)
    _must_exist(retrain_dir / "retrain_decision.json")
    _must_exist(retrain_dir / "retrain_run.json")
    _must_exist(retrain_dir / "retrain_summary.md")

    decision = _load_json(retrain_dir / "retrain_decision.json")
    decision_action = (decision.get("decision") or {}).get("action")
    if decision_action != "select_model":
        raise AssertionError(f"unexpected decision action: {decision_action}")
    if not decision.get("challenger_model_ref"):
        raise AssertionError("retrain decision missing challenger_model_ref")

    print("OK: retrain local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
