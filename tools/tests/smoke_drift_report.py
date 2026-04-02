#!/usr/bin/env python3
"""Smoke test for drift report generation."""

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


def _make_train_csv(path: Path) -> None:
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
            "num2": rng.normal(5, 2, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1]),
        }
    )
    df["target"] = 0.4 * df["num1"] - 0.2 * df["num2"] + (df["cat"] == "b").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_shifted_csv(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(99)
    n = 180
    df = pd.DataFrame(
        {
            "num1": rng.normal(2.0, 1.2, size=n),
            "num2": rng.normal(8.0, 2.5, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.1, 0.1, 0.8]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_drift"
    tmp = smoke_root / "tmp"
    out_train = smoke_root / "out_train"
    out_infer = smoke_root / "out_infer"
    tmp.mkdir(parents=True, exist_ok=True)
    out_train.mkdir(parents=True, exist_ok=True)
    out_infer.mkdir(parents=True, exist_ok=True)

    train_csv = tmp / "train.csv"
    infer_csv = tmp / "infer.csv"
    _make_train_csv(train_csv)
    _make_shifted_csv(infer_csv)

    py = sys.executable

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=dataset_register",
            "run.clearml.enabled=false",
            f"run.output_dir={out_train}",
            f"data.dataset_path={train_csv}",
            "data.target_column=target",
        ],
        cwd=repo,
    )

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=preprocess",
            "run.clearml.enabled=false",
            f"run.output_dir={out_train}",
            f"data.dataset_path={train_csv}",
            "data.target_column=target",
            "group/preprocess=stdscaler_ohe",
            "data.split.strategy=random",
            "data.split.seed=42",
        ],
        cwd=repo,
    )

    preprocess_out = _load_json(out_train / "02_preprocess" / "out.json")
    processed_ref = preprocess_out["processed_dataset_id"]

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=train_model",
            "run.clearml.enabled=false",
            f"run.output_dir={out_train}",
            f"data.processed_dataset_id={processed_ref}",
            "eval.primary_metric=rmse",
            "group/model=ridge",
            "monitor.drift.enabled=true",
        ],
        cwd=repo,
    )

    train_out = _load_json(out_train / "03_train_model" / "out.json")
    model_ref = train_out["model_id"]

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=infer",
            "run.clearml.enabled=false",
            f"run.output_dir={out_infer}",
            f"infer.model_id={model_ref}",
            "infer.mode=batch",
            f"infer.input_path={infer_csv}",
            "infer.drift.enabled=true",
            "infer.drift.psi_warn_threshold=0.1",
            "monitor.drift.enabled=true",
        ],
        cwd=repo,
    )

    infer_dir = out_infer / "04_infer"
    drift_report_path = infer_dir / "drift_report.json"
    drift_report_md_path = infer_dir / "drift_report.md"
    _must_exist(drift_report_path)
    _must_exist(drift_report_md_path)

    infer_out = _load_json(infer_dir / "out.json")
    if infer_out.get("drift_report_path") != str(drift_report_path):
        raise AssertionError("infer out.json must include drift_report_path")

    drift_report = _load_json(drift_report_path)
    summary = drift_report.get("summary") or {}
    if "psi_max" not in summary:
        raise AssertionError("drift_report.json missing summary.psi_max")
    if not drift_report.get("numeric"):
        raise AssertionError("drift_report.json must include numeric drift entries")

    print("OK: drift report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
