#!/usr/bin/env python3
"""Smoke test for enhanced drift reporting."""

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
            "num1": rng.normal(3.0, 1.4, size=n),
            "num2": rng.normal(9.0, 3.0, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n, p=[0.05, 0.15, 0.8]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_drift_enhanced"
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
            "monitor.drift.sample_n=80",
            "monitor.drift.metrics=[psi,ks]",
        ],
        cwd=repo,
    )

    train_dir = out_train / "03_train_model"
    train_profile_path = train_dir / "train_profile.json"
    _must_exist(train_profile_path)
    train_profile = _load_json(train_profile_path)
    if not train_profile.get("sampling"):
        raise AssertionError("train_profile.json must include sampling metadata")
    if "metrics" not in (train_profile.get("settings") or {}):
        raise AssertionError("train_profile.json missing settings.metrics")

    train_out = _load_json(train_dir / "out.json")
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
            "monitor.drift.enabled=true",
            "monitor.drift.sample_n=60",
            "monitor.drift.metrics=[psi,ks]",
            "monitor.drift.alert_thresholds.psi=0.05",
        ],
        cwd=repo,
    )

    infer_dir = out_infer / "04_infer"
    drift_report_path = infer_dir / "drift_report.json"
    drift_report_md_path = infer_dir / "drift_report.md"
    summary_path = infer_dir / "summary.md"
    _must_exist(drift_report_path)
    _must_exist(drift_report_md_path)
    _must_exist(summary_path)

    infer_out = _load_json(infer_dir / "out.json")
    if infer_out.get("drift_report_path") != str(drift_report_path):
        raise AssertionError("infer out.json must include drift_report_path")
    if infer_out.get("drift_alert") is not True:
        raise AssertionError("infer out.json must include drift_alert=true when threshold exceeded")

    drift_report = _load_json(drift_report_path)
    if "infer_profile" not in drift_report:
        raise AssertionError("drift_report.json must include infer_profile")
    if "sampling" not in drift_report:
        raise AssertionError("drift_report.json must include sampling info")
    metrics = [str(item).lower() for item in (drift_report.get("metrics") or [])]
    if "psi" not in metrics or "ks" not in metrics:
        raise AssertionError("drift_report.json must include psi and ks metrics")
    numeric = drift_report.get("numeric") or {}
    if not numeric:
        raise AssertionError("drift_report.json must include numeric drift entries")
    entry = next(iter(numeric.values()))
    if "ks" not in entry:
        raise AssertionError("numeric drift entries must include ks when enabled")

    summary_text = summary_path.read_text(encoding="utf-8")
    if "Drift Summary" not in summary_text:
        raise AssertionError("summary.md must include drift summary section")

    print("OK: enhanced drift report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
