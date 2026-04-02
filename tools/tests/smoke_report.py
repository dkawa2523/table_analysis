#!/usr/bin/env python3
"""Smoke test for pipeline report generation."""

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


def _make_toy_csv(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(1)
    n = 160
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(5, 2, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["target"] = 0.4 * df["num1"] - 0.2 * df["num2"] + (df["cat"] == "b").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_report"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy.csv"
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
    raw_dataset_id = _load_json(ds_dir / "out.json").get("raw_dataset_id")
    if not raw_dataset_id:
        raise AssertionError("dataset_register out.json must contain raw_dataset_id")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=pipeline",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.dataset_path={csv_path}",
            f"data.raw_dataset_id={raw_dataset_id}",
            "data.target_column=target",
        ],
        cwd=repo,
    )

    pipeline_dir = out_root / "99_pipeline"
    report_path = pipeline_dir / "report.md"
    _must_exist(report_path)
    report_text = report_path.read_text(encoding="utf-8")
    for token in ("# Pipeline Summary", "## Conclusion", "## Recommendation"):
        if token not in report_text:
            raise AssertionError(f"report.md missing section: {token}")

    pipeline_run = _load_json(pipeline_dir / "pipeline_run.json")
    leaderboard_ref = pipeline_run.get("leaderboard_ref") or {}
    leaderboard_dir = Path(leaderboard_ref.get("run_dir", ""))
    rec_path = leaderboard_dir / "recommendation.json"
    if rec_path.exists():
        recommendation = _load_json(rec_path)
        rec_id = recommendation.get("recommended_model_id")
        if rec_id and str(rec_id) not in report_text:
            raise AssertionError("report.md must mention recommended_model_id")
        infer_id = recommendation.get("infer_model_id")
        if infer_id and str(infer_id) not in report_text:
            raise AssertionError("report.md must mention infer_model_id")

    print("OK: report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
