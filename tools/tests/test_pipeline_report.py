#!/usr/bin/env python3
"""Test pipeline report outputs (markdown/json/links)."""

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


def _make_toy_csv(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for pipeline report tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(7)
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
    smoke_root = repo / "work" / "_smoke_pipeline_report"
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
            "task=pipeline",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.dataset_path={csv_path}",
            "data.target_column=target",
            "pipeline.model_variants=[ridge,lasso]",
        ],
        cwd=repo,
    )

    pipeline_dir = out_root / "99_pipeline"
    report_path = pipeline_dir / "report.md"
    report_json_path = pipeline_dir / "report.json"
    report_links_path = pipeline_dir / "report_links.json"
    run_summary_path = pipeline_dir / "run_summary.json"
    pipeline_run_path = pipeline_dir / "pipeline_run.json"
    _must_exist(report_path)
    _must_exist(report_json_path)
    _must_exist(report_links_path)
    _must_exist(run_summary_path)
    _must_exist(pipeline_run_path)

    report_text = report_path.read_text(encoding="utf-8")
    for token in ("# Pipeline Summary", "## Conclusion", "## Recommendation", "## Comparability"):
        if token not in report_text:
            raise AssertionError(f"report.md missing section: {token}")
    if "## Selection" not in report_text:
        raise AssertionError("report.md missing Selection section")

    report_payload = _load_json(report_json_path)
    pipeline_run = _load_json(pipeline_run_path)
    run_summary = _load_json(run_summary_path)
    if not report_payload.get("grid_run_id"):
        raise AssertionError("report.json missing grid_run_id")
    if pipeline_run.get("status") != "completed":
        raise AssertionError(f"pipeline_run.json must record completed status for local run: {pipeline_run}")
    if run_summary.get("status") != pipeline_run.get("status"):
        raise AssertionError("run_summary.json must mirror pipeline_run.json status")
    if report_payload.get("status") != pipeline_run.get("status"):
        raise AssertionError("report.json status must follow pipeline_run.json")
    summary = report_payload.get("summary") or {}
    selection = report_payload.get("selection") or {}
    recommended_id = summary.get("recommended_model_id")
    if not recommended_id:
        raise AssertionError("report.json missing recommended_model_id")
    infer_key = summary.get("recommended_infer_key")
    infer_value = summary.get("recommended_infer_value")
    if not infer_key or not infer_value:
        raise AssertionError(f"report.json missing operator-facing infer selection: {summary}")
    infer_assignment = summary.get("recommended_infer_assignment")
    if infer_assignment != f"{infer_key}={infer_value}":
        raise AssertionError(f"report.json must expose copy-paste friendly infer assignment: {summary}")
    selector_values = {
        "infer.model_id": summary.get("infer_model_id"),
        "infer.train_task_id": summary.get("infer_train_task_id"),
    }
    if infer_key not in selector_values:
        raise AssertionError(f"report.json exposed unexpected infer selector: {infer_key}")
    if selector_values[infer_key] != infer_value:
        raise AssertionError(
            "report.json operator-facing infer selection must match the underlying summary payload"
        )
    if str(recommended_id) not in report_text:
        raise AssertionError("report.md must mention recommended_model_id")
    if str(infer_key) not in report_text or str(infer_value) not in report_text:
        raise AssertionError("report.md must mention operator-facing infer selection")
    if str(infer_assignment) not in report_text:
        raise AssertionError("report.md must expose copy-paste friendly infer assignment")
    comparability = report_payload.get("comparability") or {}
    if "split_hash" not in comparability:
        raise AssertionError("report.json missing comparability.split_hash")
    if summary.get("completed_jobs") != pipeline_run.get("completed_jobs"):
        raise AssertionError("report.json completed_jobs must mirror pipeline_run.json")
    if summary.get("requested_jobs") != pipeline_run.get("requested_jobs"):
        raise AssertionError("report.json requested_jobs must mirror pipeline_run.json")
    if summary.get("disabled_jobs") != pipeline_run.get("disabled_jobs"):
        raise AssertionError("report.json disabled_jobs must mirror pipeline_run.json")
    if summary.get("ranked_candidates") is None:
        raise AssertionError("report.json missing summary.ranked_candidates")
    if "active_model_variants" not in selection:
        raise AssertionError("report.json missing selection.active_model_variants")
    if "ranked_candidates" not in report_text:
        raise AssertionError("report.md must expose ranked_candidates")

    links = _load_json(report_links_path)
    pipeline_link = links.get("pipeline") or {}
    if not pipeline_link.get("run_dir"):
        raise AssertionError("report_links.json missing pipeline.run_dir")
    dataset_link = links.get("dataset_register") or {}
    if dataset_link.get("run_dir") and not Path(dataset_link["run_dir"]).exists():
        raise AssertionError("dataset_register run_dir does not exist")
    leaderboard_link = links.get("leaderboard") or {}
    if leaderboard_link.get("run_dir") and not Path(leaderboard_link["run_dir"]).exists():
        raise AssertionError("leaderboard run_dir does not exist")

    print("OK: pipeline report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
