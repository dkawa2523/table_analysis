#!/usr/bin/env python3
"""Exec policy coverage for pipeline plan mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], *, cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = repo / "work" / "_exec_policy" / "out"
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=pipeline",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            "pipeline.plan_only=true",
            "pipeline.grid.preprocess_variants=[stdscaler_ohe]",
            "pipeline.grid.model_variants=[ridge,lasso]",
            "pipeline.hpo.enabled=true",
            "pipeline.hpo.params.ridge.alpha=[0.1,1.0,10.0]",
            "exec_policy.limits.max_hpo_trials=2",
            "exec_policy.limits.max_jobs=2",
            "exec_policy.limits.max_models=1",
            "eval.cv_folds=0",
        ],
        cwd=repo,
    )

    pl_dir = out_root / "99_pipeline"
    _check_common(pl_dir)
    pl_out = _load_json(pl_dir / "out.json")
    pipeline_run = pl_out.get("pipeline_run") or {}

    if not pipeline_run.get("plan_only"):
        raise AssertionError("plan_only must be true in plan mode")

    planned = pipeline_run.get("planned_jobs")
    executed = pipeline_run.get("executed_jobs")
    skipped = pipeline_run.get("skipped_due_to_policy")
    if planned != 2:
        raise AssertionError(f"planned_jobs must be 2 (got {planned})")
    if executed != 0:
        raise AssertionError(f"executed_jobs must be 0 in plan_only (got {executed})")
    if skipped != 2:
        raise AssertionError(f"skipped_due_to_policy must be 2 (got {skipped})")

    grid = pipeline_run.get("grid") or {}
    if grid.get("max_jobs") != 2:
        raise AssertionError(f"grid.max_jobs must reflect exec_policy limit (got {grid.get('max_jobs')})")

    policy = pipeline_run.get("policy") or {}
    limits = policy.get("limits") or {}
    if limits.get("max_hpo_trials") != 2:
        raise AssertionError(
            f"policy.limits.max_hpo_trials must reflect exec_policy limit (got {limits.get('max_hpo_trials')})"
        )

    print("OK: exec_policy plan mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
