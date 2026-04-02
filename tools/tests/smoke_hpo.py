#!/usr/bin/env python3
"""Local-mode smoke tests for pipeline HPO grid."""

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
    df["target"] = (
        0.3 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "b").astype(float)
    ) + rng.normal(0, 0.1, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_alpha(cfg_path: Path) -> float:
    _must_exist(cfg_path)
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as exc:
        raise RuntimeError("omegaconf is required for smoke_hpo tests.\n" + str(exc))
    cfg = OmegaConf.load(cfg_path)
    value = OmegaConf.select(cfg, "model_variant.params.alpha")
    if value is None:
        value = OmegaConf.select(cfg, "group.model.model_variant.params.alpha")
    if value is None:
        value = OmegaConf.select(cfg, "train.params.alpha")
    if value is None:
        raise AssertionError(f"alpha param missing in {cfg_path}")
    try:
        return float(value)
    except Exception as exc:
        raise AssertionError(f"train.params.alpha must be numeric: {value}") from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    smoke_root = repo / "work" / "_smoke_hpo"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_hpo.csv"
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
    ds_out = _load_json(ds_dir / "out.json")
    raw_dataset_id = ds_out.get("raw_dataset_id")
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
            "pipeline.grid.preprocess_variants=[stdscaler_ohe]",
            "pipeline.grid.model_variants=[ridge]",
            "pipeline.hpo.enabled=true",
            "pipeline.hpo.params.ridge.alpha=[0.1,1.0,10.0]",
            "eval.cv_folds=0",
        ],
        cwd=repo,
    )

    pl_dir = out_root / "99_pipeline"
    _check_common(pl_dir)
    pl_out = _load_json(pl_dir / "out.json")
    pipeline_run = pl_out.get("pipeline_run") or {}
    train_refs = pipeline_run.get("train_refs") or []
    if len(train_refs) < 3:
        raise AssertionError("pipeline must create at least 3 train runs for HPO grid")

    alphas = set()
    for ref in train_refs:
        run_dir = Path(ref.get("run_dir", ""))
        if not run_dir:
            raise AssertionError("train_refs must include run_dir in local mode")
        _check_common(run_dir)
        alpha = _load_alpha(run_dir / "config_resolved.yaml")
        alphas.add(round(alpha, 6))
        if not ref.get("hpo_run_id"):
            raise AssertionError("train_refs must include hpo_run_id when HPO is enabled")

    expected = {0.1, 1.0, 10.0}
    if alphas != expected:
        raise AssertionError(f"unexpected alphas: {sorted(alphas)} (expected {sorted(expected)})")

    leaderboard_ref = pipeline_run.get("leaderboard_ref") or {}
    lb_run_dir = Path(leaderboard_ref.get("run_dir", ""))
    if not lb_run_dir:
        raise AssertionError("leaderboard_ref.run_dir is missing")
    lb_out = _load_json(lb_run_dir / "out.json")
    if "recommended_model_id" not in lb_out:
        raise AssertionError("leaderboard out.json must contain recommended_model_id")

    print("OK: pipeline HPO grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
