#!/usr/bin/env python3
"""Local-mode smoke tests for ml-solution-tabular-analysis.

目的
- Codex 開発タスクの verify を「compile だけ」にしない（中途半端完了を防止）
- ClearML が未設定でも (run.clearml.enabled=false) 最低限の I/O 契約を確認する

設計
- 小さな合成データを生成して、CLI をサブプロセスで段階的に実行する
- 各ステージの出力ディレクトリに `config_resolved.yaml` / `out.json` / `manifest.json` があることを検証する

使い方例
  python tools/tests/smoke_local.py --until dataset_register
  python tools/tests/smoke_local.py --until preprocess
  python tools/tests/smoke_local.py --until train_model
  python tools/tests/smoke_local.py --until infer
  python tools/tests/smoke_local.py --until leaderboard
  python tools/tests/smoke_local.py --until pipeline

NOTE: このスモークは "最小" の契約のみチェックします。
評価の厳密性（split_hashの一致等）は leaderboard 実装側の verify でより強く担保します。
"""

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
    """All tasks must emit these artifacts."""
    _must_exist(stage_dir / "config_resolved.yaml")
    _must_exist(stage_dir / "out.json")
    _must_exist(stage_dir / "manifest.json")


def _make_toy_csv(path: Path) -> None:
    """Generate a tiny but non-trivial tabular dataset."""
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n" + str(e)
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
    # simple target
    df["target"] = 0.3 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "b").astype(float) + rng.normal(0, 0.1, size=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument(
        "--until",
        default="dataset_register",
        choices=["dataset_register", "preprocess", "train_model", "infer", "leaderboard", "pipeline"],
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    # deterministic output under work/_smoke
    smoke_root = repo / "work" / "_smoke"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy.csv"
    _make_toy_csv(csv_path)

    py = sys.executable

    # 1) dataset_register
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
    if "raw_dataset_id" not in ds_out:
        raise AssertionError("dataset_register out.json must contain raw_dataset_id")

    if args.until == "dataset_register":
        print("OK: dataset_register")
        return 0

    # 2) preprocess
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
            "group/preprocess=stdscaler_ohe",
            "data.split.strategy=random",
            "data.split.seed=42",
        ],
        cwd=repo,
    )
    pp_dir = out_root / "02_preprocess"
    _check_common(pp_dir)
    pp_out = _load_json(pp_dir / "out.json")
    for k in ["processed_dataset_id", "split_hash", "recipe_hash"]:
        if k not in pp_out:
            raise AssertionError(f"preprocess out.json must contain {k}")

    if args.until == "preprocess":
        print("OK: preprocess")
        return 0

    # 3) train_model
    processed_ref = pp_out["processed_dataset_id"]
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=train_model",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.processed_dataset_id={processed_ref}",
            "eval.primary_metric=rmse",
            "group/model=ridge",
        ],
        cwd=repo,
    )
    tr_dir = out_root / "03_train_model"
    _check_common(tr_dir)
    tr_out = _load_json(tr_dir / "out.json")
    for k in ["model_id", "primary_metric", "best_score"]:
        if k not in tr_out:
            raise AssertionError(f"train_model out.json must contain {k}")

    if args.until == "train_model":
        print("OK: train_model")
        return 0

    # 4) infer (single)
    model_ref = tr_out["model_id"]
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=infer",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"infer.model_id={model_ref}",
            "infer.mode=single",
            # single mode: take first row of toy.csv internally or via config (implementation chooses)
        ],
        cwd=repo,
    )
    inf_dir = out_root / "04_infer"
    _check_common(inf_dir)
    inf_out = _load_json(inf_dir / "out.json")
    if "predictions_path" not in inf_out:
        raise AssertionError("infer out.json must contain predictions_path")

    if args.until == "infer":
        print("OK: infer")
        return 0

    # 5) leaderboard
    # Local mode: train_task_ids can be treated as "refs" (either ClearML Task ID or local dir)
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=leaderboard",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"leaderboard.train_task_ids=['{tr_dir}']",
        ],
        cwd=repo,
    )
    lb_dir = out_root / "05_leaderboard"
    _check_common(lb_dir)
    lb_out = _load_json(lb_dir / "out.json")
    for k in ["leaderboard_csv", "recommended_model_id"]:
        if k not in lb_out:
            raise AssertionError(f"leaderboard out.json must contain {k}")

    if args.until == "leaderboard":
        print("OK: leaderboard")
        return 0

    # 6) pipeline
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=pipeline",
            "run.clearml.enabled=false",
            f"run.output_dir={out_root}",
            f"data.dataset_path={csv_path}",
            f"data.raw_dataset_id={ds_out['raw_dataset_id']}",
            "data.target_column=target",
        ],
        cwd=repo,
    )
    pl_dir = out_root / "99_pipeline"
    _check_common(pl_dir)
    pl_out = _load_json(pl_dir / "out.json")
    if "pipeline_run" not in pl_out:
        raise AssertionError("pipeline out.json must contain pipeline_run")

    print("OK: pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
