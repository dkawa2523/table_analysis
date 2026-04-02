#!/usr/bin/env python3
"""Smoke test for chunked batch infer."""

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


def _make_large_csv(path: Path, *, n_rows: int) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n" + str(exc)
        )

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n_rows),
            "num2": rng.normal(5, 2, size=n_rows),
            "cat": rng.choice(["a", "b", "c"], size=n_rows),
        }
    )
    df["target"] = 0.2 * df["num1"] - 0.3 * df["num2"] + (df["cat"] == "b").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_infer_csv(path: Path, *, source_csv: Path) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas is required for smoke tests. Install requirements/base.txt first.\n" + str(exc)
        )

    df = pd.read_csv(source_csv)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_batch_chunked"
    tmp = smoke_root / "tmp"
    out_train = smoke_root / "out_train"
    out_infer = smoke_root / "out_infer"
    tmp.mkdir(parents=True, exist_ok=True)
    out_train.mkdir(parents=True, exist_ok=True)
    out_infer.mkdir(parents=True, exist_ok=True)

    n_rows = 20000
    csv_path = tmp / "large.csv"
    infer_csv = tmp / "infer.csv"
    _make_large_csv(csv_path, n_rows=n_rows)
    _make_infer_csv(infer_csv, source_csv=csv_path)

    py = sys.executable

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=dataset_register",
            "run.clearml.enabled=false",
            f"run.output_dir={out_train}",
            f"data.dataset_path={csv_path}",
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
            f"data.dataset_path={csv_path}",
            "data.target_column=target",
            "group/preprocess=stdscaler_ohe",
            "data.split.strategy=random",
            "data.split.seed=42",
        ],
        cwd=repo,
    )

    pp_out = _load_json(out_train / "02_preprocess" / "out.json")
    processed_ref = pp_out["processed_dataset_id"]

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
        ],
        cwd=repo,
    )

    tr_out = _load_json(out_train / "03_train_model" / "out.json")
    model_ref = tr_out["model_id"]

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
            "infer.batch.chunk_size=5000",
            "infer.batch.output_format=csv",
            "infer.batch.write_mode=overwrite",
        ],
        cwd=repo,
    )

    inf_dir = out_infer / "04_infer"
    pred_path = inf_dir / "predictions.csv"
    _must_exist(pred_path)

    inf_out = _load_json(inf_dir / "out.json")
    chunked = inf_out.get("chunked")
    if not isinstance(chunked, dict):
        raise AssertionError("infer out.json must contain chunked info")
    if chunked.get("chunk_size") != 5000:
        raise AssertionError("chunked.chunk_size must be 5000")
    if chunked.get("rows") != n_rows:
        raise AssertionError(f"chunked.rows must be {n_rows}")
    if chunked.get("errors_count") != 0:
        raise AssertionError("chunked.errors_count must be 0 for valid input")

    line_count = _count_lines(pred_path)
    if line_count != n_rows + 1:
        raise AssertionError("predictions.csv line count mismatch")

    print("OK: chunked batch infer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
