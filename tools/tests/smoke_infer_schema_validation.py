#!/usr/bin/env python3
"""Smoke test for infer schema validation (warn/strict)."""

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


def _run_expect_fail(cmd: List[str], *, cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode == 0:
        raise RuntimeError(f"Command unexpectedly succeeded\n$ {' '.join(cmd)}\n\n{p.stdout}")
    return p.stdout


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
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n" + str(exc)
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
    df["target"] = 0.3 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "b").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_missing_input(path: Path, source_csv: Path, *, missing_column: str) -> None:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas is required for smoke tests. Install requirements/base.txt first.\n" + str(exc)
        )

    df = pd.read_csv(source_csv)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    if missing_column in df.columns:
        df = df.drop(columns=[missing_column])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    smoke_root = repo / "work" / "_smoke_schema_validation"
    tmp = smoke_root / "tmp"
    out_train = smoke_root / "out_train"
    out_warn = smoke_root / "out_warn"
    out_strict = smoke_root / "out_strict"
    tmp.mkdir(parents=True, exist_ok=True)
    out_train.mkdir(parents=True, exist_ok=True)
    out_warn.mkdir(parents=True, exist_ok=True)
    out_strict.mkdir(parents=True, exist_ok=True)

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

    missing_csv = tmp / "infer_missing.csv"
    _make_missing_input(missing_csv, csv_path, missing_column="num2")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=infer",
            "run.clearml.enabled=false",
            f"run.output_dir={out_warn}",
            f"infer.model_id={model_ref}",
            "infer.mode=single",
            f"infer.input_path={missing_csv}",
            "infer.validation.mode=warn",
        ],
        cwd=repo,
    )

    warn_dir = out_warn / "04_infer"
    _must_exist(warn_dir / "errors.json")
    _must_exist(warn_dir / "errors.csv")
    warn_out = _load_json(warn_dir / "out.json")
    if "schema_validation" not in warn_out:
        raise AssertionError("infer out.json must contain schema_validation")
    if warn_out["schema_validation"].get("mode") != "warn":
        raise AssertionError("infer schema_validation.mode must be warn")
    if "errors_path" not in warn_out:
        raise AssertionError("infer out.json must contain errors_path on warnings")

    _run_expect_fail(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=infer",
            "run.clearml.enabled=false",
            f"run.output_dir={out_strict}",
            f"infer.model_id={model_ref}",
            "infer.mode=single",
            f"infer.input_path={missing_csv}",
            "infer.validation.mode=strict",
        ],
        cwd=repo,
    )

    strict_dir = out_strict / "04_infer"
    _must_exist(strict_dir / "errors.json")
    _must_exist(strict_dir / "errors.csv")

    print("OK: infer schema validation (warn/strict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
