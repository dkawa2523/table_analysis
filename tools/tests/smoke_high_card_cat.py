#!/usr/bin/env python3
"""Smoke test for high-cardinality categorical encoding."""

from __future__ import annotations

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


def _make_toy_csv(path: Path) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(123)
    n = 600
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n),
            "num2": rng.normal(1, 2, size=n),
            "cat_high": [f"cat_{i}" for i in range(n)],
            "target": rng.normal(0, 1, size=n),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    smoke_root = repo / "work" / "_smoke_high_card_cat"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "high_card.csv"
    _make_toy_csv(csv_path)

    py = sys.executable
    n_features = 64
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
            "preprocess.categorical.encoding=hashing",
            f"preprocess.categorical.hashing.n_features={n_features}",
        ],
        cwd=repo,
    )

    pp_dir = out_root / "02_preprocess"
    _must_exist(pp_dir / "out.json")
    _must_exist(pp_dir / "processed_dataset.parquet")
    _must_exist(pp_dir / "categorical_encoding_report.json")

    report = _load_json(pp_dir / "categorical_encoding_report.json")
    col_report = report.get("columns", {}).get("cat_high", {})
    if col_report.get("encoding") != "hashing":
        raise AssertionError("categorical_encoding_report.json must record hashing for cat_high")

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required for smoke tests.") from exc

    processed = pd.read_parquet(pp_dir / "processed_dataset.parquet")
    expected_cols = 2 + n_features + 1
    if processed.shape[1] != expected_cols:
        raise AssertionError(
            f"processed_dataset.parquet has {processed.shape[1]} cols; expected {expected_cols}"
        )

    print("OK: high-card categorical encoding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
