#!/usr/bin/env python3
"""Smoke test for data quality outputs."""

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
    except Exception as e:
        raise RuntimeError(
            "pandas/numpy are required for smoke tests. Install requirements/base.txt first.\n" + str(e)
        )

    rng = np.random.default_rng(42)
    n = 50
    base = rng.normal(0, 1, size=n)
    target = 2.0 * base + rng.normal(0, 0.1, size=n)
    leak = target + rng.normal(0, 1e-4, size=n)
    df = pd.DataFrame(
        {
            "num": base,
            "noise": rng.normal(0, 1, size=n),
            "leak_feature": leak,
            "cat": rng.choice(["a", "b", None], size=n),
            "target": target,
        }
    )
    df.loc[0, "noise"] = None
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]

    smoke_root = repo / "work" / "_smoke_quality"
    tmp = smoke_root / "tmp"
    out_root = smoke_root / "out"
    tmp.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = tmp / "toy_quality.csv"
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
    _must_exist(ds_dir / "data_quality.json")
    _must_exist(ds_dir / "data_quality.md")

    quality = _load_json(ds_dir / "data_quality.json")
    if "duplicates_count" not in quality:
        raise AssertionError("data_quality.json must include duplicates_count")
    if "leak_suspects" not in quality:
        raise AssertionError("data_quality.json must include leak_suspects")

    md_text = (ds_dir / "data_quality.md").read_text(encoding="utf-8")
    if "duplicates_count" not in md_text:
        raise AssertionError("data_quality.md must mention duplicates_count")
    if "leak_suspects" not in md_text:
        raise AssertionError("data_quality.md must mention leak_suspects")

    print("OK: data_quality outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
