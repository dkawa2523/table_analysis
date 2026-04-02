#!/usr/bin/env python3
"""Regression test for data quality gate behavior (warn/fail)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], *, cwd: Path, expect_fail: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if expect_fail:
        if proc.returncode == 0:
            raise AssertionError(f"Expected failure but command succeeded:\n$ {' '.join(cmd)}")
        return proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode})\n$ {' '.join(cmd)}\n\n{proc.stdout}"
        )
    return proc.stdout


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_toy_csv(path: Path, *, exact_leak: bool) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for tests. Install requirements/base.txt first.\n"
            + str(exc)
        )

    rng = np.random.default_rng(7)
    n = 120
    base = rng.normal(0, 1, size=n)
    target = base + rng.normal(0, 0.1, size=n)
    if exact_leak:
        leak = target
    else:
        leak = target + rng.normal(0, 1e-3, size=n)
    df = pd.DataFrame(
        {
            "record_id": list(range(n)),
            "cat_high": [f"cat_{i}" for i in range(n)],
            "feature": base,
            "leak_feature": leak,
            "target": target,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "work" / "_quality_gate"
    tmp = root / "tmp"
    out_warn = root / "out_warn"
    out_pre = root / "out_preprocess"
    out_fail = root / "out_fail"
    tmp.mkdir(parents=True, exist_ok=True)
    out_warn.mkdir(parents=True, exist_ok=True)
    out_pre.mkdir(parents=True, exist_ok=True)
    out_fail.mkdir(parents=True, exist_ok=True)

    warn_csv = tmp / "quality_warn.csv"
    fail_csv = tmp / "quality_fail.csv"
    _make_toy_csv(warn_csv, exact_leak=False)
    _make_toy_csv(fail_csv, exact_leak=True)

    py = sys.executable
    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=dataset_register",
            "run.clearml.enabled=false",
            f"run.output_dir={out_warn}",
            f"data.dataset_path={warn_csv}",
            "data.target_column=target",
            "data.quality.mode=warn",
        ],
        cwd=repo,
    )
    ds_dir = out_warn / "01_dataset_register"
    quality_path = ds_dir / "data_quality.json"
    md_path = ds_dir / "data_quality.md"
    if not quality_path.exists() or not md_path.exists():
        raise AssertionError("data_quality artifacts missing for warn run.")

    quality = _load_json(quality_path)
    if quality.get("quality_status") != "warn":
        raise AssertionError(f"Expected quality_status=warn, got {quality.get('quality_status')}")
    if int(quality.get("quality_issue_count") or 0) <= 0:
        raise AssertionError("quality_issue_count should be > 0 for warn data.")
    if not quality.get("id_like_columns"):
        raise AssertionError("Expected id_like_columns to be detected.")
    if not quality.get("high_cardinality_columns"):
        raise AssertionError("Expected high_cardinality_columns to be detected.")

    _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=preprocess",
            "run.clearml.enabled=false",
            f"run.output_dir={out_pre}",
            f"data.dataset_path={warn_csv}",
            "data.target_column=target",
            "data.quality.mode=warn",
        ],
        cwd=repo,
    )
    pp_dir = out_pre / "02_preprocess"
    if not (pp_dir / "data_quality.json").exists():
        raise AssertionError("preprocess must output data_quality.json")
    if not (pp_dir / "data_quality.md").exists():
        raise AssertionError("preprocess must output data_quality.md")

    output = _run(
        [
            py,
            "-m",
            "tabular_analysis.cli",
            "task=dataset_register",
            "run.clearml.enabled=false",
            f"run.output_dir={out_fail}",
            f"data.dataset_path={fail_csv}",
            "data.target_column=target",
            "data.quality.mode=fail",
        ],
        cwd=repo,
        expect_fail=True,
    )
    if "data_quality gate failed" not in output:
        raise AssertionError("Expected failure message for quality gate in fail mode.")

    print("OK: data quality gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
