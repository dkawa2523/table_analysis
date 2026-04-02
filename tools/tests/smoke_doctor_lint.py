#!/usr/bin/env python3
"""Smoke test for doctor contract lint."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str], *, cwd: Path) -> None:
    proc = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode})\n$ {' '.join(cmd)}\n\n{proc.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    tests_dir = repo / "tools" / "tests"
    smoke_root = repo / "work" / "_smoke"
    out_root = smoke_root / "out"
    tmp_root = smoke_root / "tmp"

    if out_root.exists():
        shutil.rmtree(out_root)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    py = sys.executable
    _run(
        [
            py,
            str(tests_dir / "smoke_local.py"),
            "--repo",
            str(repo),
            "--until",
            "pipeline",
        ],
        cwd=repo,
    )

    _run(
        [
            py,
            "-m",
            "tabular_analysis.doctor",
            "--lint-run",
            str(out_root),
            "--mode",
            "fail",
        ],
        cwd=repo,
    )

    print("OK: doctor lint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
