#!/usr/bin/env python3
"""Test env snapshot output in local mode."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tabular_analysis.platform_adapter_task import init_task_context


def _build_cfg(out_root: Path) -> SimpleNamespace:
    clearml = SimpleNamespace(enabled=False, execution="local", project_name=None, task_name=None)
    run = SimpleNamespace(usecase_id="test", output_dir=str(out_root), clearml=clearml)
    task = SimpleNamespace(stage="env_snapshot", name="env_snapshot", project_name="local")
    return SimpleNamespace(run=run, task=task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    root = repo / "work" / "_env_snapshot"
    if root.exists():
        shutil.rmtree(root)
    out_root = root / "out"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = _build_cfg(out_root)
    ctx = init_task_context(cfg, stage="00_env_snapshot", task_name="env_snapshot")

    env_path = ctx.output_dir / "env.json"
    freeze_path = ctx.output_dir / "pip_freeze.txt"

    if not env_path.exists():
        raise AssertionError(f"env.json missing: {env_path}")
    if not freeze_path.exists():
        raise AssertionError(f"pip_freeze.txt missing: {freeze_path}")

    payload = json.loads(env_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("env.json must be a JSON object")
    python_info = payload.get("python")
    platform_info = payload.get("platform")
    solution_info = payload.get("solution")
    if not isinstance(python_info, dict) or not python_info.get("version"):
        raise AssertionError("env.json must include python version")
    if not isinstance(platform_info, dict) or not platform_info.get("platform"):
        raise AssertionError("env.json must include platform info")
    if not isinstance(solution_info, dict) or not solution_info.get("version"):
        raise AssertionError("env.json must include solution version")

    freeze_text = freeze_path.read_text(encoding="utf-8").strip()
    if not freeze_text:
        raise AssertionError("pip_freeze.txt must not be empty")

    print("OK: env snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
