#!/usr/bin/env python3
"""Test alerting file sink output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tabular_analysis.ops.alerting import emit_alert
from tabular_analysis.platform_adapter_task import TaskContext


def _load_last_json_line(path: Path) -> dict[str, object]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise AssertionError("alert log must contain at least one line")
    return json.loads(lines[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = repo / "work" / "_alerting" / "out"
    stage_dir = out_root / "00_alerting"
    stage_dir.mkdir(parents=True, exist_ok=True)

    log_path = stage_dir / "alerts.jsonl"
    if log_path.exists():
        log_path.unlink()

    cfg_disabled = OmegaConf.create(
        {"run": {"output_dir": str(out_root), "alerts": {"enabled": False}}}
    )
    ctx = TaskContext(
        task=None,
        project_name="local",
        task_name="alerting_test",
        output_dir=stage_dir,
    )
    emit_alert(
        "data_quality",
        "error",
        "Disabled alert",
        "should not write",
        {"_cfg": cfg_disabled, "_ctx": ctx},
    )
    if log_path.exists():
        raise AssertionError("alert log should not be created when alerts are disabled")

    cfg_enabled = OmegaConf.create(
        {
            "run": {
                "output_dir": str(out_root),
                "alerts": {"enabled": True, "sinks": {"file": {"enabled": True}}},
            }
        }
    )
    emit_alert(
        "data_quality",
        "error",
        "Data quality gate failed",
        "duplicates_count>0",
        {"_cfg": cfg_enabled, "_ctx": ctx, "duplicates_count": 3},
    )
    if not log_path.exists():
        raise AssertionError("alert log must be created when file sink is enabled")

    payload = _load_last_json_line(log_path)
    if payload.get("kind") != "data_quality":
        raise AssertionError("alert kind must be data_quality")
    if payload.get("severity") != "error":
        raise AssertionError("alert severity must be error")
    context = payload.get("context") or {}
    if isinstance(context, dict):
        if context.get("duplicates_count") != 3:
            raise AssertionError("alert context must include duplicates_count")
    else:
        raise AssertionError("alert context must be a dict")

    print("OK: alerting logfile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
