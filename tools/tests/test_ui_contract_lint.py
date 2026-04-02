#!/usr/bin/env python3
"""Contract lint test (expected pass/fail paths)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str], *, cwd: Path, expect_ok: bool = True) -> None:
    proc = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if expect_ok and proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode})\n$ {' '.join(cmd)}\n\n{proc.stdout}")
    if not expect_ok and proc.returncode == 0:
        raise RuntimeError(f"Expected failure but command succeeded.\n$ {' '.join(cmd)}\n\n{proc.stdout}")


def _ensure_smoke_outputs(repo: Path) -> Path:
    smoke_root = repo / "work" / "_smoke"
    out_root = smoke_root / "out"
    tmp_root = smoke_root / "tmp"
    probe = out_root / "03_train_model" / "manifest.json"
    if probe.exists():
        return out_root
    if out_root.exists():
        shutil.rmtree(out_root)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    py = sys.executable
    tests_dir = repo / "tools" / "tests"
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
    return out_root


def _clone_stage_dir(stage_src: Path, tmp_root: Path, name: str) -> Path:
    stage_dst = tmp_root / name
    if stage_dst.exists():
        shutil.rmtree(stage_dst)
    shutil.copytree(stage_src, stage_dst)
    return stage_dst


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = _ensure_smoke_outputs(repo)
    smoke_root = repo / "work" / "_smoke"
    tmp_root = smoke_root / "tmp" / "ui_contract_lint"

    py = sys.executable
    lint_cmd = [py, "-m", "tabular_analysis.ops.ui_contract_lint"]

    _run(
        [
            *lint_cmd,
            "--run-root",
            str(out_root),
            "--mode",
            "fail",
        ],
        cwd=repo,
    )

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    stage_src = out_root / "03_train_model"
    stage_dst = _clone_stage_dir(stage_src, tmp_root, "03_train_model_missing_manifest")
    manifest_path = stage_dst / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    _run(
        [
            *lint_cmd,
            "--run-dir",
            str(stage_dst),
            "--mode",
            "fail",
        ],
        cwd=repo,
        expect_ok=False,
    )

    stage_dst = _clone_stage_dir(stage_src, tmp_root, "03_train_model_missing_out_key")
    out_path = stage_dst / "out.json"
    out_payload = json.loads(out_path.read_text(encoding="utf-8"))
    out_payload.pop("model_id", None)
    _write_json(out_path, out_payload)

    _run(
        [
            *lint_cmd,
            "--run-dir",
            str(stage_dst),
            "--mode",
            "fail",
        ],
        cwd=repo,
        expect_ok=False,
    )

    stage_dst = _clone_stage_dir(stage_src, tmp_root, "03_train_model_process_mismatch")
    manifest_path = stage_dst / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["process"] = "pipeline"
    _write_json(manifest_path, manifest_payload)

    _run(
        [
            *lint_cmd,
            "--run-dir",
            str(stage_dst),
            "--mode",
            "fail",
        ],
        cwd=repo,
        expect_ok=False,
    )

    print("OK: ui_contract_lint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
