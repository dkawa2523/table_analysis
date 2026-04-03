#!/usr/bin/env python3
"""Unified verification runner for local development and CI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

Step = Tuple[str, List[str]]


def _repo_root(repo_arg: str | None) -> Path:
    if repo_arg:
        return Path(repo_arg).resolve()
    return Path(__file__).resolve().parents[2]


def _python_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str((repo / "src").resolve())
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    step_name: str,
    max_seconds: int,
    heartbeat_seconds: int,
) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        env=_python_env(cwd),
        stdout=None,
        stderr=None,
    )
    started = time.monotonic()
    next_heartbeat = started + max(1, heartbeat_seconds)
    while proc.poll() is None:
        now = time.monotonic()
        elapsed = int(now - started)
        if elapsed >= max_seconds:
            proc.kill()
            proc.wait(timeout=5)
            print(f"FAILED (timeout={max_seconds}s, step={step_name})")
            return 124
        if now >= next_heartbeat:
            print(f"[heartbeat] step={step_name} elapsed={elapsed}s")
            next_heartbeat += max(1, heartbeat_seconds)
        time.sleep(1)
    if proc.returncode != 0:
        print(f"FAILED (exit={proc.returncode}, step={step_name})")
    return proc.returncode


def _quick_steps(repo: Path, py: str) -> List[Step]:
    tests_dir = repo / "tools" / "tests"
    workspace_root = repo.parent
    cleanup_cmd = [
        py,
        str(repo / "tools" / "cleanup_repo.py"),
        "--repo",
        str(repo),
        "--check",
    ]
    cleanup_apply_cmd = [
        py,
        str(repo / "tools" / "cleanup_repo.py"),
        "--repo",
        str(repo),
        "--apply",
    ]
    steps: List[Step] = [
        ("cleanup_residue_check_start", cleanup_cmd),
        (
            "appledouble_check",
            [
                py,
                str(tests_dir / "check_appledouble.py"),
                "--root",
                str(workspace_root),
            ],
        ),
        (
            "broad_except_check",
            [
                py,
                str(tests_dir / "check_broad_excepts.py"),
                "--repo",
                str(repo),
            ],
        ),
        ("compileall", [py, "-m", "compileall", "-q", "src"]),
        (
            "smoke_ci_config",
            [
                py,
                str(tests_dir / "smoke_ci_config.py"),
                "--repo",
                str(repo),
            ],
        ),
        (
            "docs_paths_check",
            [
                py,
                str(tests_dir / "check_docs_paths.py"),
                "--repo",
                str(repo),
            ],
        ),
        (
            "platform_adapter_dataset_split",
            [
                py,
                str(tests_dir / "test_platform_adapter_dataset_split.py"),
                "--repo",
                str(repo),
            ],
        ),
        (
            "platform_adapter_core_split_shim",
            [
                py,
                str(tests_dir / "test_platform_adapter_core_split_shim.py"),
                "--repo",
                str(repo),
            ],
        ),
        (
            "smoke_local",
            [
                py,
                str(tests_dir / "smoke_local.py"),
                "--repo",
                str(repo),
                "--until",
                "pipeline",
            ],
        ),
        (
            "smoke_classification",
            [
                py,
                str(tests_dir / "smoke_classification.py"),
                "--repo",
                str(repo),
                "--until",
                "leaderboard",
            ],
        ),
        (
            "smoke_multiclass",
            [
                py,
                str(tests_dir / "smoke_multiclass.py"),
                "--repo",
                str(repo),
            ],
        ),
        ("smoke_high_card_cat", [py, str(tests_dir / "smoke_high_card_cat.py")]),
        (
            "check_optional_models",
            [
                py,
                str(tests_dir / "check_optional_models.py"),
                "--repo",
                str(repo),
                "--models",
                "lgbm,xgboost,catboost,tabpfn",
            ],
        ),
        ("smoke_hpo", [py, str(tests_dir / "smoke_hpo.py"), "--repo", str(repo)]),
        ("smoke_report", [py, str(tests_dir / "smoke_report.py"), "--repo", str(repo)]),
        ("smoke_plots", [py, str(tests_dir / "smoke_plots.py"), "--repo", str(repo)]),
        ("cleanup_residue_apply_end", cleanup_apply_cmd),
        ("cleanup_residue_check_end", cleanup_cmd),
    ]
    doctor_script = tests_dir / "smoke_doctor_lint.py"
    doctor_module = repo / "src" / "tabular_analysis" / "doctor.py"
    serving_script = tests_dir / "test_serving_app.py"
    serving_module = repo / "src" / "tabular_analysis" / "serve" / "app.py"
    if serving_script.exists() and serving_module.exists():
        steps.insert(4, ("serving_app", [py, str(serving_script)]))
    else:
        print(f"SKIP: serving app smoke is unavailable ({serving_module})")
    if doctor_script.exists() and doctor_module.exists():
        steps.insert(
            10,
            (
                "smoke_doctor_lint",
                [py, str(doctor_script), "--repo", str(repo)],
            ),
        )
    else:
        print(f"SKIP: doctor lint is unavailable ({doctor_module})")
    ui_contract_lint_script = tests_dir / "test_ui_contract_lint.py"
    ui_contract_lint_module = repo / "src" / "tabular_analysis" / "ops" / "ui_contract_lint.py"
    if ui_contract_lint_script.exists() and ui_contract_lint_module.exists():
        steps.insert(
            10,
            (
                "ui_contract_lint",
                [py, str(ui_contract_lint_script), "--repo", str(repo)],
            ),
        )
    else:
        print(f"SKIP: ui_contract_lint is unavailable ({ui_contract_lint_module})")
    return steps


def _full_steps(repo: Path, py: str) -> List[Step]:
    steps = _quick_steps(repo, py)
    tests_dir = repo / "tools" / "tests"
    cleanup_apply_cmd = [
        py,
        str(repo / "tools" / "cleanup_repo.py"),
        "--repo",
        str(repo),
        "--apply",
    ]
    cleanup_check_cmd = [
        py,
        str(repo / "tools" / "cleanup_repo.py"),
        "--repo",
        str(repo),
        "--check",
    ]

    def _append_step_if_script_exists(name: str, script: Path, *extra_args: str) -> None:
        if not script.exists():
            print(f"SKIP: {script} not found")
            return
        steps.append((name, [py, str(script), *extra_args]))

    regression_script = tests_dir / "smoke_train_regression_model.py"
    if not regression_script.exists():
        print(f"SKIP: {regression_script} not found")
        return steps
    steps.extend(
        [
            (
                "smoke_train_regression_model (ridge)",
                [
                    py,
                    str(regression_script),
                    "--repo",
                    str(repo),
                    "--model",
                    "ridge",
                ],
            ),
            (
                "smoke_train_regression_model (random_forest)",
                [
                    py,
                    str(regression_script),
                    "--repo",
                    str(repo),
                    "--model",
                    "random_forest",
                    "--expect-feature-importance",
                ],
            ),
        ]
    )
    _append_step_if_script_exists(
        "smoke_imbalance",
        tests_dir / "smoke_imbalance.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_uncertainty",
        tests_dir / "smoke_uncertainty.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_calibration",
        tests_dir / "smoke_calibration.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_metric_ci",
        tests_dir / "smoke_metric_ci.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_decision_summary",
        tests_dir / "smoke_decision_summary.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_champion_registry",
        tests_dir / "smoke_champion_registry.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_drift_enhanced",
        tests_dir / "smoke_drift_enhanced.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "smoke_batch_chunked",
        tests_dir / "smoke_batch_chunked.py",
        "--repo",
        str(repo),
    )
    _append_step_if_script_exists(
        "test_refactor_foundation",
        tests_dir / "test_refactor_foundation.py",
    )
    _append_step_if_script_exists(
        "test_clearml_runtime_contracts",
        tests_dir / "test_clearml_runtime_contracts.py",
    )
    serve_pkg = repo / "src" / "tabular_analysis" / "serve" / "__init__.py"
    smoke_serve_import = tests_dir / "smoke_serve_import.py"
    if serve_pkg.exists() and smoke_serve_import.exists():
        steps.append(("smoke_serve_import", [py, str(smoke_serve_import)]))
    else:
        print(f"SKIP: serve import smoke is unavailable ({serve_pkg})")
    steps.extend(
        [
            ("cleanup_residue_apply_full_end", cleanup_apply_cmd),
            ("cleanup_residue_check_full_end", cleanup_check_cmd),
        ]
    )
    return steps


def _run_steps(
    steps: List[Step],
    *,
    cwd: Path,
    max_seconds: int,
    heartbeat_seconds: int,
) -> int:
    for name, cmd in steps:
        print(f"\n==> {name}")
        rc = _run(
            cmd,
            cwd=cwd,
            step_name=name,
            max_seconds=max_seconds,
            heartbeat_seconds=heartbeat_seconds,
        )
        if rc != 0:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--max-seconds", type=int, default=900, help="Per-step timeout in seconds.")
    ap.add_argument("--heartbeat-seconds", type=int, default=30, help="Progress heartbeat interval.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run short verification suite.")
    mode.add_argument("--full", action="store_true", help="Run extended verification suite.")
    args = ap.parse_args()

    repo = _repo_root(args.repo)
    py = sys.executable

    if not args.quick and not args.full:
        args.quick = True

    steps = _full_steps(repo, py) if args.full else _quick_steps(repo, py)
    max_seconds = int(max(args.max_seconds, 1))
    heartbeat_seconds = int(max(args.heartbeat_seconds, 1))
    return _run_steps(
        steps,
        cwd=repo,
        max_seconds=max_seconds,
        heartbeat_seconds=heartbeat_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
