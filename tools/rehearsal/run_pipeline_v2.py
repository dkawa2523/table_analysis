#!/usr/bin/env python3
"""Run rehearsal scenarios (dataset_register + pipeline) for ClearML integration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform as platform_mod
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _with_repo_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = repo / ".venv" / "bin"
    if venv_bin.exists():
        path = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}{os.pathsep}{path}" if path else str(venv_bin)
    src_dir = repo / "src"
    if src_dir.exists():
        existing = env.get("PYTHONPATH", "")
        entries = [item for item in existing.split(os.pathsep) if item]
        if str(src_dir) not in entries:
            entries.insert(0, str(src_dir))
            env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run(cmd: Sequence[str], *, cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> str:
    line = f"$ {_format_cmd(cmd)}"
    print(line)
    if dry_run:
        return ""
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode})\n{line}\n\n{proc.stdout}")
    return proc.stdout


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized.strip("-_") or "unknown"


def _dataset_token(path: Path) -> str:
    stem = Path(path.name).stem or path.name
    return _sanitize_identifier(stem)


def _utc_stamp(now: dt.datetime | None = None) -> tuple[str, str]:
    now_value = now or dt.datetime.now(dt.timezone.utc)
    stamp = now_value.strftime("%Y%m%d_%H%M%S")
    iso = now_value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return stamp, iso


def _needs_quote(text: str) -> bool:
    if not text:
        return True
    for ch in text:
        if ch.isspace() or ch in "[]{}(),=":
            return True
    return False


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _format_value(value: str) -> str:
    text = str(value)
    return _quote(text) if _needs_quote(text) else text


def _hydra_list(values: Iterable[str]) -> str:
    return "[" + ",".join(values) + "]"


def _make_toy_csv(path: Path, *, task_type: str) -> None:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "pandas/numpy are required for rehearsal runs. Install requirements/base.txt first.\n"
            + str(exc)
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
    if task_type == "classification":
        score = 0.4 * df["num1"] - 0.2 * df["num2"] + (df["cat"] == "b").astype(float)
        prob = 1 / (1 + np.exp(-score))
        df["target"] = (prob > 0.5).astype(int)
    else:
        df["target"] = 0.3 * df["num1"] - 0.1 * df["num2"] + (df["cat"] == "b").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _render_log(
    *,
    timestamp_iso: str,
    execution: str,
    task_type: str,
    dry_run: bool,
    repo: Path,
    dataset_path: Path,
    output_dir: Path,
    usecase_id: str,
    commands: Iterable[Sequence[str]],
    result: str,
    error: str | None,
) -> str:
    lines: list[str] = [
        f"## {timestamp_iso}",
        f"- execution: {execution}",
        f"- task_type: {task_type}",
        f"- dry_run: {str(dry_run).lower()}",
        f"- usecase_id: {usecase_id}",
        f"- repo: {repo}",
        f"- dataset_path: {dataset_path}",
        f"- output_dir: {output_dir}",
        f"- python: {sys.version.split()[0]} ({sys.executable})",
        f"- platform: {platform_mod.platform()}",
        "- commands:",
        "```bash",
    ]
    for cmd in commands:
        lines.append(f"$ {_format_cmd(cmd)}")
    lines.extend(["```", f"- result: {result}"])
    if error:
        lines.append(f"- error: {error}")
    lines.append("")
    return "\n".join(lines)


def _summarize_error(exc: Exception) -> str:
    text = str(exc).strip().replace("\r\n", "\n")
    if len(text) > 1200:
        return text[:1200] + "..."
    return text


def _format_override(key: str, value: str | list[str]) -> str:
    if isinstance(value, list):
        return f"{key}={_hydra_list(value)}"
    return f"{key}={_format_value(str(value))}"


def _output_dir_override(repo: Path, output_dir: Path) -> str:
    try:
        return str(output_dir.relative_to(repo))
    except ValueError:
        return str(output_dir)


def _build_dataset_register_cmd(
    *,
    execution: str,
    py: str,
    output_dir: str,
    dataset_path: Path,
    target_column: str,
    usecase_id: str,
    project_root: str | None,
    task_type: str,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "tabular_analysis.cli",
        "task=dataset_register",
        _format_override("run.usecase_id", usecase_id),
        _format_override("run.output_dir", output_dir),
        _format_override("data.dataset_path", str(dataset_path)),
        _format_override("data.target_column", target_column),
    ]
    if execution == "local":
        cmd.append("run.clearml.enabled=false")
    else:
        cmd.append("run.clearml.enabled=true")
        cmd.append("run.clearml.execution=logging")
    if project_root:
        cmd.append(_format_override("run.clearml.project_root", project_root))
    if task_type == "classification":
        cmd.append("eval.task_type=classification")
    return cmd


def _resolve_model_overrides(task_type: str, models: str | None, model_set: str | None) -> list[str]:
    if model_set:
        return [_format_override("+pipeline.model_set", model_set)]
    if not models or models == "small":
        if task_type == "classification":
            return [_format_override("+pipeline.model_variants", ["logistic_regression"])]
        return [_format_override("+pipeline.model_variants", ["ridge", "elasticnet"])]
    if models == "all":
        if task_type == "regression":
            return [_format_override("+pipeline.model_set", "regression_all")]
        return [_format_override("+pipeline.model_variants", ["logistic_regression"])]
    model_list = [item.strip() for item in models.split(",") if item.strip()]
    if not model_list:
        return []
    return [_format_override("+pipeline.model_variants", model_list)]


def _resolve_preprocess_overrides(preprocess: str) -> list[str]:
    items = [item.strip() for item in preprocess.split(",") if item.strip()]
    if not items:
        return []
    if len(items) == 1:
        return [_format_override("+pipeline.preprocess_variant", items[0])]
    return [_format_override("+pipeline.preprocess_variants", items)]


def _build_pipeline_cmd(
    *,
    execution: str,
    py: str,
    output_dir: str,
    dataset_path: Path,
    raw_dataset_id: str,
    target_column: str,
    usecase_id: str,
    project_root: str | None,
    queue_name: str | None,
    task_type: str,
    preprocess: str,
    models: str | None,
    model_set: str | None,
) -> list[str]:
    cmd = [
        py,
        "-m",
        "tabular_analysis.cli",
        "task=pipeline",
        _format_override("run.usecase_id", usecase_id),
        _format_override("run.output_dir", output_dir),
        _format_override("data.raw_dataset_id", raw_dataset_id),
        _format_override("data.target_column", target_column),
    ]
    if raw_dataset_id.startswith("local:"):
        cmd.append(_format_override("data.dataset_path", str(dataset_path)))
    if execution == "local":
        cmd.append("run.clearml.enabled=false")
    elif execution == "logging":
        cmd.append("run.clearml.enabled=true")
        cmd.append("run.clearml.execution=logging")
    else:
        cmd.append("run.clearml.enabled=true")
        cmd.append("run.clearml.execution=pipeline_controller")
        if queue_name:
            cmd.append(_format_override("run.clearml.queue_name", queue_name))
        cmd.append("run.clearml.env.bootstrap=uv")
        cmd.append("run.clearml.env.uv.all_extras=true")
        cmd.append("run.clearml.env.uv.frozen=true")
    if project_root:
        cmd.append(_format_override("run.clearml.project_root", project_root))
    if task_type == "classification":
        cmd.append("eval.task_type=classification")
        cmd.append("eval.primary_metric=accuracy")
    cmd.extend(_resolve_preprocess_overrides(preprocess))
    cmd.extend(_resolve_model_overrides(task_type, models, model_set))
    return cmd


def _read_raw_dataset_id(output_dir: Path) -> str:
    out_path = output_dir / "01_dataset_register" / "out.json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    raw_dataset_id = payload.get("raw_dataset_id")
    if not raw_dataset_id:
        raise RuntimeError("raw_dataset_id not found in dataset_register out.json")
    return str(raw_dataset_id)


def _read_pipeline_task_id(output_dir: Path) -> str | None:
    links_path = output_dir / "99_pipeline" / "report_links.json"
    if not links_path.exists():
        return None
    links = json.loads(links_path.read_text(encoding="utf-8"))
    if not isinstance(links, dict):
        return None
    pipeline_entry = links.get("pipeline")
    if not isinstance(pipeline_entry, dict):
        return None
    task_id = pipeline_entry.get("task_id")
    return str(task_id) if task_id else None


def _write_summary(output_dir: Path, summary: dict[str, str | None]) -> Path:
    path = output_dir / "rehearsal_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _clearml_config_present(repo_root: Path) -> bool:
    config_candidates: list[Path] = []
    env_path = os.environ.get("CLEARML_CONFIG_FILE")
    if env_path:
        config_candidates.append(Path(env_path).expanduser())
    config_candidates.extend(
        [
            repo_root / "clearml.conf",
            Path.home() / "clearml.conf",
            Path.home() / ".clearml.conf",
            Path.home() / ".config" / "clearml.conf",
        ]
    )
    if any(candidate.exists() for candidate in config_candidates):
        return True
    for key in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY", "CLEARML_API_HOST", "CLEARML_WEB_HOST"):
        if os.environ.get(key):
            return True
    return False


def _run_ui_verify(repo: Path, usecase_id: str, project_root: str | None) -> None:
    verify_script = repo / "tools" / "tests" / "rehearsal_verify_clearml_ui.py"
    if not verify_script.exists():
        print("[warn] UI audit script not found; skipping.")
        return
    cmd = [sys.executable, str(verify_script), "--usecase-id", usecase_id]
    if project_root:
        cmd.extend(["--project-root", project_root])
    _run(cmd, cwd=repo, dry_run=False, env=_with_repo_env(repo))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run dataset_register + pipeline rehearsal.")
    ap.add_argument("--repo", default=".", help="Repository root (default: current directory)")
    ap.add_argument(
        "--execution",
        default="local",
        choices=["local", "logging", "agent"],
        help="Execution mode",
    )
    ap.add_argument("--queue-name", default=None, help="ClearML queue name (agent only)")
    ap.add_argument("--task-type", default="regression", choices=["regression", "classification"])
    ap.add_argument("--preprocess", default="stdscaler_ohe", help="Comma-separated preprocess variants")
    ap.add_argument("--models", default="small", help="Comma list or 'small'/'all'")
    ap.add_argument("--model-set", default=None, help="Pipeline model_set name")
    ap.add_argument("--project-root", default=None, help="Override run.clearml.project_root")
    ap.add_argument("--dataset-path", default=None, help="Optional dataset path (skip toy data)")
    ap.add_argument(
        "--output-root",
        default=None,
        help="Base output dir (default: work/rehearsal/out)",
    )
    ap.add_argument(
        "--usecase-id",
        default=None,
        help="Optional usecase_id override (default: test_<dataset>_<timestamp>)",
    )
    ap.add_argument("--target-column", default="target", help="Target column name")
    ap.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute")
    ap.add_argument(
        "--skip-ui-verify",
        action="store_true",
        help="Skip ClearML UI audit step",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    rehearsal_root = repo / "work" / "rehearsal"
    tmp_dir = rehearsal_root / "tmp"
    if args.output_root:
        output_root = Path(args.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = repo / output_root
    else:
        output_root = rehearsal_root / "out"

    dataset_path = Path(args.dataset_path).expanduser() if args.dataset_path else tmp_dir / "toy.csv"
    stamp, timestamp_iso = _utc_stamp()
    dataset_token = _dataset_token(dataset_path)
    usecase_id = args.usecase_id or f"test_{dataset_token}_{stamp}"
    output_dir = output_root / args.execution / usecase_id
    output_dir_arg = _output_dir_override(repo, output_dir)

    py = sys.executable
    commands: list[list[str]] = []
    result = "dry-run (not executed)" if args.dry_run else "success"
    error: str | None = None

    try:
        cmd_register = _build_dataset_register_cmd(
            execution=args.execution,
            py=py,
            output_dir=output_dir_arg,
            dataset_path=dataset_path,
            target_column=args.target_column,
            usecase_id=usecase_id,
            project_root=args.project_root,
            task_type=args.task_type,
        )
        commands.append(cmd_register)
        cmd_pipeline: list[str] | None = None
        env = _with_repo_env(repo)
        if not args.dry_run:
            if not args.dataset_path:
                _make_toy_csv(dataset_path, task_type=args.task_type)
            output_dir.mkdir(parents=True, exist_ok=True)
            _run(cmd_register, cwd=repo, dry_run=False, env=env)
            raw_dataset_id = _read_raw_dataset_id(output_dir)
            cmd_pipeline = _build_pipeline_cmd(
                execution=args.execution,
                py=py,
                output_dir=output_dir_arg,
                dataset_path=dataset_path,
                raw_dataset_id=raw_dataset_id,
                target_column=args.target_column,
                usecase_id=usecase_id,
                project_root=args.project_root,
                queue_name=args.queue_name,
                task_type=args.task_type,
                preprocess=args.preprocess,
                models=args.models,
                model_set=args.model_set,
            )
            commands.append(cmd_pipeline)
            _run(cmd_pipeline, cwd=repo, dry_run=False, env=env)
            pipeline_task_id = _read_pipeline_task_id(output_dir)
            summary = {
                "usecase_id": usecase_id,
                "raw_dataset_id": raw_dataset_id,
                "pipeline_task_id": pipeline_task_id,
                "execution": args.execution,
                "task_type": args.task_type,
                "output_dir": str(output_dir),
            }
            _write_summary(output_dir, summary)
            print(f"usecase_id: {usecase_id}")
            print(f"raw_dataset_id: {raw_dataset_id}")
            if pipeline_task_id:
                print(f"pipeline_task_id: {pipeline_task_id}")
            if args.execution != "local" and not args.skip_ui_verify:
                try:
                    import clearml  # noqa: F401
                except Exception:
                    print("[warn] clearml not installed; skipping UI audit.")
                else:
                    if _clearml_config_present(repo):
                        _run_ui_verify(repo, usecase_id, args.project_root)
                    else:
                        print("[warn] ClearML config not detected; skipping UI audit.")
        else:
            raw_dataset_id = "local:<hash>"
            cmd_pipeline = _build_pipeline_cmd(
                execution=args.execution,
                py=py,
                output_dir=output_dir_arg,
                dataset_path=dataset_path,
                raw_dataset_id=raw_dataset_id,
                target_column=args.target_column,
                usecase_id=usecase_id,
                project_root=args.project_root,
                queue_name=args.queue_name,
                task_type=args.task_type,
                preprocess=args.preprocess,
                models=args.models,
                model_set=args.model_set,
            )
            commands.append(cmd_pipeline)
            _run(cmd_register, cwd=repo, dry_run=True, env=env)
            _run(cmd_pipeline, cwd=repo, dry_run=True, env=env)
    except Exception as exc:
        result = "failure"
        error = _summarize_error(exc)
    finally:
        log_path = rehearsal_root / "rehearsal_log.md"
        entry = _render_log(
            timestamp_iso=timestamp_iso,
            execution=args.execution,
            task_type=args.task_type,
            dry_run=args.dry_run,
            repo=repo,
            dataset_path=dataset_path,
            output_dir=output_dir,
            usecase_id=usecase_id,
            commands=commands,
            result=result,
            error=error,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    if result == "failure":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
