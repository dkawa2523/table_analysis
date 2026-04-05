#!/usr/bin/env python3
"""Run rehearsal scenarios for ClearML integration.

This helper can prepare a toy dataset via ``dataset_register`` before launching a
pipeline run, but the standard visible-template contract remains
``data.raw_dataset_id`` as the pipeline input.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import logging
import os
import platform as platform_mod
import re
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


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
    platform_src = repo.parent / "ml_platform_v1-master" / "src"
    if platform_src.exists():
        existing = env.get("PYTHONPATH", "")
        entries = [item for item in existing.split(os.pathsep) if item]
        if str(platform_src) not in entries:
            entries.insert(0, str(platform_src))
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


def _resolve_model_overrides(
    task_type: str,
    models: str | None,
    model_set: str | None,
    *,
    use_selection: bool,
) -> list[str]:
    if model_set:
        return [_format_override("pipeline.model_set", model_set)]
    if not models or models == "small":
        if task_type == "classification":
            key = "pipeline.selection.enabled_model_variants" if use_selection else "pipeline.model_variants"
            return [_format_override(key, ["logistic_regression"])]
        key = "pipeline.selection.enabled_model_variants" if use_selection else "pipeline.model_variants"
        return [_format_override(key, ["ridge", "elasticnet"])]
    if models == "all":
        if task_type == "regression":
            return [_format_override("pipeline.model_set", "regression_all")]
        key = "pipeline.selection.enabled_model_variants" if use_selection else "pipeline.model_variants"
        return [_format_override(key, ["logistic_regression"])]
    model_list = [item.strip() for item in models.split(",") if item.strip()]
    if not model_list:
        return []
    key = "pipeline.selection.enabled_model_variants" if use_selection else "pipeline.model_variants"
    return [_format_override(key, model_list)]


def _resolve_preprocess_overrides(preprocess: str, *, use_selection: bool) -> list[str]:
    items = [item.strip() for item in preprocess.split(",") if item.strip()]
    if not items:
        return []
    if use_selection:
        return [_format_override("pipeline.selection.enabled_preprocess_variants", items)]
    if len(items) == 1:
        return [_format_override("pipeline.preprocess_variant", items[0])]
    return [_format_override("pipeline.preprocess_variants", items)]


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
    template_task_id: str | None,
    pipeline_profile: str | None,
    task_type: str,
    preprocess: str,
    models: str | None,
    model_set: str | None,
    plan_only: bool,
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
    use_selection = execution == "agent"
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
        if template_task_id:
            cmd.append(_format_override("run.clearml.pipeline.template_task_id", template_task_id))
        cmd.append("run.clearml.env.bootstrap=uv")
        cmd.append("run.clearml.env.uv.frozen=true")
    if project_root:
        cmd.append(_format_override("run.clearml.project_root", project_root))
    if task_type == "classification":
        cmd.append("eval.task_type=classification")
        cmd.append("eval.primary_metric=accuracy")
    if pipeline_profile:
        cmd.append(_format_override("+pipeline.profile", pipeline_profile))
    if plan_only:
        cmd.append("pipeline.plan_only=true")
    cmd.extend(_resolve_preprocess_overrides(preprocess, use_selection=use_selection))
    cmd.extend(_resolve_model_overrides(task_type, models, model_set, use_selection=use_selection))
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


def _ensure_repo_src_on_path(repo: Path) -> None:
    src_dir = repo / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _normalize_task_status(value: Any) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    if not text:
        return None
    if text in {"completed", "closed", "finished", "published", "success"}:
        return "completed"
    if text in {"failed", "error"}:
        return "failed"
    if text in {"stopped", "aborted"}:
        return "stopped"
    if text in {"queued", "created", "pending"}:
        return "queued"
    if text in {"in_progress", "running"}:
        return "running"
    return text


def _wait_for_pipeline_task(
    task_id: str,
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> str:
    try:
        from clearml import Task as ClearMLTask
    except Exception as exc:
        raise RuntimeError("clearml is required to wait for remote pipeline completion.") from exc

    deadline = time.time() + max(float(timeout_sec), 1.0)
    last_status: str | None = None
    while True:
        task = ClearMLTask.get_task(task_id=str(task_id))
        status = _normalize_task_status(getattr(task, "status", None) or getattr(task, "get_status", lambda: None)())
        if status and status != last_status:
            print(f"pipeline_status: {status}")
            last_status = status
        if status in {"completed", "failed", "stopped"}:
            return str(status)
        if time.time() >= deadline:
            return "timeout"
        time.sleep(max(float(poll_interval_sec), 1.0))


def _build_minimal_clearml_cfg() -> Any:
    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        raise RuntimeError("omegaconf is required to sync ClearML artifacts.") from exc
    return OmegaConf.create({"run": {"clearml": {"enabled": True, "execution": "logging"}}})


def _list_task_artifact_names(task_id: str) -> set[str]:
    try:
        from clearml import Task as ClearMLTask
    except Exception:
        return set()
    try:
        task = ClearMLTask.get_task(task_id=str(task_id))
    except Exception:
        return set()
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, dict):
        return {str(key) for key in artifacts.keys()}
    names: set[str] = set()
    if artifacts is not None and not isinstance(artifacts, (str, bytes)):
        try:
            for item in artifacts:
                key = None
                if isinstance(item, dict):
                    key = item.get("key") or item.get("name")
                else:
                    key = getattr(item, "key", None) or getattr(item, "name", None)
                if key:
                    names.add(str(key))
        except Exception:
            return names
    return names


@contextlib.contextmanager
def _suppress_clearml_artifact_logs() -> Iterable[None]:
    logger_names = ("clearml", "clearml.storage")
    loggers = [logging.getLogger(name) for name in logger_names]
    previous = [(logger, logger.level, logger.propagate) for logger in loggers]
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        for logger in loggers:
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
        yield
    finally:
        logging.disable(previous_disable)
        for (logger, level, propagate) in previous:
            logger.setLevel(level)
            logger.propagate = propagate


def _sync_pipeline_artifacts(
    repo: Path,
    output_dir: Path,
    pipeline_task_id: str,
    *,
    missing_only: bool = False,
) -> dict[str, Path]:
    _ensure_repo_src_on_path(repo)
    from tabular_analysis.platform_adapter_task import get_task_artifact_local_copy

    cfg = _build_minimal_clearml_cfg()
    stage_dir = output_dir / "99_pipeline"
    stage_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    available = _list_task_artifact_names(pipeline_task_id)
    fetch_failures: list[str] = []
    candidate_names = (
        "pipeline_run.json",
        "run_summary.json",
        "report.json",
        "report.md",
        "report_links.json",
        "out.json",
        "manifest.json",
        "config_resolved.yaml",
    )
    for name in candidate_names:
        target = stage_dir / name
        if missing_only and target.exists():
            copied[name] = target
            continue
        if available and name not in available:
            continue
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                _suppress_clearml_artifact_logs(),
            ):
                source = get_task_artifact_local_copy(cfg, pipeline_task_id, name)
        except Exception:
            fetch_failures.append(name)
            continue
        try:
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        except OSError:
            shutil.copy2(source, target)
        copied[name] = target
    pipeline_run = copied.get("pipeline_run.json")
    if pipeline_run is not None and "run_summary.json" not in copied:
        fallback = stage_dir / "run_summary.json"
        shutil.copy2(pipeline_run, fallback)
        copied["run_summary.json"] = fallback
    if fetch_failures and not missing_only:
        print(
            "[warn] Some pipeline artifacts were unavailable from fileserver; "
            f"falling back where possible: {', '.join(fetch_failures)}"
        )
    elif fetch_failures:
        print(
            "[warn] Some supplemental pipeline artifacts were unavailable from fileserver; "
            f"reused rebuilt outputs where possible: {', '.join(fetch_failures)}"
        )
    return copied


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _extract_tag_value(tags: Sequence[str], prefix: str) -> str | None:
    for tag in tags:
        text = str(tag).strip()
        if text.startswith(prefix):
            value = text[len(prefix) :].strip()
            return value or None
    return None


def _build_live_task_ref(
    *,
    task_id: str,
    tags: Sequence[str],
    process: str,
) -> dict[str, Any]:
    ref: dict[str, Any] = {"task_id": task_id}
    preprocess_variant = _extract_tag_value(tags, "preprocess:")
    if preprocess_variant:
        ref["preprocess_variant"] = preprocess_variant
    if process == "train_model":
        model_variant = _extract_tag_value(tags, "model:")
        if model_variant:
            ref["model_variant"] = model_variant
        ref["train_task_id"] = task_id
    if process == "train_ensemble":
        method = _extract_tag_value(tags, "ensemble_method:")
        if method:
            ref["ensemble_method"] = method
        ref["train_task_id"] = task_id
    return ref


def _rebuild_pipeline_outputs_from_clearml(
    *,
    repo: Path,
    output_dir: Path,
    usecase_id: str,
    pipeline_task_id: str,
    controller_status: str | None,
    project_root: str | None,
) -> dict[str, Any]:
    _ensure_repo_src_on_path(repo)
    from tabular_analysis.platform_adapter_task import (
        clearml_task_id,
        clearml_task_project_name,
        clearml_task_status_from_obj,
        clearml_task_tags,
        list_clearml_tasks_by_tags,
    )
    from tabular_analysis.processes import pipeline as pipeline_module
    from tabular_analysis.reporting.pipeline_report import build_pipeline_report_bundle

    stage_dir = output_dir / "99_pipeline"
    stage_dir.mkdir(parents=True, exist_ok=True)
    base_summary = _load_json_if_exists(stage_dir / "pipeline_run.json") or {}
    prefix = str(project_root).rstrip("/") if project_root else ""
    tasks = list_clearml_tasks_by_tags([f"usecase:{usecase_id}"])
    by_process: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        task_id = clearml_task_id(task)
        if not task_id:
            continue
        tags = [str(tag) for tag in (clearml_task_tags(task) or [])]
        if ("solution:tabular-analysis" not in tags) and task_id != pipeline_task_id:
            continue
        project_name = clearml_task_project_name(task) or ""
        if prefix and (not str(project_name).startswith(prefix)):
            continue
        process = _extract_tag_value(tags, "process:")
        if not process:
            continue
        by_process.setdefault(process, []).append(
            {
                "task_id": task_id,
                "project_name": project_name,
                "status": _normalize_task_status(clearml_task_status_from_obj(task)),
                "tags": tags,
            }
        )

    def _sort_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
        tags = [str(tag) for tag in (item.get("tags") or [])]
        return (
            _extract_tag_value(tags, "preprocess:") or "",
            _extract_tag_value(tags, "model:") or _extract_tag_value(tags, "ensemble_method:") or "",
            str(item.get("task_id") or ""),
        )

    dataset_items = sorted(by_process.get("dataset_register") or [], key=_sort_key)
    preprocess_items = sorted(by_process.get("preprocess") or [], key=_sort_key)
    train_items = sorted(by_process.get("train_model") or [], key=_sort_key)
    ensemble_items = sorted(by_process.get("train_ensemble") or [], key=_sort_key)
    leaderboard_items = sorted(by_process.get("leaderboard") or [], key=_sort_key)
    infer_items = sorted(by_process.get("infer") or [], key=_sort_key)
    controller_items = sorted(by_process.get("pipeline") or [], key=_sort_key)

    dataset_register_ref = (
        _build_live_task_ref(task_id=str(dataset_items[0]["task_id"]), tags=dataset_items[0]["tags"], process="dataset_register")
        if dataset_items
        else base_summary.get("dataset_register_ref")
    )
    preprocess_refs = [
        _build_live_task_ref(task_id=str(item["task_id"]), tags=item["tags"], process="preprocess")
        for item in preprocess_items
    ]
    train_refs = [
        _build_live_task_ref(task_id=str(item["task_id"]), tags=item["tags"], process="train_model")
        for item in train_items
    ]
    train_ensemble_refs = [
        _build_live_task_ref(task_id=str(item["task_id"]), tags=item["tags"], process="train_ensemble")
        for item in ensemble_items
    ]
    leaderboard_ref = (
        _build_live_task_ref(task_id=str(leaderboard_items[0]["task_id"]), tags=leaderboard_items[0]["tags"], process="leaderboard")
        if leaderboard_items
        else base_summary.get("leaderboard_ref")
    )
    infer_ref = (
        _build_live_task_ref(task_id=str(infer_items[0]["task_id"]), tags=infer_items[0]["tags"], process="infer")
        if infer_items
        else base_summary.get("infer_ref")
    )

    summary = dict(base_summary)
    summary["pipeline_task_id"] = pipeline_task_id
    summary["dataset_register_ref"] = dataset_register_ref
    summary["preprocess_ref"] = preprocess_refs
    summary["train_refs"] = train_refs
    summary["train_ensemble_refs"] = train_ensemble_refs
    summary["leaderboard_ref"] = leaderboard_ref
    summary["infer_ref"] = infer_ref
    summary["executed_jobs"] = len(train_refs) + len(train_ensemble_refs)
    if "planned_jobs" not in summary:
        summary["planned_jobs"] = summary["executed_jobs"]
    controller_tags = [str(tag) for tag in ((controller_items[0]["tags"] if controller_items else []))]
    pipeline_profile = _extract_tag_value(controller_tags, "pipeline_profile:")
    if pipeline_profile:
        summary["pipeline_profile"] = pipeline_profile
    grid_run_id = _extract_tag_value(controller_tags, "grid:")
    if grid_run_id:
        summary["grid_run_id"] = grid_run_id

    cfg = _build_minimal_clearml_cfg()
    summary = pipeline_module._finalize_pipeline_run_summary(cfg, summary, status=controller_status)
    report_bundle = build_pipeline_report_bundle(
        summary,
        cfg=cfg,
        pipeline_run_dir=stage_dir,
        pipeline_task_id=pipeline_task_id,
    )
    (stage_dir / "pipeline_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "out.json").write_text(json.dumps({"pipeline_run": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "report.json").write_text(json.dumps(report_bundle.payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "report.md").write_text(report_bundle.markdown, encoding="utf-8")
    (stage_dir / "report_links.json").write_text(json.dumps(report_bundle.links, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


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
    ap = argparse.ArgumentParser(description="Run ClearML rehearsal flows, optionally registering a dataset before pipeline execution.")
    ap.add_argument("--repo", default=".", help="Repository root (default: current directory)")
    ap.add_argument(
        "--execution",
        default="local",
        choices=["local", "logging", "agent"],
        help="Execution mode",
    )
    ap.add_argument("--queue-name", default=None, help="ClearML queue name (agent only)")
    ap.add_argument("--template-task-id", default=None, help="Optional seed/base pipeline task id (legacy config key name is kept)")
    ap.add_argument(
        "--pipeline-profile",
        default=None,
        choices=["pipeline", "train_model_full", "train_ensemble_full"],
        help="Optional visible pipeline profile override",
    )
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
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="Build or launch only the controller plan without executing child steps",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute")
    ap.add_argument(
        "--skip-ui-verify",
        action="store_true",
        help="Skip ClearML UI audit step",
    )
    ap.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=None,
        help="Wait for remote pipeline completion (default: on for --execution agent)",
    )
    ap.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Do not wait for remote pipeline completion",
    )
    ap.add_argument(
        "--wait-timeout-sec",
        type=float,
        default=14400.0,
        help="Timeout for waiting on remote pipeline completion",
    )
    ap.add_argument(
        "--poll-interval-sec",
        type=float,
        default=15.0,
        help="Polling interval while waiting on remote pipeline completion",
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
    wait_for_completion = (args.execution == "agent") if args.wait is None else bool(args.wait)

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
                template_task_id=args.template_task_id,
                pipeline_profile=args.pipeline_profile,
                task_type=args.task_type,
                preprocess=args.preprocess,
                models=args.models,
                model_set=args.model_set,
                plan_only=args.plan_only,
            )
            commands.append(cmd_pipeline)
            _run(cmd_pipeline, cwd=repo, dry_run=False, env=env)
            pipeline_task_id = _read_pipeline_task_id(output_dir)
            controller_status: str | None = None
            copied_artifacts: dict[str, Path] = {}
            pipeline_run_payload: dict[str, Any] | None = None
            if args.execution == "agent" and pipeline_task_id and wait_for_completion:
                controller_status = _wait_for_pipeline_task(
                    pipeline_task_id,
                    timeout_sec=args.wait_timeout_sec,
                    poll_interval_sec=args.poll_interval_sec,
                )
                if controller_status in {"completed", "failed", "stopped"}:
                    rebuild_error: Exception | None = None
                    try:
                        pipeline_run_payload = _rebuild_pipeline_outputs_from_clearml(
                            repo=repo,
                            output_dir=output_dir,
                            usecase_id=usecase_id,
                            pipeline_task_id=pipeline_task_id,
                            controller_status=controller_status,
                            project_root=args.project_root,
                        )
                    except Exception as exc:
                        rebuild_error = exc
                    copied_artifacts = _sync_pipeline_artifacts(
                        repo,
                        output_dir,
                        pipeline_task_id,
                        missing_only=True,
                    )
                    if pipeline_run_payload is None:
                        pipeline_run_payload = _load_json_if_exists(copied_artifacts.get("pipeline_run.json"))
                    payload_status = _normalize_task_status(
                        pipeline_run_payload.get("status") if isinstance(pipeline_run_payload, dict) else None
                    )
                    if pipeline_run_payload is None or payload_status in {"queued", "running"}:
                        if rebuild_error is not None:
                            raise RuntimeError(
                                "Failed to rebuild pipeline outputs from ClearML state after remote execution."
                            ) from rebuild_error
                else:
                    copied_artifacts = _sync_pipeline_artifacts(
                        repo,
                        output_dir,
                        pipeline_task_id,
                        missing_only=False,
                    )
                    pipeline_run_payload = _load_json_if_exists(copied_artifacts.get("pipeline_run.json"))
            summary = {
                "usecase_id": usecase_id,
                "raw_dataset_id": raw_dataset_id,
                "pipeline_task_id": pipeline_task_id,
                "execution": args.execution,
                "task_type": args.task_type,
                "output_dir": str(output_dir),
            }
            if controller_status:
                summary["controller_status"] = controller_status
            if pipeline_run_payload:
                for key in (
                    "status",
                    "requested_jobs",
                    "planned_jobs",
                    "executed_jobs",
                    "disabled_jobs",
                    "completed_jobs",
                    "failed_jobs",
                    "stopped_jobs",
                    "running_jobs",
                    "queued_jobs",
                    "skipped_due_to_policy",
                    "pipeline_profile",
                    "grid_run_id",
                ):
                    if key in pipeline_run_payload:
                        summary[key] = pipeline_run_payload.get(key)
            _write_summary(output_dir, summary)
            print(f"usecase_id: {usecase_id}")
            print(f"raw_dataset_id: {raw_dataset_id}")
            if pipeline_task_id:
                print(f"pipeline_task_id: {pipeline_task_id}")
            if controller_status == "timeout":
                raise RuntimeError(
                    f"Timed out while waiting for pipeline controller {pipeline_task_id} to finish."
                )
            if controller_status in {"failed", "stopped"}:
                raise RuntimeError(
                    f"Pipeline controller {pipeline_task_id} finished with status={controller_status}."
                )
            if args.execution != "local" and not args.skip_ui_verify:
                try:
                    import clearml  # noqa: F401
                except Exception:
                    print("[warn] clearml not installed; skipping UI audit.")
                else:
                    if _clearml_config_present(repo):
                        if args.execution != "agent" or (not wait_for_completion) or controller_status == "completed":
                            _run_ui_verify(repo, usecase_id, args.project_root)
                        else:
                            print("[warn] Pipeline is not finalized yet; skipping UI audit.")
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
                template_task_id=args.template_task_id,
                pipeline_profile=args.pipeline_profile,
                task_type=args.task_type,
                preprocess=args.preprocess,
                models=args.models,
                model_set=args.model_set,
                plan_only=args.plan_only,
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
