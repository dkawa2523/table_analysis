#!/usr/bin/env python3
"""Verify ClearML UI structure for a rehearsal usecase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tabular_analysis.platform_adapter_clearml_env import normalize_clearml_version_num
from tabular_analysis.platform_adapter_task import (
    clearml_task_id,
    clearml_task_project_name,
    clearml_task_script,
    clearml_task_status_from_obj,
    clearml_task_tags,
    list_clearml_tasks_by_tags,
)


EXPECTED_SECTIONS = {"inputs", "dataset", "selection", "preprocess", "model", "eval", "pipeline", "clearml"}


def _task_name(task: Any) -> str:
    for attr in ("name", "task_name"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    data = getattr(task, "data", None)
    name = getattr(data, "name", None) if data is not None else None
    return str(name) if name else "unknown"


def _task_project(task: Any) -> str | None:
    return clearml_task_project_name(task)


def _extract_process(tags: list[str]) -> str:
    for tag in tags:
        if tag.startswith("process:"):
            return tag.split(":", 1)[1]
    return "unknown"


def _extract_sections(params: dict[str, Any]) -> set[str]:
    sections: set[str] = set()
    for key in params.keys():
        text = str(key)
        if "/" in text:
            section = text.split("/", 1)[0]
        elif "." in text:
            section = text.split(".", 1)[0]
        else:
            continue
        sections.add(section.strip().lower())
    return sections


def _get_parameters(task: Any) -> dict[str, Any]:
    getter = getattr(task, "get_parameters", None)
    if callable(getter):
        try:
            payload = getter()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
    getter = getattr(task, "get_parameters_as_dict", None)
    if callable(getter):
        try:
            payload = getter()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
    return {}


def _summarize_task(task: Any) -> dict[str, Any]:
    task_id = clearml_task_id(task)
    tags = clearml_task_tags(task)
    script = clearml_task_script(task)
    version_num = normalize_clearml_version_num(script.get("version_num"))
    params = _get_parameters(task)
    sections = _extract_sections(params) if params else set()
    return {
        "id": task_id,
        "name": _task_name(task),
        "project": _task_project(task),
        "status": clearml_task_status_from_obj(task),
        "process": _extract_process(tags),
        "version_num": version_num or "",
        "sections": sorted(sections),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify ClearML UI structure for a usecase_id.")
    ap.add_argument("--usecase-id", required=True, help="Target usecase_id tag.")
    ap.add_argument("--project-root", default=None, help="Optional project root hint.")
    ap.add_argument("--require-pipeline", dest="require_pipeline", action="store_true")
    ap.add_argument("--no-require-pipeline", dest="require_pipeline", action="store_false")
    ap.set_defaults(require_pipeline=True)
    ap.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = ap.parse_args()

    try:
        tasks = list_clearml_tasks_by_tags([f"usecase:{args.usecase_id}"])
    except Exception as exc:
        print(f"[error] failed to query ClearML tasks: {exc}")
        return 1

    summaries = [_summarize_task(task) for task in tasks]
    if args.project_root:
        prefix = str(args.project_root).rstrip("/")
        summaries = [
            summary
            for summary in summaries
            if summary.get("project") and str(summary["project"]).startswith(prefix)
        ]
    by_process: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_process.setdefault(summary["process"], []).append(summary)

    required = {"preprocess", "train_model", "leaderboard"}
    if args.require_pipeline:
        required.add("pipeline")

    errors: list[str] = []
    warnings: list[str] = []

    missing = [proc for proc in sorted(required) if proc not in by_process]
    if missing:
        errors.append("missing processes: " + ", ".join(missing))

    for summary in summaries:
        if summary.get("version_num"):
            warnings.append(f"version_num pinned: {summary['process']} {summary['id']}")
        sections = set(summary.get("sections") or [])
        if sections and sections == {"general"}:
            warnings.append(f"hyperparameters only in General: {summary['process']} {summary['id']}")
        elif sections and not (sections & EXPECTED_SECTIONS):
            warnings.append(f"hyperparameters missing expected sections: {summary['process']} {summary['id']}")
        elif not sections:
            warnings.append(f"hyperparameters not detected: {summary['process']} {summary['id']}")

    output = {
        "usecase_id": args.usecase_id,
        "project_root_hint": args.project_root,
        "counts": {key: len(value) for key, value in by_process.items()},
        "tasks": summaries,
        "warnings": warnings,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"usecase_id: {args.usecase_id}")
        for process, items in sorted(by_process.items()):
            print(f"- {process}: {len(items)}")
            for item in items:
                print(
                    f"  - {item['id']} {item['status'] or ''} {item['name']} "
                    f"version_num={item['version_num'] or 'empty'}"
                )
        if warnings:
            print("warnings:")
            for msg in warnings:
                print(f"- {msg}")
        if errors:
            print("errors:")
            for msg in errors:
                print(f"- {msg}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
