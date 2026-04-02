#!/usr/bin/env python3
"""Regression checks for ClearML runtime contracts."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PLATFORM_SRC = _REPO.parent / "ml_platform_v1-master" / "src"
for candidate in (_SRC, _PLATFORM_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tabular_analysis.clearml import templates as template_module
from tabular_analysis.common.clearml_config import read_clearml_api_section
from tabular_analysis.processes.infer_support import resolve_batch_execution_mode


def _assert_clearml_hocon_reader() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "clearml.conf"
        path.write_text(
            'api {\n'
            '  api_server: "https://api.example.local"\n'
            '  web_server: "https://app.example.local"\n'
            '  files_server: "https://files.example.local"\n'
            '}\n',
            encoding="utf-8",
        )
        section = read_clearml_api_section(config_file=path)
        if section.get("api_server") != "https://api.example.local":
            raise AssertionError(f"unexpected api section: {section}")


def _assert_batch_execution_mode() -> None:
    inline = resolve_batch_execution_mode({"infer": {"batch": {"execution": "inline"}}}, clearml_enabled=False)
    if inline != "inline":
        raise AssertionError(f"unexpected inline execution: {inline}")
    children = resolve_batch_execution_mode({"infer": {"batch": {"execution": "clearml_children"}}}, clearml_enabled=True)
    if children != "clearml_children":
        raise AssertionError(f"unexpected child execution: {children}")
    try:
        resolve_batch_execution_mode({"infer": {"batch": {"execution": "clearml_children"}}}, clearml_enabled=False)
    except ValueError:
        return
    raise AssertionError("clearml_children should require ClearML")


def _assert_strict_template_lookup() -> None:
    calls: list[list[str]] = []
    original = {
        "list": template_module.list_clearml_tasks_by_tags,
        "status": template_module.clearml_task_status_from_obj,
        "tags": template_module.clearml_task_tags,
        "script": template_module.clearml_task_script,
        "id": template_module.clearml_task_id,
        "mismatch": template_module.clearml_script_mismatches,
        "spec": template_module.resolve_clearml_script_spec,
    }

    class _FakeTask:
        pass

    task = _FakeTask()

    try:
        template_module.list_clearml_tasks_by_tags = lambda tags, project_name=None: calls.append(list(tags)) or [task]
        template_module.clearml_task_status_from_obj = lambda task_obj: "created"
        template_module.clearml_task_tags = lambda task_obj: ["template:true", "usecase:TabularAnalysis", "process:infer", "schema:v1", "template_set:default", "solution:tabular-analysis"]
        template_module.clearml_task_script = lambda task_obj: {"entry_point": "tools/clearml_entrypoint.py"}
        template_module.clearml_task_id = lambda task_obj: "template-123"
        template_module.clearml_script_mismatches = lambda expected, actual: False
        template_module.resolve_clearml_script_spec = lambda *args, **kwargs: {"entry_point": "tools/clearml_entrypoint.py"}
        task_id = template_module.resolve_template_task_id({"run": {"clearml": {"template_usecase_id": "TabularAnalysis", "template_set_id": "default"}}, "task": {"name": "infer"}}, "infer")
        if task_id != "template-123":
            raise AssertionError(f"unexpected task id: {task_id}")
        if len(calls) != 1:
            raise AssertionError(f"template lookup should be strict and single-pass: {calls}")
        tags = calls[0]
        if "template_set:default" not in tags or "schema:v1" not in tags:
            raise AssertionError(f"strict lookup missing canonical tags: {tags}")
    finally:
        template_module.list_clearml_tasks_by_tags = original["list"]
        template_module.clearml_task_status_from_obj = original["status"]
        template_module.clearml_task_tags = original["tags"]
        template_module.clearml_task_script = original["script"]
        template_module.clearml_task_id = original["id"]
        template_module.clearml_script_mismatches = original["mismatch"]
        template_module.resolve_clearml_script_spec = original["spec"]


def main() -> int:
    _assert_clearml_hocon_reader()
    _assert_batch_execution_mode()
    _assert_strict_template_lookup()
    print("OK: clearml runtime contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
