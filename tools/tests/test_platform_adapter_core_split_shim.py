#!/usr/bin/env python3
"""Smoke test for platform_adapter_core delegation after physical splits."""

from __future__ import annotations

import argparse

from tabular_analysis import platform_adapter_core as core
from tabular_analysis import platform_adapter_model as model
from tabular_analysis import platform_adapter_task_ops as task_ops


def _assert_model_wrapper() -> None:
    original = model.update_registry_model_tags
    calls: list[tuple[str, list[str], list[str]]] = []

    def _fake(*, model_id, add_tags=None, remove_prefixes=None):  # type: ignore[no-untyped-def]
        calls.append((str(model_id), list(add_tags or []), list(remove_prefixes or [])))
        return ["ok"]

    model.update_registry_model_tags = _fake  # type: ignore[assignment]
    try:
        result = core.update_registry_model_tags(
            model_id="model-1",
            add_tags=["stage:prod"],
            remove_prefixes=["stage:"],
        )
    finally:
        model.update_registry_model_tags = original  # type: ignore[assignment]

    if result != ["ok"]:
        raise AssertionError("core.update_registry_model_tags wrapper failed")
    if calls != [("model-1", ["stage:prod"], ["stage:"])]:
        raise AssertionError(f"unexpected model wrapper calls: {calls}")


def _assert_task_wrapper() -> None:
    original = task_ops.ensure_clearml_task_tags
    calls: list[tuple[str, list[str]]] = []

    def _fake(task_id, tags):  # type: ignore[no-untyped-def]
        calls.append((str(task_id), [str(tag) for tag in tags]))
        return True

    task_ops.ensure_clearml_task_tags = _fake  # type: ignore[assignment]
    try:
        result = core.ensure_clearml_task_tags("task-1", ["a", "b"])
    finally:
        task_ops.ensure_clearml_task_tags = original  # type: ignore[assignment]

    if result is not True:
        raise AssertionError("core.ensure_clearml_task_tags wrapper failed")
    if calls != [("task-1", ["a", "b"])]:
        raise AssertionError(f"unexpected task wrapper calls: {calls}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.parse_args()

    _assert_model_wrapper()
    _assert_task_wrapper()
    print("OK: platform adapter core split shim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
