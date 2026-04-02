#!/usr/bin/env python3
"""Smoke test for platform_adapter dataset split compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabular_analysis import platform_adapter_artifacts as artifacts
from tabular_analysis import platform_adapter_core as core
from tabular_analysis import platform_adapter_dataset as dataset


def _assert_exports() -> None:
    if artifacts.register_dataset is not dataset.register_dataset:
        raise AssertionError("artifacts.register_dataset must reference dataset implementation")
    if artifacts.get_dataset_local_copy is not dataset.get_dataset_local_copy:
        raise AssertionError("artifacts.get_dataset_local_copy must reference dataset implementation")
    if artifacts.get_dataset_info is not dataset.get_dataset_info:
        raise AssertionError("artifacts.get_dataset_info must reference dataset implementation")


def _assert_core_delegation(repo: Path) -> None:
    original_register = dataset.register_dataset
    original_local_copy = dataset.get_dataset_local_copy
    original_info = dataset.get_dataset_info
    calls: list[str] = []
    fake_copy = (repo / "work" / "_platform_adapter_dataset" / "model_bundle.joblib").resolve()
    fake_copy.parent.mkdir(parents=True, exist_ok=True)
    fake_copy.write_bytes(b"")

    def _fake_register(cfg, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("register")
        if "dataset_name" not in kwargs:
            raise AssertionError("register_dataset kwargs missing dataset_name")
        return "dataset-123"

    def _fake_local_copy(cfg, dataset_id):  # type: ignore[no-untyped-def]
        calls.append("local_copy")
        if str(dataset_id) != "dataset-123":
            raise AssertionError("unexpected dataset_id")
        return fake_copy

    def _fake_info(cfg, dataset_id):  # type: ignore[no-untyped-def]
        calls.append("info")
        return {"dataset_id": str(dataset_id), "dataset_name": "fake"}

    dataset.register_dataset = _fake_register  # type: ignore[assignment]
    dataset.get_dataset_local_copy = _fake_local_copy  # type: ignore[assignment]
    dataset.get_dataset_info = _fake_info  # type: ignore[assignment]
    try:
        cfg = object()
        dataset_id = core.register_dataset(
            cfg,
            dataset_path=repo,
            dataset_name="fake_dataset",
        )
        if dataset_id != "dataset-123":
            raise AssertionError("core.register_dataset delegation failed")

        local_path = core.get_dataset_local_copy(cfg, dataset_id)
        if local_path != fake_copy:
            raise AssertionError("core.get_dataset_local_copy delegation failed")

        info = core.get_dataset_info(cfg, dataset_id)
        if info.get("dataset_name") != "fake":
            raise AssertionError("core.get_dataset_info delegation failed")
    finally:
        dataset.register_dataset = original_register  # type: ignore[assignment]
        dataset.get_dataset_local_copy = original_local_copy  # type: ignore[assignment]
        dataset.get_dataset_info = original_info  # type: ignore[assignment]

    if calls != ["register", "local_copy", "info"]:
        raise AssertionError(f"unexpected delegation call order: {calls}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    _assert_exports()
    _assert_core_delegation(repo)
    print("OK: platform adapter dataset split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
