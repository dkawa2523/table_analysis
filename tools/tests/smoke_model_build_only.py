#!/usr/bin/env python3
"""Smoke test: import + instantiate models without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List


def _normalize_task_type(value: str) -> str:
    key = str(value or "").strip().lower()
    if key in ("classification", "classifier", "class"):
        return "classification"
    if key in ("regression", "regressor", "reg"):
        return "regression"
    raise ValueError(f"Unsupported task_type: {value}")


def _parse_models(value: str) -> List[str]:
    items = [item.strip() for item in (value or "").split(",")]
    models = [item for item in items if item]
    if not models:
        raise ValueError("--models must include at least one model name.")
    return models


def _load_model_variant(config_path: Path, *, seed: int, task_type: str):
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as exc:
        raise RuntimeError("omegaconf is required for model build smoke tests.") from exc

    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "eval.seed", seed, merge=True)
    OmegaConf.update(cfg, "eval.task_type", task_type, merge=True)
    variant = OmegaConf.select(cfg, "model_variant")
    if variant is None:
        raise ValueError(f"model_variant not found in {config_path}")
    return variant


def _check_tabpfn_weight_guard() -> None:
    from tabular_analysis.registry.models import ModelWeightsUnavailableError
    try:
        from tabular_analysis.registry.tabpfn import TabPFNClassifier
    except Exception:
        print("SKIP: tabpfn wrapper module is unavailable")
        return

    class _MissingWeightModel:
        def predict(self, *args, **kwargs):
            raise FileNotFoundError("weights are missing")

    classifier = TabPFNClassifier.__new__(TabPFNClassifier)
    classifier._auto_download = False
    classifier._params = {}
    classifier._model = _MissingWeightModel()
    try:
        classifier.predict([0])
    except ModelWeightsUnavailableError:
        print("OK: tabpfn weight guard")
        return
    raise RuntimeError("tabpfn wrapper did not convert missing-weight error.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--task-type", default="regression")
    ap.add_argument("--models", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    src_root = repo / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))

    task_type = _normalize_task_type(args.task_type)
    models = _parse_models(args.models)

    try:
        from tabular_analysis.registry.models import build_model
    except Exception as exc:
        raise RuntimeError("tabular_analysis must be importable for smoke tests.") from exc

    for name in models:
        cfg_path = repo / "conf" / "group" / "model" / f"{name}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))
        variant = _load_model_variant(cfg_path, seed=args.seed, task_type=task_type)
        try:
            build_model(variant, task_type=task_type)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build model '{name}' for task_type '{task_type}'."
            ) from exc
        print(f"OK: {name}")

    _check_tabpfn_weight_guard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
