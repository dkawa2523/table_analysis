#!/usr/bin/env python3
"""Check optional model instantiation with clear missing dependency errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

DEFAULT_INSTALL_HINT = "uv sync --extra models"
MODEL_INSTALL_HINTS = {
    "tabpfn": "uv sync --extra tabpfn",
}
TASK_TYPES = ("regression", "classification")


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
        raise RuntimeError("omegaconf is required for optional model checks.") from exc

    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "eval.seed", seed, merge=True)
    OmegaConf.update(cfg, "eval.task_type", task_type, merge=True)
    variant = OmegaConf.select(cfg, "model_variant")
    if variant is None:
        raise ValueError(f"model_variant not found in {config_path}")
    return variant


def _install_hint(model_name: str) -> str:
    return MODEL_INSTALL_HINTS.get(model_name, DEFAULT_INSTALL_HINT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--models", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    src_root = repo / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))

    models = _parse_models(args.models)

    try:
        from tabular_analysis.registry.models import (
            MissingOptionalDependencyError,
            build_model,
        )
    except Exception as exc:
        raise RuntimeError("tabular_analysis must be importable for optional model checks.") from exc

    for name in models:
        cfg_path = repo / "conf" / "group" / "model" / f"{name}.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))
        for task_type in TASK_TYPES:
            variant = _load_model_variant(cfg_path, seed=args.seed, task_type=task_type)
            try:
                build_model(variant, task_type=task_type)
            except MissingOptionalDependencyError as exc:
                message = str(exc)
                install_hint = _install_hint(name)
                if install_hint not in message:
                    raise AssertionError(
                        f"Missing dependency message must include: {install_hint}"
                    ) from exc
                print(f"MISSING: {name} ({task_type})")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to build model '{name}' for task_type '{task_type}'."
                ) from exc
            else:
                print(f"OK: {name} ({task_type})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
