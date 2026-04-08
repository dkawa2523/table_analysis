from __future__ import annotations

"""Search-backend adapter for infer optimize mode.

The current backend is Optuna, but infer core should depend on these adapter
functions instead of vendor-specific APIs so future replacement stays local.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..viz.optuna_plots import (
    build_contour,
    build_optimization_history,
    build_parallel_coordinate,
    build_param_importance,
)


def resolve_optimize_backend_name(optimize_settings: Mapping[str, Any] | None) -> str:
    value = None if not optimize_settings else optimize_settings.get("backend")
    text = str(value).strip().lower() if value is not None else ""
    return text or "optuna"


def _parse_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _build_optuna_sampler(name: str, seed: int | None) -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for infer.mode=optimize. Install with: uv sync --extra optuna (or pip install optuna)"
        ) from exc
    key = (name or "").strip().lower()
    if key in ("tpe", "tp"):
        return optuna.samplers.TPESampler(seed=seed)
    if key in ("random", "rand"):
        return optuna.samplers.RandomSampler(seed=seed)
    if key in ("cmaes", "cma"):
        try:
            return optuna.samplers.CmaEsSampler(seed=seed)
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("infer.optimize.sampler=cmaes requires cmaes dependency.") from exc
    raise ValueError("infer.optimize.sampler must be tpe, random, or cmaes.")


def create_optimize_study(optimize_settings: Mapping[str, Any]) -> tuple[str, Any, Any]:
    backend_name = resolve_optimize_backend_name(optimize_settings)
    if backend_name != "optuna":
        raise ValueError(f"Unsupported infer.optimize.backend: {backend_name}")
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for infer.mode=optimize. Install with: uv sync --extra optuna (or pip install optuna)"
        ) from exc
    sampler = _build_optuna_sampler(optimize_settings.get("sampler"), optimize_settings.get("sampler_seed"))
    study = optuna.create_study(direction=optimize_settings.get("direction"), sampler=sampler)
    return (backend_name, study, TrialState)


def _suggest_optuna_params(trial: Any, search_space: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for entry in search_space:
        name = entry.get("name")
        if not name:
            continue
        type_name = entry.get("type")
        if not type_name:
            if "choices" in entry or "values" in entry:
                type_name = "categorical"
            else:
                type_name = "float"
        key = str(type_name).strip().lower()
        if key in {"int", "integer"}:
            type_name = "int"
        elif key in {"categorical", "category", "choice", "choices", "discrete"}:
            type_name = "categorical"
        else:
            type_name = "float"
        if type_name == "categorical":
            choices = entry.get("choices") or entry.get("values")
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                raise ValueError(f"infer.optimize.search_space {name} missing choices.")
            params[str(name)] = trial.suggest_categorical(str(name), list(choices))
            continue
        low = entry.get("low")
        high = entry.get("high")
        if low is None or high is None:
            raise ValueError(f"infer.optimize.search_space {name} requires low/high.")
        step = entry.get("step")
        log_scale = _parse_bool(entry.get("log")) if "log" in entry else False
        if step is not None and log_scale:
            raise ValueError(f"infer.optimize.search_space {name} cannot set log and step together.")
        if type_name == "int":
            try:
                low_int = int(low)
                high_int = int(high)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"infer.optimize.search_space {name} requires int low/high.") from exc
            step_int = None
            if step is not None:
                try:
                    step_int = int(step)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"infer.optimize.search_space {name} step must be int.") from exc
            params[str(name)] = trial.suggest_int(str(name), low_int, high_int, step=step_int, log=log_scale)
            continue
        try:
            low_float = float(low)
            high_float = float(high)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"infer.optimize.search_space {name} requires float low/high.") from exc
        step_float = None
        if step is not None:
            try:
                step_float = float(step)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"infer.optimize.search_space {name} step must be float.") from exc
        params[str(name)] = trial.suggest_float(str(name), low_float, high_float, step=step_float, log=log_scale)
    return params


def suggest_optimize_params(
    backend_name: str,
    *,
    trial: Any,
    search_space: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if backend_name != "optuna":
        raise ValueError(f"Unsupported infer.optimize.backend: {backend_name}")
    return _suggest_optuna_params(trial, search_space)


def build_optimize_plots(
    backend_name: str,
    *,
    study: Any,
    optimize_settings: Mapping[str, Any],
    search_space: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if backend_name != "optuna":
        return {}
    figures: dict[str, Any] = {
        "history": build_optimization_history(
            study,
            log_scale=bool(optimize_settings.get("history_log_scale")),
            output_path=output_dir / "optuna_history.png",
        ),
        "parallel": build_parallel_coordinate(study, output_path=output_dir / "optuna_parallel.png"),
        "importance": build_param_importance(study, output_path=output_dir / "optuna_importance.png"),
    }
    contour_params = optimize_settings.get("contour_params")
    if not contour_params:
        contour_params = [entry.get("name") for entry in search_space if entry.get("name")][:2]
    if contour_params and len(contour_params) >= 2:
        figures["contour"] = build_contour(
            study,
            params=contour_params,
            output_path=output_dir / "optuna_contour.png",
        )
    return figures


__all__ = [
    "build_optimize_plots",
    "create_optimize_study",
    "resolve_optimize_backend_name",
    "suggest_optimize_params",
]
