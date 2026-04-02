from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from .render_common import fallback_image
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
def _normalize_params(params: Sequence[str] | None) -> list[str] | None:
    if not params:
        return None
    normalized: list[str] = []
    for item in params:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized or None
def build_optimization_history(
    study: Any,
    *,
    log_scale: bool = False,
    output_path: Path | None = None,
) -> Any | Path | None:
    try:
        from optuna.visualization import plot_optimization_history  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS:
        return fallback_image(output_path, "Optimization History")
    try:
        fig = plot_optimization_history(study)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return fallback_image(output_path, "Optimization History")
    if log_scale and hasattr(fig, "update_yaxes"):
        try:
            fig.update_yaxes(type="log")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return fig
def build_parallel_coordinate(
    study: Any,
    *,
    output_path: Path | None = None,
) -> Any | Path | None:
    try:
        from optuna.visualization import plot_parallel_coordinate  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS:
        return fallback_image(output_path, "Parallel Coordinate")
    try:
        fig = plot_parallel_coordinate(study)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return fallback_image(output_path, "Parallel Coordinate")
    return fig
def build_param_importance(
    study: Any,
    *,
    output_path: Path | None = None,
) -> Any | Path | None:
    try:
        from optuna.visualization import plot_param_importances  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS:
        return fallback_image(output_path, "Param Importances")
    try:
        fig = plot_param_importances(study)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return fallback_image(output_path, "Param Importances")
    return fig
def build_contour(
    study: Any,
    *,
    params: Sequence[str] | None = None,
    output_path: Path | None = None,
) -> Any | Path | None:
    try:
        from optuna.visualization import plot_contour  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS:
        return fallback_image(output_path, "Response Surface")
    param_list = _normalize_params(params)
    try:
        if param_list:
            fig = plot_contour(study, params=param_list)
        else:
            fig = plot_contour(study)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return fallback_image(output_path, "Response Surface")
    return fig
