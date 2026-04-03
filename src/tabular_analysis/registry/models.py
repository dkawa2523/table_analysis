from __future__ import annotations
from ..common.collection_utils import to_container as _to_container
from ..common.config_utils import normalize_task_type as _normalize_task_type
from ..common.repo_utils import resolve_repo_root_fallback as _resolve_repo_root
import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_MODEL_BUILD_RECOVERABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_OMEGACONF_RECOVERABLE_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
OPTIONAL_DEPENDENCIES = {
    "lightgbm": "models",
    "xgboost": "models",
    "catboost": "models",
    "tabpfn": "tabpfn",
}
_OPTIONAL_VARIANT_EXTRAS = {
    "lgbm": "models",
    "xgboost": "models",
    "catboost": "models",
    "tabpfn": "tabpfn",
}
_CLASS_PATH_TASK_TYPE_OVERRIDES = {
    "sklearn.linear_model.LogisticRegression": "classification",
    "sklearn.linear_model.ElasticNet": "regression",
    "sklearn.linear_model.Lasso": "regression",
}
class MissingOptionalDependencyError(RuntimeError):
    def __init__(self, *, module: str, class_path: str, extra: str | None = None):
        if extra:
            install_hint = f'uv sync --extra {extra} (or pip install -e ".[{extra}]")'
        else:
            install_hint = f"uv add {module} (or pip install {module})"
        message = (
            f"Optional dependency '{module}' is required for model '{class_path}'. "
            f"Install it with: {install_hint}"
        )
        super().__init__(message)
        self.module = module
        self.class_path = class_path
        self.extra = extra
class ModelWeightsUnavailableError(RuntimeError):
    def __init__(
        self,
        *,
        model_name: str,
        auto_download: bool | None = None,
        detail: str | None = None,
    ):
        base = f"Required weights for model '{model_name}' are not available."
        if detail:
            base = f"{base} {detail}"
        if auto_download is True:
            hint = (
                "Auto-download was enabled but weights could not be retrieved. "
                "Check network access or pre-download the weights."
            )
        elif auto_download is False:
            hint = (
                "Set model_variant.params.auto_download=true to allow download, "
                "or pre-download the weights."
            )
        else:
            hint = "Ensure the required weights are available."
        super().__init__(f"{base} {hint}")
        self.model_name = model_name
        self.auto_download = auto_download
        self.detail = detail
def _normalize_task_types(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (list, tuple, set)):
        items: Iterable[Any] = values
    else:
        items = [values]
    normalized: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.add(_normalize_task_type(text))
    return normalized
def _task_types_from_class_path(class_path: Any) -> set[str]:
    class_path = _to_container(class_path)
    if isinstance(class_path, Mapping):
        return {t for t in (_normalize_task_type(key) for key in class_path.keys()) if t}
    if isinstance(class_path, str):
        override = _CLASS_PATH_TASK_TYPE_OVERRIDES.get(class_path)
        if override:
            return {_normalize_task_type(override)}
        if "Classifier" in class_path:
            return {"classification"}
        if "Regressor" in class_path or "Regression" in class_path:
            return {"regression"}
    return set()
def _resolve_class_path(class_path: Any, *, task_type: str | None) -> str:
    class_path = _to_container(class_path)
    if isinstance(class_path, Mapping):
        if not task_type:
            raise ValueError("model_variant.class_path requires eval.task_type for selection.")
        normalized = _normalize_task_type(task_type)
        selected = class_path.get(normalized)
        if not selected:
            raise ValueError(
                f"model_variant.class_path missing entry for task_type '{normalized}'."
            )
        class_path = selected
    if not isinstance(class_path, str) or "." not in class_path:
        raise ValueError(f"Invalid model_variant.class_path: {class_path}")
    return class_path
def _import_model_module(module_name: str, *, class_path: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        root_module = module_name.split(".", 1)[0]
        optional_module = (
            "tabpfn" if module_name.startswith("tabular_analysis.registry.tabpfn") else root_module
        )
        missing_names = {module_name, root_module, "tabpfn", "tabpfn_client"}
        if exc.name in missing_names and optional_module in OPTIONAL_DEPENDENCIES:
            raise MissingOptionalDependencyError(
                module=optional_module,
                class_path=class_path,
                extra=OPTIONAL_DEPENDENCIES[optional_module],
            ) from exc
        raise ImportError(f"Failed to import module '{module_name}' for model '{class_path}'.") from exc
    except (ImportError, ValueError) as exc:
        raise ImportError(f"Failed to import module '{module_name}' for model '{class_path}'.") from exc
def _resolve_model_class(module: Any, *, module_name: str, class_name: str):
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Model class '{class_name}' not found in '{module_name}'.") from exc
def _resolve_model_params(variant: dict[str, Any]) -> dict[str, Any]:
    params = _to_container(variant.get("params") or {}) or {}
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError("model_variant.params must be a dict.")
    return params
def _instantiate_model(model_cls: Any, *, class_path: str, params: dict[str, Any]):
    try:
        return model_cls(**params)
    except (MissingOptionalDependencyError, ModelWeightsUnavailableError):
        raise
    except _MODEL_BUILD_RECOVERABLE_ERRORS as exc:
        raise RuntimeError(f"Failed to instantiate model '{class_path}' with params {params}.") from exc
def build_model(model_variant: Dict[str, Any], *, task_type: str | None = None):
    if not model_variant:
        raise ValueError("model_variant is required to build a model.")
    variant = _to_container(model_variant) or {}
    if not isinstance(variant, dict):
        raise TypeError("model_variant must be a dict-like object.")
    class_path = variant.get("class_path")
    if not class_path:
        raise ValueError("model_variant.class_path is required.")
    class_path = _resolve_class_path(class_path, task_type=task_type)
    module_name, class_name = class_path.rsplit(".", 1)
    module = _import_model_module(module_name, class_path=class_path)
    model_cls = _resolve_model_class(module, module_name=module_name, class_name=class_name)
    params = _resolve_model_params(variant)
    return _instantiate_model(model_cls, class_path=class_path, params=params)
def _extract_model_variant(path: Path) -> tuple[str, Mapping[str, Any]]:
    payload = _load_yaml(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Model config must be a mapping: {path}")
    variant = payload.get("model_variant")
    if not isinstance(variant, Mapping):
        raise ValueError(f"model_variant section is missing: {path}")
    name = variant.get("name") or path.stem
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"model_variant.name is missing: {path}")
    return name.strip(), variant
def _load_yaml(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS as exc:  # pragma: no cover - omegaconf is expected in runtime
        raise RuntimeError("OmegaConf is required to load model configs.") from exc
    try:
        cfg = OmegaConf.load(path)
    except _OMEGACONF_RECOVERABLE_ERRORS as exc:
        raise ValueError(f"Failed to load model config: {path}") from exc
    return OmegaConf.to_container(cfg, resolve=False)
def _model_supports_task_type(model_variant: Mapping[str, Any], task_type: str) -> bool:
    normalized = _normalize_task_type(task_type)
    explicit = _normalize_task_types(model_variant.get("task_type"))
    if explicit:
        return normalized in explicit
    class_types = _task_types_from_class_path(model_variant.get("class_path"))
    if class_types:
        return normalized in class_types
    return False
def list_model_variants(task_type: str | None = None) -> list[str]:
    """List model variant names from conf/group/model.

    Args:
        task_type: Optional task type filter ("regression" or "classification").
    """
    model_dir = _resolve_repo_root() / "conf" / "group" / "model"
    if not model_dir.exists():
        return []
    variants: list[str] = []
    seen: set[str] = set()
    normalized_task_type = _normalize_task_type(task_type) if task_type else None
    for path in sorted(model_dir.glob("*.yaml")):
        name, variant = _extract_model_variant(path)
        if normalized_task_type and not _model_supports_task_type(variant, normalized_task_type):
            continue
        if name in seen:
            continue
        seen.add(name)
        variants.append(name)
    return variants


def resolve_model_variant_optional_extra(model_variant_name: str | None) -> str | None:
    name = str(model_variant_name).strip() if model_variant_name is not None else ""
    if not name:
        return None
    return _OPTIONAL_VARIANT_EXTRAS.get(name) or OPTIONAL_DEPENDENCIES.get(name)
