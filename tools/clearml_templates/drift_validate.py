from __future__ import annotations

from typing import Any, Iterable, Mapping


VISIBLE_PIPELINE_PARAMETER_SECTIONS = frozenset(
    {
        "Args",
        "inputs",
        "dataset",
        "selection",
        "preprocess",
        "model",
        "eval",
        "optimize",
        "pipeline",
        "clearml",
        "properties",
    }
)


def task_name(task: Any) -> str | None:
    value = getattr(task, "name", None) or getattr(task, "task_name", None)
    return str(value) if value else None


def task_user_properties(task: Any) -> dict[str, Any]:
    getter = getattr(task, "get_user_properties", None)
    if callable(getter):
        try:
            payload = getter()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            normalized: dict[str, Any] = {}
            for key, value in payload.items():
                if isinstance(value, Mapping) and "value" in value:
                    normalized[str(key)] = value.get("value")
                else:
                    normalized[str(key)] = value
            return normalized
    data = getattr(task, "data", None)
    execution = getattr(data, "execution", None) if data is not None else None
    user_properties = getattr(execution, "user_properties", None) if execution is not None else None
    if isinstance(user_properties, Mapping):
        return {
            str(key): value.get("value") if isinstance(value, Mapping) and "value" in value else value
            for key, value in user_properties.items()
        }
    return {}


def task_runtime(task: Any) -> dict[str, Any]:
    data = getattr(task, "data", None)
    runtime = getattr(data, "runtime", None) if data is not None else None
    if isinstance(runtime, Mapping):
        return {str(key): value for key, value in runtime.items()}
    return {}


def task_artifact_names(task: Any) -> set[str]:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, Mapping):
        return {str(key) for key in artifacts.keys()}
    names: set[str] = set()
    if artifacts is not None and not isinstance(artifacts, (str, bytes)):
        try:
            for item in artifacts:
                key = None
                if isinstance(item, Mapping):
                    key = item.get("key") or item.get("name")
                else:
                    key = getattr(item, "key", None) or getattr(item, "name", None)
                if key:
                    names.add(str(key))
        except Exception:
            return names
    return names


def task_parameters(task: Any) -> dict[str, Any]:
    getter = getattr(task, "get_parameters_as_dict", None)
    if callable(getter):
        try:
            payload = getter(cast=False)
        except TypeError:
            payload = getter()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return {str(key): value for key, value in payload.items()}
    getter = getattr(task, "get_parameters", None)
    if callable(getter):
        try:
            payload = getter()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return {str(key): value for key, value in payload.items()}
    return {}


def iter_parameter_paths(payload: Any, *, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from iter_parameter_paths(value, prefix=next_prefix)
        return
    if prefix:
        yield prefix


def configuration_paths(payload: Any, *, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from configuration_paths(value, prefix=next_prefix)
        return
    if prefix:
        yield prefix


def pipeline_parameter_keys(task: Any) -> set[str]:
    keys: set[str] = set()
    for text in iter_parameter_paths(task_parameters(task)):
        text = str(text)
        if text.startswith("Args/"):
            keys.add(text.split("/", 1)[1].replace(".", "/"))
            continue
        if "/" in text:
            section, remainder = text.split("/", 1)
            if section in VISIBLE_PIPELINE_PARAMETER_SECTIONS and remainder:
                keys.add(remainder.replace(".", "/"))
            continue
        if "." in text:
            section, remainder = text.split(".", 1)
            if section in VISIBLE_PIPELINE_PARAMETER_SECTIONS and remainder:
                keys.add(remainder.replace(".", "/"))
    return keys


def pipeline_disallowed_arg_keys(task: Any) -> set[str]:
    disallowed = {
        "ops/clearml_policy",
        "ops/usecase_id_policy",
        "ops.clearml_policy",
        "ops.usecase_id_policy",
    }
    hits: set[str] = set()
    for text in iter_parameter_paths(task_parameters(task)):
        value = str(text)
        if not value.startswith("Args/"):
            continue
        arg_key = value.split("/", 1)[1]
        if arg_key in disallowed:
            hits.add(arg_key)
    return hits


def pipeline_noncanonical_parameter_paths(task: Any) -> set[str]:
    issues: set[str] = set()
    for text in iter_parameter_paths(task_parameters(task)):
        value = str(text)
        if value.startswith("Args/"):
            arg_key = value.split("/", 1)[1]
            if "%" in arg_key or "/" in arg_key:
                issues.add(value)
            continue
        if "/" not in value:
            continue
        section, remainder = value.split("/", 1)
        if section not in VISIBLE_PIPELINE_PARAMETER_SECTIONS or not remainder:
            continue
        if "%" in remainder:
            issues.add(value)
            continue
        if any("." in segment for segment in remainder.split("/") if segment):
            issues.add(value)
    return issues


def pipeline_duplicate_visible_param_keys(task: Any, *, expected_param_keys: set[str]) -> set[str]:
    occurrences: dict[str, set[str]] = {}
    for text in iter_parameter_paths(task_parameters(task)):
        value = str(text)
        if not value.startswith("Args/"):
            if "/" not in value:
                continue
            section, remainder = value.split("/", 1)
            if section not in VISIBLE_PIPELINE_PARAMETER_SECTIONS or not remainder:
                continue
            canonical = remainder.replace(".", "/")
            if canonical not in expected_param_keys:
                continue
            occurrences.setdefault(canonical, set()).add(value)
            continue
        arg_key = value.split("/", 1)[1]
        canonical = arg_key.replace(".", "/")
        if canonical not in expected_param_keys:
            continue
        occurrences.setdefault(canonical, set()).add(value)
    return {
        key
        for (key, representations) in occurrences.items()
        if len(representations) > 1
    }


def normalize_task_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"completed", "closed", "finished", "published", "success"}:
        return "completed"
    if text in {"failed", "error"}:
        return "failed"
    if text in {"stopped", "aborted"}:
        return "stopped"
    if text in {"created", "queued", "pending"}:
        return "queued"
    if text in {"in_progress", "running"}:
        return "running"
    return text


__all__ = [
    "VISIBLE_PIPELINE_PARAMETER_SECTIONS",
    "configuration_paths",
    "iter_parameter_paths",
    "normalize_task_status",
    "pipeline_duplicate_visible_param_keys",
    "pipeline_disallowed_arg_keys",
    "pipeline_noncanonical_parameter_paths",
    "pipeline_parameter_keys",
    "task_artifact_names",
    "task_name",
    "task_parameters",
    "task_runtime",
    "task_user_properties",
]
