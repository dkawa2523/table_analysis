from __future__ import annotations

from typing import Any, Mapping, Sequence

_ITERABLE_COERCE_ERRORS = (AttributeError, TypeError, ValueError)


def normalize_drift_metric_names(
    value: Any,
    *,
    default_metrics: Sequence[str] | None = None,
) -> list[str]:
    defaults = [str(item).strip().lower() for item in (default_metrics or ("psi",)) if str(item).strip()]
    if value is None:
        return defaults

    items: list[str] = []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
        items = [part for part in parts if part]
    elif isinstance(value, Mapping):
        items = [str(key) for key in value.keys()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value if item is not None]
    else:
        try:
            items = [str(item) for item in list(value) if item is not None]
        except _ITERABLE_COERCE_ERRORS:
            items = [str(value)]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        if key in ("psi", "ks") and key not in seen:
            normalized.append(key)
            seen.add(key)
    return normalized or defaults


def select_primary_drift_metric(report: Mapping[str, Any], *, default: str = "psi") -> str:
    metrics = report.get("metrics") or []
    normalized = [str(item).lower() for item in metrics if item is not None]
    if "psi" in normalized:
        return "psi"
    if "ks" in normalized:
        return "ks"
    return str(default)


def collect_top_drift_features(
    report: Mapping[str, Any],
    *,
    metric: str = "psi",
    limit: int = 5,
    include_metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    extras = [str(item) for item in (include_metrics or ()) if str(item)]
    for kind in ("numeric", "categorical"):
        section = report.get(kind) or {}
        if not isinstance(section, Mapping):
            continue
        for (name, entry) in section.items():
            value = entry.get(metric)
            if value is None:
                continue
            item = {
                "feature": str(name),
                "value": float(value),
                "status": entry.get("status"),
                "kind": kind,
            }
            if metric not in item:
                item[metric] = float(value)
            for extra_metric in extras:
                extra_value = entry.get(extra_metric)
                if extra_value is not None:
                    item[extra_metric] = float(extra_value)
            candidates.append(item)
    candidates.sort(key=lambda item: item["value"], reverse=True)
    return candidates[: int(max(limit, 0))]

