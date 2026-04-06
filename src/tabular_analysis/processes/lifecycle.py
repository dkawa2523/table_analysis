from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Mapping
from ..platform_adapter_artifacts import hash_config, hash_recipe, hash_split, save_config_resolved, write_manifest, write_out_json
from ..platform_adapter_clearml_env import is_clearml_enabled, resolve_version_props
from ..platform_adapter_task_context import init_task_context
def start_runtime(cfg: Any, *, stage: str, task_name: str, tags: list[str] | None, properties: Mapping[str, Any] | None, task_type: Any | None=None, system_tags: list[str] | None=None) -> Any:
    """Create task context and persist config_resolved in one place."""
    ctx = init_task_context(cfg, stage=stage, task_name=task_name, tags=tags, properties=properties, task_type=task_type, system_tags=system_tags)
    save_config_resolved(ctx, cfg)
    return ctx
def _resolve_hash_value(value: Any) -> Any:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return value
    (kind, payload) = value
    kind_text = str(kind).strip().lower()
    if kind_text == 'config':
        return hash_config(payload)
    if kind_text == 'split':
        return hash_split(payload)
    if kind_text == 'recipe':
        return hash_recipe(payload)
    return value
def emit_outputs_and_manifest(ctx: Any, cfg: Any, *, process: str, out: Mapping[str, Any], inputs: Mapping[str, Any], outputs: Mapping[str, Any], hash_payloads: Mapping[str, Any], manifest_extra: Mapping[str, Any] | None=None, clearml_enabled: bool | None=None) -> None:
    """Write out.json and manifest.json with shared version/hash resolution."""
    write_out_json(ctx, dict(out))
    clearml = is_clearml_enabled(cfg) if clearml_enabled is None else bool(clearml_enabled)
    versions = resolve_version_props(cfg, clearml_enabled=clearml)
    hashes = {str(key): _resolve_hash_value(value) for (key, value) in hash_payloads.items()}
    manifest: dict[str, Any] = {'schema_version': versions.get('schema_version', 'unknown'), 'code_version': versions.get('code_version', 'unknown'), 'platform_version': versions.get('platform_version', 'unknown'), 'process': process, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'inputs': dict(inputs), 'outputs': dict(outputs), 'hashes': hashes}
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    write_manifest(ctx, manifest)
