from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .common.collection_utils import stringify_payload as _stringify_payload, to_container as _to_container
from .platform_adapter_common import (
    PlatformAdapterError,
    RECOVERABLE_CLEARML_ERRORS,
    apply_clearml_files_host_substitution,
    normalize_files_host,
)
from .platform_adapter_dataset import get_dataset_info, get_dataset_local_copy, register_dataset
from .platform_adapter_task_query import _get_clearml_task
from .platform_adapter_clearml_env import is_clearml_enabled

if TYPE_CHECKING:
    from .platform_adapter_task_context import TaskContext


def _hash_artifact_payload(payload: Any, *, symbol: str) -> str:
    try:
        from ml_platform import artifacts as platform_artifacts
    except RECOVERABLE_CLEARML_ERRORS:
        platform_artifacts = None
    hasher = getattr(platform_artifacts, symbol, None)
    if callable(hasher):
        return hasher(payload)
    normalized = _stringify_payload(_to_container(payload, resolve=True))
    data = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_config(payload: Any) -> str:
    return _hash_artifact_payload(payload, symbol="hash_config")


def hash_split(payload: Any) -> str:
    return _hash_artifact_payload(payload, symbol="hash_split")


def hash_recipe(payload: Any) -> str:
    return _hash_artifact_payload(payload, symbol="hash_recipe")


def resolve_output_dir(cfg: Any, stage: str) -> Path:
    base = Path(getattr(cfg.run, "output_dir", "outputs"))
    return base / stage


def save_config_resolved(ctx: "TaskContext", cfg: Any) -> Path:
    from .platform_adapter_task_context import save_config_resolved as _impl

    return _impl(ctx, cfg)


def write_out_json(ctx: "TaskContext", out: dict[str, Any]) -> Path:
    from .platform_adapter_task_context import write_out_json as _impl

    return _impl(ctx, out)


def upload_artifact(ctx: "TaskContext", name: str, path: Path) -> None:
    from .platform_adapter_task_context import upload_artifact as _impl

    _impl(ctx, name, path)


def write_manifest(ctx: "TaskContext", manifest: dict[str, Any]) -> Path:
    try:
        from ml_platform.artifacts import write_manifest as platform_write_manifest
    except ImportError as exc:
        if ctx.task is not None:
            raise PlatformAdapterError("ml_platform.artifacts.write_manifest not available.") from exc
        platform_write_manifest = None
    if platform_write_manifest is not None:
        try:
            return platform_write_manifest(
                manifest,
                output_dir=ctx.output_dir,
                task=ctx.task,
                filename="manifest.json",
            )
        except RECOVERABLE_CLEARML_ERRORS as exc:
            raise PlatformAdapterError(f"Failed to write manifest via ml_platform: {exc}") from exc
    path = ctx.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _resolve_task_artifact(task: Any, artifact_name: str) -> Any | None:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, dict):
        if artifact_name in artifacts:
            return artifacts[artifact_name]
    elif artifacts is not None and not isinstance(artifacts, (str, bytes)):
        try:
            for item in artifacts:
                if isinstance(item, dict):
                    key = item.get("key") or item.get("name")
                    if key == artifact_name:
                        return item
                else:
                    key = getattr(item, "key", None) or getattr(item, "name", None)
                    if key == artifact_name:
                        return item
        except RECOVERABLE_CLEARML_ERRORS:
            pass
    getter = getattr(task, "get_artifact", None)
    if callable(getter):
        try:
            return getter(artifact_name)
        except RECOVERABLE_CLEARML_ERRORS:
            return None
    return None


def _artifact_local_copy(artifact: Any) -> str | None:
    if artifact is None:
        return None
    if isinstance(artifact, Path):
        return str(artifact)
    if isinstance(artifact, str):
        return artifact
    getter = getattr(artifact, "get_local_copy", None)
    if callable(getter):
        try:
            return getter()
        except RECOVERABLE_CLEARML_ERRORS:
            return None
    if isinstance(artifact, dict):
        for key in ("local_copy", "local_path", "path", "artifact_local_path"):
            value = artifact.get(key)
            if value:
                return str(value)
    return None


def get_task_artifact_local_copy(cfg: Any, task_id: str, artifact_name: str) -> Path:
    if not is_clearml_enabled(cfg):
        raise PlatformAdapterError("ClearML is disabled; cannot fetch task artifacts.")
    apply_clearml_files_host_substitution()
    task = _get_clearml_task(task_id)
    artifact = _resolve_task_artifact(task, artifact_name)
    local_path = _artifact_local_copy(artifact)
    if not local_path:
        uri = None
        if isinstance(artifact, dict):
            uri = artifact.get("uri") or artifact.get("url")
        else:
            uri = getattr(artifact, "uri", None) or getattr(artifact, "url", None)
        if uri:
            try:
                from clearml.backend_api import Session
            except ImportError:
                Session = None
            try:
                import requests
            except ImportError:
                requests = None
            if Session is not None and requests is not None:
                try:
                    session = Session()
                    files_host = os.getenv("CLEARML_FILES_HOST") or session.config.get("api.files_server")
                    if files_host:
                        normalized = normalize_files_host(files_host)
                        if normalized:
                            parsed = urlparse(uri)
                            if parsed.hostname in {"host.docker.internal", "clearml-fileserver"}:
                                uri = uri.replace(f"{parsed.scheme}://{parsed.netloc}", normalized)
                    creds = session.config.get("api.credentials")
                    token_resp = session.send_request(
                        service="auth",
                        action="login",
                        json={"access_key": creds["access_key"], "secret_key": creds["secret_key"]},
                    )
                    token = token_resp.json()["data"]["token"]
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(uri, headers=headers, timeout=30)
                    response.raise_for_status()
                    target_dir = Path("/tmp/clearml_artifacts") / task_id
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / artifact_name
                    target_path.write_bytes(response.content)
                    local_path = str(target_path)
                except RECOVERABLE_CLEARML_ERRORS:
                    local_path = None
        if not local_path:
            raise PlatformAdapterError(f"Artifact {artifact_name} not found on ClearML task {task_id}.")
    path = Path(local_path)
    if not path.exists():
        raise PlatformAdapterError(f"Artifact {artifact_name} local copy does not exist: {path}")
    return path


def resolve_clearml_task_url(cfg: Any, task_id: str) -> str | None:
    if not is_clearml_enabled(cfg) or not task_id:
        return None
    try:
        task = _get_clearml_task(task_id)
    except PlatformAdapterError:
        return None
    for getter_name in ("get_output_log_web_page", "get_task_output_log_web_page"):
        getter = getattr(task, getter_name, None)
        if callable(getter):
            try:
                url = getter()
            except (RuntimeError, TypeError, ValueError, AttributeError):
                url = None
            if url:
                return str(url)
    url = getattr(task, "output_log_web_page", None)
    if url:
        return str(url)
    return None


__all__ = [
    "hash_config",
    "hash_split",
    "hash_recipe",
    "resolve_output_dir",
    "save_config_resolved",
    "write_out_json",
    "upload_artifact",
    "write_manifest",
    "get_task_artifact_local_copy",
    "resolve_clearml_task_url",
    "register_dataset",
    "get_dataset_local_copy",
    "get_dataset_info",
]
