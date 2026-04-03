#!/usr/bin/env python3
"""ClearML entrypoint wrapper for src/ layout."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_BOOTSTRAP_ENV = "TABULAR_ANALYSIS_BOOTSTRAPPED"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from tabular_analysis.common.clearml_bootstrap import (
    resolve_model_variant_name_from_overrides,
    resolve_required_uv_extras,
)
from tabular_analysis.common.clearml_config import read_clearml_api_section


def _find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / "conf").exists():
                return parent
    raise RuntimeError("Could not locate repo root containing conf/ directory.")


def _flatten_params(prefix: str, value: Any, out: dict[str, Any], *, sep: str = ".") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}{sep}{key}" if prefix else str(key)
            _flatten_params(next_prefix, item, out, sep=sep)
        return
    if prefix:
        out[prefix] = value


def _extract_cli_keys(argv: list[str]) -> set[str]:
    keys: set[str] = set()
    for item in argv:
        if not item or item.startswith("-") or "=" not in item:
            continue
        key = item.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and ((text[0] == text[-1] == "'") or (text[0] == text[-1] == '"')):
        return text[1:-1]
    return text


def _parse_cli_overrides(argv: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in argv:
        if not item or item.startswith("-") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key.startswith("+"):
            key = key.lstrip("+")
        if not key:
            continue
        overrides[key] = _strip_quotes(value.strip())
    return overrides


def _override_key_candidates(key: str) -> list[str]:
    candidates: list[str] = []
    for candidate in (key, key.replace(".", "/"), key.replace("/", ".")):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _get_override(overrides: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        for candidate in _override_key_candidates(key):
            value = overrides.get(candidate)
            if value is not None:
                return value
    return None


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    text = _strip_quotes(value.strip())
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    items = [item.strip() for item in text.split(",") if item.strip()]
    return [_strip_quotes(item) for item in items if item]


def _is_clearml_context() -> bool:
    return any(
        os.getenv(key)
        for key in (
            "CLEARML_TASK_ID",
            "TRAINS_TASK_ID",
            "CLEARML_AGENT_TASK_ID",
            "CLEARML_TASK",
        )
    )


def _in_docker() -> bool:
    return Path("/.dockerenv").exists()


def _maybe_patch_clearml_files_host() -> None:
    existing = os.getenv("CLEARML_FILES_HOST")
    if existing:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(existing)
            host = parsed.hostname
        except Exception:
            host = None
        if not (_in_docker() and host in {"localhost", "127.0.0.1"}):
            return

    def _set_from_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
        except Exception:
            return False
        host = parsed.hostname
        if not host:
            return False
        if host in {"localhost", "127.0.0.1"} and _in_docker():
            host = "host.docker.internal"
        if "docker.internal" not in host:
            return False
        scheme = parsed.scheme or "http"
        os.environ["CLEARML_FILES_HOST"] = f"{scheme}://{host}:8081"
        return True

    api_host = os.getenv("CLEARML_API_HOST") or os.getenv("CLEARML_WEB_HOST") or ""
    if api_host and _set_from_url(api_host):
        return

    def _set_from_config(path: Path) -> bool:
        try:
            api_section = read_clearml_api_section(config_file=path)
        except Exception:
            return False
        files_host = api_section.get("files_server") or api_section.get("files") or ""
        if files_host and _set_from_url(files_host):
            return True
        api_server = api_section.get("host") or api_section.get("api_server") or api_section.get("web_server")
        if api_server and _set_from_url(api_server):
            return True
        return False

    cfg_path = os.getenv("CLEARML_CONFIG_FILE")
    if cfg_path:
        if _set_from_config(Path(cfg_path)):
            return

    for candidate in (
        Path.cwd() / "clearml.conf",
        Path.home() / "clearml.conf",
        Path.home() / ".clearml.conf",
        Path.home() / ".config" / "clearml.conf",
    ):
        if candidate.exists() and _set_from_config(candidate):
            return


def _resolve_bootstrap_mode(overrides: dict[str, str]) -> str:
    value = _get_override(overrides, "run.clearml.env.bootstrap", "run.clearml.bootstrap")
    if value:
        return value.strip().lower()
    for key in ("TABULAR_ANALYSIS_CLEARML_BOOTSTRAP", "TABULAR_ANALYSIS_BOOTSTRAP"):
        value = os.getenv(key)
        if value:
            return value.strip().lower()
    return "none"


def _resolve_task_name(overrides: dict[str, str]) -> str | None:
    value = _get_override(overrides, "task", "task.name")
    if value:
        name = value.split("/")[-1].strip()
        return name or None
    return None


def _infer_optimize_enabled(overrides: dict[str, str]) -> bool:
    value = _get_override(overrides, "infer.mode", "infer/mode")
    if value and value.strip().lower() == "optimize":
        return True
    return False


def _resolve_uv_extras(overrides: dict[str, str]) -> list[str]:
    explicit_raw = _get_override(overrides, "run.clearml.env.uv.extras")
    explicit_extras = _parse_list(explicit_raw) if explicit_raw is not None else None
    task_name = _resolve_task_name(overrides)
    model_variant_name = resolve_model_variant_name_from_overrides(overrides)
    infer_mode = "optimize" if _infer_optimize_enabled(overrides) else _get_override(overrides, "infer.mode")
    return resolve_required_uv_extras(
        task_name=task_name,
        model_variant_name=model_variant_name,
        infer_mode=infer_mode,
        explicit_extras=explicit_extras,
        explicit_extras_provided=explicit_raw is not None,
    )


def _resolve_uv_settings(overrides: dict[str, str]) -> tuple[str, list[str], bool, bool]:
    venv_dir = _get_override(overrides, "run.clearml.env.uv.venv_dir") or ".venv"
    extras = _resolve_uv_extras(overrides)
    all_extras = _parse_bool(_get_override(overrides, "run.clearml.env.uv.all_extras"), default=False)
    frozen = _parse_bool(_get_override(overrides, "run.clearml.env.uv.frozen"), default=True)
    return (venv_dir, extras, all_extras, frozen)


def _resolve_apt_settings(overrides: dict[str, str]) -> tuple[list[str], bool, bool]:
    packages = _parse_list(_get_override(overrides, "run.clearml.env.apt_packages"))
    if not packages:
        packages = _parse_list(os.getenv("TABULAR_ANALYSIS_APT_PACKAGES"))
    update = _parse_bool(_get_override(overrides, "run.clearml.env.apt_update"), default=True)
    allow_local = _parse_bool(_get_override(overrides, "run.clearml.env.apt_allow_local"), default=False)
    return packages, update, allow_local


def _maybe_install_apt_packages(argv: list[str]) -> None:
    def _warn(message: str) -> None:
        print(f"[clearml_entrypoint] {message}", file=sys.stderr)

    overrides = _parse_cli_overrides(argv)
    packages, update, allow_local = _resolve_apt_settings(overrides)
    if not packages:
        return
    if not _is_clearml_context() and not allow_local:
        return
    if os.geteuid() != 0:
        _warn("Skipping apt install (requires root).")
        return
    if shutil.which("apt-get") is None:
        _warn("Skipping apt install (apt-get not found).")
        return
    env = os.environ.copy()
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    if update:
        subprocess.run(["apt-get", "update", "-y"], check=True, env=env)
    cmd = ["apt-get", "install", "-y", "--no-install-recommends", *packages]
    subprocess.run(cmd, check=True, env=env)


def _resolve_uv_command() -> list[str]:
    uv_bin = shutil.which("uv")
    if uv_bin:
        return [uv_bin]
    raise RuntimeError(
        "uv binary is required for ClearML bootstrap. "
        "Install it in the agent image or task runtime environment."
    )


def _ensure_uv_available() -> None:
    _resolve_uv_command()


def _apply_uv_runtime_env(env: dict[str, str], *, venv_path: Path | None = None) -> dict[str, str]:
    patched = dict(env)
    if venv_path is not None:
        patched["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    patched.setdefault("UV_PYTHON", sys.executable)
    if os.name != "nt" and (_is_clearml_context() or _in_docker()):
        patched.setdefault("UV_CACHE_DIR", "/root/.clearml/uv-cache")
    return patched


def _uv_sync(
    repo_root: Path,
    venv_path: Path,
    *,
    extras: list[str],
    all_extras: bool,
    frozen: bool,
) -> None:
    cmd = [*_resolve_uv_command(), "sync", "--project", str(repo_root)]
    if frozen:
        cmd.append("--frozen")
    if all_extras:
        cmd.append("--all-extras")
    else:
        for extra in extras:
            cmd.extend(["--extra", extra])
    env = _apply_uv_runtime_env(os.environ.copy(), venv_path=venv_path)
    subprocess.run(cmd, check=True, env=env)


def _exec_uv_run(repo_root: Path, venv_path: Path, argv: list[str]) -> None:
    env = _apply_uv_runtime_env(os.environ.copy(), venv_path=venv_path)
    env[_BOOTSTRAP_ENV] = "1"
    env.setdefault("TABULAR_ANALYSIS_CONFIG_DIR", str(repo_root / "conf"))
    cmd = [
        *_resolve_uv_command(),
        "run",
        "--project",
        str(repo_root),
        "--",
        "python",
        "-m",
        "tabular_analysis.cli",
        *argv,
    ]
    subprocess.run(cmd, check=True, env=env)
    raise SystemExit(0)


def _maybe_bootstrap_uv(repo_root: Path, argv: list[str]) -> None:
    if os.getenv(_BOOTSTRAP_ENV):
        return
    overrides = _parse_cli_overrides(argv)
    mode = _resolve_bootstrap_mode(overrides)
    if mode in {"none", "false", "0", "off"}:
        return
    if mode == "auto" and not _is_clearml_context():
        return
    venv_dir, extras, all_extras, frozen = _resolve_uv_settings(overrides)
    lock_path = repo_root / "uv.lock"
    if frozen and not lock_path.exists():
        raise RuntimeError("uv.lock is required for frozen ClearML bootstrap.")
    _ensure_uv_available()
    venv_path = Path(venv_dir)
    if not venv_path.is_absolute():
        venv_path = (repo_root / venv_path).resolve()
    _uv_sync(
        repo_root,
        venv_path,
        extras=extras,
        all_extras=all_extras,
        frozen=frozen,
    )
    _exec_uv_run(repo_root, venv_path, argv)


def _looks_like_container(text: str) -> bool:
    return (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}"))


def _needs_quote(text: str) -> bool:
    if not text:
        return True
    if _looks_like_container(text):
        return False
    for ch in text:
        if ch.isspace() or ch in "(),=":
            return True
    return False


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


_JSON_OVERRIDE_KEYS = {
    "infer.input_json",
    "infer.batch.inputs_json",
    "infer.validation.inputs_json",
    "infer.optimize.search_space",
}


def _is_json_text(text: str) -> bool:
    if not text:
        return False
    try:
        import json

        json.loads(text)
        return True
    except Exception:
        return False


def _format_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return _quote(text) if _needs_quote(text) else text


def _format_override_value_for_key(key: str, value: Any) -> str:
    text = "" if value is None else str(value)
    if key in _JSON_OVERRIDE_KEYS and _is_json_text(text):
        return _quote(text)
    return _format_override_value(value)


def _normalize_json_override_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in argv:
        if not item or item.startswith("-") or "=" not in item:
            normalized.append(item)
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            normalized.append(item)
            continue
        key_for_check = key.lstrip("+")
        raw = value.strip()
        if key_for_check in _JSON_OVERRIDE_KEYS:
            text = _strip_quotes(raw)
            if _is_json_text(text):
                normalized.append(f"{key}={_quote(text)}")
                continue
        normalized.append(item)
    return normalized


def _load_clearml_overrides() -> dict[str, Any]:
    def _warn(message: str) -> None:
        print(f"[clearml_entrypoint] {message}", file=sys.stderr)

    def _resolve_task_id() -> str | None:
        for key in (
            "CLEARML_TASK_ID",
            "TRAINS_TASK_ID",
            "CLEARML_AGENT_TASK_ID",
            "TASK_ID",
            "CLEARML_TASK",
        ):
            value = os.getenv(key)
            if value:
                return str(value)
        return None

    try:
        from clearml import Task  # type: ignore
    except Exception:
        return {}
    def _collect_overrides(task: Any) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        params: Any = {}
        try:
            params = task.get_parameters() or {}
        except Exception:
            params = {}
        if isinstance(params, dict):
            args_section = params.get("Args")
            if isinstance(args_section, dict):
                for key, value in args_section.items():
                    _flatten_params(str(key), value, overrides, sep="/")
            else:
                for key, value in params.items():
                    if not isinstance(key, str) or not key.startswith("Args/"):
                        continue
                    override_key = key[5:]
                    if override_key:
                        overrides[override_key] = value
        if overrides:
            return overrides
        params = {}
        try:
            params = task.get_parameters_as_dict() or {}
        except Exception:
            params = {}
        args_section = params.get("Args") if isinstance(params, dict) else None
        if isinstance(args_section, dict):
            for key, value in args_section.items():
                _flatten_params(str(key), value, overrides, sep="/")
        elif isinstance(params, dict):
            for key, value in params.items():
                if not isinstance(key, str) or not key.startswith("Args/"):
                    continue
                override_key = key[5:]
                if override_key:
                    overrides[override_key] = value
        return overrides

    task = Task.current_task()
    if task is not None:
        overrides = _collect_overrides(task)
        if overrides:
            return overrides

    task_id = _resolve_task_id()
    if task_id:
        try:
            task = Task.get_task(task_id=str(task_id))
        except Exception:
            _warn(f"failed to load ClearML task {task_id}; using CLI defaults.")
            task = None
        if task is not None:
            overrides = _collect_overrides(task)
            if overrides:
                return overrides
            _warn("no ClearML overrides detected; check task parameters and agent environment.")
    return {}


def _merge_clearml_overrides(argv: list[str]) -> list[str]:
    argv = _normalize_json_override_args(list(argv))
    overrides = _load_clearml_overrides()
    if not overrides:
        return argv
    existing = _extract_cli_keys(argv)
    merged = list(argv)
    for key, value in overrides.items():
        if key in existing:
            continue
        merged.append(f"{key}={_format_override_value_for_key(key, value)}")
    return merged


def main(argv: list[str] | None = None) -> None:
    repo_root = _find_repo_root()
    src_path = repo_root / "src"
    src_str = str(src_path)
    if src_path.exists() and src_str not in sys.path:
        sys.path.insert(0, src_str)

    if not os.getenv("TABULAR_ANALYSIS_CONFIG_DIR"):
        os.environ["TABULAR_ANALYSIS_CONFIG_DIR"] = str(repo_root / "conf")
    _maybe_patch_clearml_files_host()

    args = _merge_clearml_overrides(list(argv or []))
    _maybe_install_apt_packages(args)
    _maybe_bootstrap_uv(repo_root, args)

    from tabular_analysis import cli

    cli.main(args)


if __name__ == "__main__":
    main(sys.argv[1:])
