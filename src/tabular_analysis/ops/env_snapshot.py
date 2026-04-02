from __future__ import annotations
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Iterable
def _run_command(cmd: Iterable[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr
def _find_git_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None
def _resolve_git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    code, out, _ = _run_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    commit = out.strip() if code == 0 else ""
    return commit or None
def _capture_pip_freeze() -> str:
    commands = (
        [sys.executable, "-m", "pip", "freeze"],
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
    )
    errors: list[str] = []
    for cmd in commands:
        code, out, err = _run_command(cmd)
        output = (out or "").strip()
        if code == 0 and output:
            return f"{output}\n"
        if code == 0 and not output:
            errors.append(f"{' '.join(cmd)} returned empty output")
        else:
            detail = (err or "").strip() or "non-zero exit"
            errors.append(f"{' '.join(cmd)} failed: {detail}")
    lines = ["# pip freeze unavailable"]
    lines.extend([f"# {item}" for item in errors if item])
    return "\n".join(lines) + "\n"
def capture_env_snapshot(output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env_path = out_dir / "env.json"
    freeze_path = out_dir / "pip_freeze.txt"
    repo_root = _find_git_root(Path.cwd())
    git_commit = _resolve_git_commit(repo_root)
    payload = {
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": {
            "version": platform.python_version(),
            "version_detail": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "solution": {
            "version": git_commit or "unknown",
            "version_source": "git" if git_commit else "unknown",
            "git_root": str(repo_root) if repo_root else None,
        },
        "pip_freeze_file": freeze_path.name,
    }
    env_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    freeze_path.write_text(_capture_pip_freeze(), encoding="utf-8")
    return env_path, freeze_path
