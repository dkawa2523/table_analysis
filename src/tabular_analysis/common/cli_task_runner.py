from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
def run_cli_task(
    args: list[str],
    *,
    cwd: Path,
    config_dir: Path | None = None,
    step_name: str | None = None,
) -> None:
    cmd = [sys.executable, "-m", "tabular_analysis.cli", *args]
    env = os.environ.copy()
    if config_dir is not None and "TABULAR_ANALYSIS_CONFIG_DIR" not in env:
        env["TABULAR_ANALYSIS_CONFIG_DIR"] = str(config_dir)
    # Avoid child tasks inheriting parent ClearML task IDs.
    env.pop("CLEARML_TASK_ID", None)
    env.pop("TRAINS_TASK_ID", None)
    env.pop("CLEARML_PROC_MASTER_ID", None)
    env.pop("TRAINS_PROC_MASTER_ID", None)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        prefix = f"{step_name} failed" if step_name else "Command failed"
        raise RuntimeError(
            f"{prefix} (exit={proc.returncode})\n$ {' '.join(cmd)}\n\n{proc.stdout}"
        )
