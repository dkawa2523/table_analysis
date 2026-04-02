# 26_REPRODUCIBLE_RUNTIME (Env Snapshot and Lockfile Guide)

This document describes the runtime snapshot artifacts and lockfile workflow used to
improve reproducibility without changing existing dependency layouts.

## Env snapshot outputs
Each task captures a runtime snapshot during common initialization and writes:
- `env.json`: python version, OS/platform details, and solution version (git hash when available)
- `pip_freeze.txt`: `python -m pip freeze` output (fallbacks to `pip list --format=freeze` when needed)

These files are written under each task's run directory (for example:
`outputs/<stage>/env.json`, `outputs/<stage>/pip_freeze.txt`). When ClearML is enabled,
both files are uploaded as artifacts for UI traceability. When ClearML is disabled,
the upload step is a no-op but local files are still created.

## Lockfile workflow (uv recommended)
`uv.lock` is the primary lockfile. It is generated from `pyproject.toml` and used by
ClearML entrypoint bootstrap (`uv sync --all-extras --frozen`) for stable, repo-side
environments.

Note: `run.clearml.env.apt_packages` installs **OS libraries** at task runtime via
`apt-get`. This is **Linux-only** and requires root + apt-get in the agent image.
Windows environments need a different OS package mechanism (e.g., winget/choco),
so do not assume apt-based installs there.

### Update steps
1. Regenerate the lockfile after dependency changes:
   ```bash
   uv lock
   ```
2. Sync a reproducible environment locally:
   ```bash
   uv sync --frozen
   ```
   - For ClearML parity, use `uv sync --all-extras --frozen`.

### Legacy pip lock (optional)
If you must use pip-only environments, you can still generate a `requirements/lock.txt`
snapshot. It is not tracked by default and is not used by ClearML templates once uv
bootstrap is enabled.
