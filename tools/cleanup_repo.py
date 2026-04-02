#!/usr/bin/env python3
"""Repository cleanup utility and residue checker."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

DEFAULT_PATTERNS = [
    "outputs",
    "multirun",
    "dist",
    "build",
    "*.egg-info",
    ".clearml_cache",
    "**/.pytest_cache",
    "**/__pycache__",
    "*.pyc",
    "**/.mypy_cache",
    "**/.ruff_cache",
    ".coverage",
    "htmlcov",
    "**/.DS_Store",
    "**/._*",
    ".codex_exec_selfcheck.txt",
    "work/_smoke*",
    "work/_platform_adapter*",
    "work/_alerting",
    "work/_env_snapshot",
    "work/_exec_policy",
    "work/_quality_gate",
    "work/_test_model_lifecycle",
    "work/_test_retrain",
    "work/_tmp_pipeline_test",
    "work/rehearsal/out*",
    "work/rehearsal/tmp",
    "work/rehearsal/rehearsal_log.md",
    "artifacts/template_plan.json",
    "artifacts/template_plan.md",
]
IGNORE_PARTS = {".git", ".venv"}
IGNORE_RELATIVE_PATHS: set[str] = set()


def iter_matches(repo: Path, pattern: str):
    if "**" in pattern or pattern.startswith("**/"):
        return repo.rglob(pattern.replace("**/", ""))
    return repo.glob(pattern)


def is_ignored(path: Path) -> bool:
    return any(part in IGNORE_PARTS for part in path.parts)


def collect_targets(repo: Path) -> list[Path]:
    targets: set[Path] = set()
    for pattern in DEFAULT_PATTERNS:
        for path in iter_matches(repo, pattern):
            if is_ignored(path):
                continue
            if path.resolve() == repo.resolve():
                continue
            rel = str(path.relative_to(repo))
            if rel in IGNORE_RELATIVE_PATHS:
                continue
            targets.add(path)
    return sorted(targets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repo root (default: .)")
    ap.add_argument("--check", action="store_true", help="fail when residue exists")
    ap.add_argument("--dry-run", action="store_true", help="show targets only")
    ap.add_argument("--apply", action="store_true", help="delete targets")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"Invalid repo: {repo}")
        return 2

    targets = collect_targets(repo)
    if not targets:
        print("No cleanup targets found.")
        return 0

    print("Cleanup targets:")
    for path in targets:
        print(f" - {path.relative_to(repo)}")

    if args.check:
        print(f"Residue found: {len(targets)}")
        return 1

    if args.dry_run or not args.apply:
        print("\nDry-run only. Use --apply to delete.")
        return 0

    failures = 0
    for path in reversed(targets):
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
            else:
                path.unlink(missing_ok=True)
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            failures += 1
            print(f"Failed: {path} ({exc})")
    if failures:
        print(f"Cleanup completed with failures: {failures}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
