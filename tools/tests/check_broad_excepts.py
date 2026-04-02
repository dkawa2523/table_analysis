#!/usr/bin/env python3
"""Fail when broad exception handlers are introduced in tabular_analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

_PATTERNS = (
    re.compile(r"^\s*except\s+Exception(\s+as\s+\w+)?\s*:\s*$"),
    re.compile(r"^\s*except\s*:\s*$"),
)


def _repo_root(repo_arg: str | None) -> Path:
    if repo_arg:
        return Path(repo_arg).resolve()
    return Path(__file__).resolve().parents[2]


def _iter_python_files(root: Path):
    src = root / "src" / "tabular_analysis"
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _find_violations(root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            violations.append((path, 0, f"read_error: {exc}"))
            continue
        for idx, line in enumerate(lines, start=1):
            if any(pattern.match(line) for pattern in _PATTERNS):
                violations.append((path, idx, line.strip()))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = _repo_root(args.repo)
    violations = _find_violations(repo)
    if not violations:
        print("OK: no broad except handlers under src/tabular_analysis")
        return 0

    print("NG: broad except handlers detected")
    for path, lineno, text in violations:
        rel = path.relative_to(repo)
        print(f"- {rel}:{lineno}: {text}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
