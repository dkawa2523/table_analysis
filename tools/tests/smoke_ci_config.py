#!/usr/bin/env python3
"""Smoke test for CI workflow config."""

from __future__ import annotations

import argparse
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing file: {path}") from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    workflow = repo / ".github" / "workflows" / "ci.yml"
    text = _read_text(workflow)

    required = "tools/tests/verify_all.py --quick"
    if required not in text:
        raise ValueError(f"ci.yml missing required command: {required}")

    print("OK: ci.yml contains verify_all.py --quick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
