#!/usr/bin/env python3
"""Fail when AppleDouble metadata files are present."""

from __future__ import annotations

import argparse
from pathlib import Path


def _collect(root: Path) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("._*"):
        if path.is_file():
            matches.append(path)
    return sorted(matches)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to scan recursively.")
    ap.add_argument("--max-show", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))

    matches = _collect(root)
    if matches:
        print(f"NG: found {len(matches)} AppleDouble files under {root}")
        for path in matches[: args.max_show]:
            print(path)
        return 1

    print(f"OK: no AppleDouble files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
