#!/usr/bin/env python3
"""Validate markdown file path references under src/tabular_analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_PATH_RE = re.compile(r"`(src/tabular_analysis/[A-Za-z0-9_./-]+)`")


def _extract_refs(markdown: Path) -> set[str]:
    refs: set[str] = set()
    text = markdown.read_text(encoding="utf-8")
    for match in _PATH_RE.findall(text):
        if "*" in match or "<" in match or ">" in match:
            continue
        refs.add(match)
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument(
        "--docs",
        nargs="+",
        default=[
            "docs/54_CLEARML_MINIMALITY_GUIDE.md",
            "docs/65_DEV_GUIDE_DIRECTORY_MAP.md",
            "docs/84_REHEARSAL_GUIDE.md",
        ],
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    missing: list[str] = []

    for rel_doc in args.docs:
        doc_path = (repo / rel_doc).resolve()
        if not doc_path.exists():
            missing.append(rel_doc)
            continue
        for ref in sorted(_extract_refs(doc_path)):
            target = (repo / ref).resolve()
            if not target.exists():
                missing.append(ref)

    if missing:
        print(f"missing={len(missing)}")
        for item in sorted(set(missing)):
            print(item)
        return 1

    print("missing=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
