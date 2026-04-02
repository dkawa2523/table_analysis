#!/usr/bin/env python3
"""Smoke test for optional serving import."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    import tabular_analysis.serve as serve

    if not hasattr(serve, "create_app"):
        raise AssertionError("tabular_analysis.serve.create_app is missing.")

    if importlib.util.find_spec("fastapi") is None:
        return 0

    serve_app = importlib.import_module("tabular_analysis.serve.app")
    if not hasattr(serve_app, "create_app"):
        raise AssertionError("tabular_analysis.serve.app.create_app is missing.")

    app = serve_app.create_app(model_bundle_path=None)
    if app is None:
        raise AssertionError("create_app returned None.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
