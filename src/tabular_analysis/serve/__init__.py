"""Optional serving surface for local API integrations."""

from __future__ import annotations

from .app import app, create_app

__all__ = ["app", "create_app"]
