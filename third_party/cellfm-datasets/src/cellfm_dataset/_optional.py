"""Helpers for optional dependencies."""

from __future__ import annotations

from importlib import import_module


def require_dependency(module_name: str, install_hint: str):
    """Import an optional dependency with a precise installation hint."""

    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Optional dependency '{module_name}' is required for this operation. "
            f"Install it with: {install_hint}"
        ) from exc
