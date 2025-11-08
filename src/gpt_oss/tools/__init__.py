"""Tooling utilities for GPT-OSS."""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["sera_quickstart", "sera_transfer"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(__all__) + list(globals().keys()))
