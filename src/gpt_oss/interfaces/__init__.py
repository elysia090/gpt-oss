"""Public facing interfaces exposed by :mod:`gpt_oss`."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["api", "cli"]


def __getattr__(name: str) -> ModuleType:
    """Lazily import interface modules to avoid heavy dependencies."""

    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - imported for static analyzers
    from . import api, cli  # noqa: F401
