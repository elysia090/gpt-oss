"""Compatibility layer that lazily forwards to :mod:`gpt_oss.interfaces.cli`."""

from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:  # pragma: no cover - simple forwarding
    module = importlib.import_module("gpt_oss.interfaces.cli")
    return getattr(module, name)


def __dir__() -> list[str]:  # pragma: no cover - reflective helper
    module = importlib.import_module("gpt_oss.interfaces.cli")
    attrs = set(module.__all__) if hasattr(module, "__all__") else set(dir(module))
    return sorted(attrs)
