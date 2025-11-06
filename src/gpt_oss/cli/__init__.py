"""Compatibility layer that lazily forwards to :mod:`gpt_oss.interfaces.cli`."""

from __future__ import annotations

import importlib
from typing import Any

_LOCAL_SUBMODULES = {
    "chat_cli": "gpt_oss.cli.chat_cli",
    "generate_cli": "gpt_oss.cli.generate_cli",
    "sera_chat": "gpt_oss.cli.sera_chat",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - simple forwarding
    if name in _LOCAL_SUBMODULES:
        return importlib.import_module(_LOCAL_SUBMODULES[name])
    module = importlib.import_module("gpt_oss.interfaces.cli")
    return getattr(module, name)


def __dir__() -> list[str]:  # pragma: no cover - reflective helper
    module = importlib.import_module("gpt_oss.interfaces.cli")
    attrs = set(_LOCAL_SUBMODULES)
    attrs.update(module.__all__ if hasattr(module, "__all__") else dir(module))
    return sorted(attrs)


__all__ = sorted(_LOCAL_SUBMODULES)
