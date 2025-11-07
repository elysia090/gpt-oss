"""Compatibility shim that prefers the real :mod:`safetensors` package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _filter_sys_path(path_entries: Iterable[str]) -> list[str]:
    filtered: list[str] = []
    for entry in path_entries:
        try:
            resolved = Path(entry).resolve()
        except OSError:
            filtered.append(entry)
            continue
        if resolved == _REPO_ROOT:
            continue
        filtered.append(entry)
    return filtered


def _load_real_safetensors() -> ModuleType:
    saved_module = sys.modules.pop(__name__, None)
    original_sys_path = list(sys.path)
    try:
        sys.path = _filter_sys_path(original_sys_path)
        module = importlib.import_module("safetensors")
    except ModuleNotFoundError:
        if saved_module is not None:
            sys.modules[__name__] = saved_module
        raise
    finally:
        sys.path = original_sys_path

    sys.modules[__name__] = module
    return module


try:  # pragma: no cover - executed when the real wheel is installed
    _real_safetensors = _load_real_safetensors()
except ModuleNotFoundError:  # pragma: no cover - deterministic fallback
    _stub = importlib.import_module("gpt_oss._stubs.safetensors")
    globals().update({name: getattr(_stub, name) for name in getattr(_stub, "__all__", [])})
    __all__ = list(getattr(_stub, "__all__", []))
    if "__version__" not in __all__:
        __all__.append("__version__")
    globals()["__version__"] = getattr(_stub, "__version__", "0.0-test")
    sys.modules[__name__] = _stub
    _stub_numpy = importlib.import_module("gpt_oss._stubs.safetensors.numpy")
    sys.modules.setdefault("safetensors.numpy", _stub_numpy)
else:  # pragma: no cover - executed when safetensors is available
    globals().update(_real_safetensors.__dict__)
    __all__ = getattr(
        _real_safetensors,
        "__all__",
        [name for name in _real_safetensors.__dict__ if not name.startswith("_")],
    )
