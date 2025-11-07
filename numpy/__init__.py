"""Compatibility shim that prefers the real :mod:`numpy` package."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _filter_sys_path(path_entries: Iterable[str]) -> list[str]:
    """Return *path_entries* without the repository root.

    The compatibility shim lives inside the repository, so leaving the root on
    :data:`sys.path` would cause :mod:`importlib` to rediscover this module
    instead of the distribution provided by ``pip``.  Filtering makes it
    possible to import the real package when it is installed.
    """

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


def _load_real_numpy() -> ModuleType:
    saved_module = sys.modules.pop(__name__, None)
    original_sys_path = list(sys.path)
    try:
        sys.path = _filter_sys_path(original_sys_path)
        module = importlib.import_module("numpy")
    except ModuleNotFoundError:
        if saved_module is not None:
            sys.modules[__name__] = saved_module
        raise
    finally:
        sys.path = original_sys_path

    sys.modules[__name__] = module
    return module


try:  # pragma: no cover - exercised in environments with real numpy
    _real_numpy = _load_real_numpy()
except ModuleNotFoundError:  # pragma: no cover - deterministic fallback
    stub_path = Path(__file__).resolve().parent.parent / "src" / "gpt_oss" / "_compat" / "numpy_stub.py"
    spec = importlib.util.spec_from_file_location("gpt_oss._compat.numpy_stub", stub_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError("numpy stub")
    _stub = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gpt_oss._compat.numpy_stub", _stub)
    spec.loader.exec_module(_stub)
    globals().update({name: getattr(_stub, name) for name in getattr(_stub, "__all__", [])})
    __all__ = list(getattr(_stub, "__all__", []))
    __all__.append("__version__") if "__version__" not in __all__ else None
    globals()["__version__"] = getattr(_stub, "__version__", "0.0-test")
else:  # pragma: no cover - executed when numpy is available
    globals().update(_real_numpy.__dict__)
    __all__ = getattr(_real_numpy, "__all__", [name for name in _real_numpy.__dict__ if not name.startswith("_")])
