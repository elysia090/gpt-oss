"""Compatibility shim that provides :mod:`numpy` or a lightweight stub."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_real_numpy() -> ModuleType:
    current_file = __file__
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        spec = find_spec("numpy")
        if spec is None or spec.origin in {None, current_file}:
            continue
        loader = spec.loader
        if loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    raise ModuleNotFoundError("numpy")


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
    sys.modules[__name__] = _real_numpy
