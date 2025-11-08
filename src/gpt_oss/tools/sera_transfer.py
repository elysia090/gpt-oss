"""Public wrapper re-exporting the Sera transfer implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    from . import _sera_transfer_impl as _impl
except ImportError:  # pragma: no cover - fallback for direct loading
    _impl_path = Path(__file__).resolve().with_name("_sera_transfer_impl.py")
    _impl_spec = importlib.util.spec_from_file_location(
        "gpt_oss.tools._sera_transfer_impl", _impl_path
    )
    if _impl_spec is None or _impl_spec.loader is None:
        raise
    _impl_module = importlib.util.module_from_spec(_impl_spec)
    sys.modules.setdefault(_impl_spec.name, _impl_module)
    _impl_spec.loader.exec_module(_impl_module)
    _impl = _impl_module

__all__: list[str] = []
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)
    if not _name.startswith("_"):
        __all__.append(_name)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    sys.exit(_impl.main())
