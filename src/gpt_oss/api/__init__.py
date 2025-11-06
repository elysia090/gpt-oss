"""Compatibility package redirecting to :mod:`gpt_oss.interfaces.api`."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Dict, Iterable, Optional

_TARGET_PACKAGE = "gpt_oss.interfaces.api"

_SUBMODULES: Iterable[str] = (
    "responses",
    "responses.api_server",
    "responses.events",
    "responses.serve",
    "responses.types",
    "responses.utils",
    "responses.inference",
    "responses.inference.metal",
    "responses.inference.ollama",
    "responses.inference.stub",
    "responses.inference.transformers",
    "responses.inference.triton",
    "responses.inference.vllm",
)


class _LazyModule(ModuleType):
    """Module proxy that defers importing the compatibility target."""

    _INTERNAL_ATTRS = {
        "_target_name",
        "_parent_name",
        "_attr_name",
        "_loaded",
        "_load",
        "_INTERNAL_ATTRS",
    }
    _RESERVED = {
        "__class__",
        "__dict__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__repr__",
        "__name__",
        "__loader__",
        "__package__",
        "__path__",
        "__spec__",
    }

    def __init__(
        self,
        name: str,
        target: str,
        parent_name: Optional[str],
        attr_name: str,
    ) -> None:
        super().__init__(name)
        super().__setattr__("_target_name", target)
        super().__setattr__("_parent_name", parent_name)
        super().__setattr__("_attr_name", attr_name)
        super().__setattr__("_loaded", None)

    def _load(self) -> ModuleType:
        module = super().__getattribute__("_loaded")
        if module is None:
            module = importlib.import_module(super().__getattribute__("_target_name"))
            module_name = super().__getattribute__("__name__")
            sys.modules[module_name] = module
            parent_name = super().__getattribute__("_parent_name")
            if parent_name:
                parent = sys.modules[parent_name]
                attr_name = super().__getattribute__("_attr_name")
                parent.__dict__[attr_name] = module
            super().__setattr__("_loaded", module)
        return module

    def __getattribute__(self, name: str):  # pragma: no cover - delegation
        if name in _LazyModule._INTERNAL_ATTRS or name in _LazyModule._RESERVED:
            return super().__getattribute__(name)
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value):  # pragma: no cover - delegation
        if name in self._INTERNAL_ATTRS or name in self._RESERVED:
            super().__setattr__(name, value)
        else:
            setattr(self._load(), name, value)

    def __dir__(self):  # pragma: no cover - simple delegation
        return dir(self._load())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<_LazyModule proxying {super().__getattribute__('_target_name')!r}>"


def _prime_aliases(names: Iterable[str]) -> None:
    """Register compatibility aliases for a collection of dotted names."""

    seen: Dict[str, ModuleType] = {}
    for dotted_name in names:
        parts = dotted_name.split(".")
        for depth in range(1, len(parts) + 1):
            partial = ".".join(parts[:depth])
            if partial in seen:
                continue
            alias_name = f"{__name__}.{partial}"
            target_name = f"{_TARGET_PACKAGE}.{partial}"
            parent_alias: Optional[str]
            if depth == 1:
                parent_alias = __name__
            else:
                parent_alias = f"{__name__}.{'.'.join(parts[: depth - 1])}"
            attr_name = parts[depth - 1]
            module = _LazyModule(alias_name, target_name, parent_alias, attr_name)
            sys.modules[alias_name] = module
            parent_module = sys.modules[parent_alias]
            parent_module.__dict__[attr_name] = module
            seen[partial] = module


sys.modules.setdefault(__name__, sys.modules[__name__])

_prime_aliases(_SUBMODULES)

_target_root = importlib.import_module(_TARGET_PACKAGE)
__all__ = getattr(_target_root, "__all__", [])
for exported_name in __all__:
    globals()[exported_name] = getattr(_target_root, exported_name)


def __getattr__(name: str):
    """Delegate attribute access to :mod:`gpt_oss.interfaces.api`."""

    return getattr(_target_root, name)


def __dir__() -> Iterable[str]:
    """Mirror the directory listing of the target package."""

    return sorted(set(__all__) | set(dir(_target_root)))
