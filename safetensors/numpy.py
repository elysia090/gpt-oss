"""Minimal helpers for writing JSON based safetensor stubs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

__all__ = ["save_file"]


def _flatten(values) -> List[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, list):
        result: List[float] = []
        for item in values:
            result.extend(_flatten(item))
        return result
    if isinstance(values, tuple):
        result: List[float] = []
        for item in values:
            result.extend(_flatten(item))
        return result
    raise TypeError(f"Unsupported tensor element type: {type(values)!r}")


def _infer_shape(values) -> List[int]:
    if isinstance(values, (int, float)):
        return []
    if isinstance(values, list):
        if not values:
            return [0]
        first = values[0]
        inner_shape = _infer_shape(first)
        return [len(values)] + inner_shape
    if isinstance(values, tuple):
        if not values:
            return [0]
        first = values[0]
        inner_shape = _infer_shape(first)
        return [len(values)] + inner_shape
    raise TypeError(f"Unsupported tensor element type: {type(values)!r}")


def save_file(tensors: Dict[str, Sequence], path: Path) -> None:
    serialised = {"tensors": {}}
    for name, tensor in tensors.items():
        serialised["tensors"][name] = {
            "shape": _infer_shape(tensor),
            "dtype": "f32",
            "data": _flatten(tensor),
        }
    path.write_text(json.dumps(serialised))

