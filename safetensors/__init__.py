"""A lightweight stub of the :mod:`safetensors` package used in the tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

__all__ = ["safe_open"]


@dataclass
class _TensorRecord:
    shape: Tuple[int, ...]
    dtype: str
    data: List[float]

    def to_nested(self) -> List:
        if not self.shape:
            return self.data[0]
        total = 1
        for dim in self.shape:
            total *= dim
        if total != len(self.data):
            raise ValueError("Malformed tensor record")

        def _unflatten(values: List[float], dims: Tuple[int, ...]) -> List:
            if not dims:
                return values[0]
            if len(dims) == 1:
                return values
            step = dims[1]
            parts = []
            for i in range(dims[0]):
                start = i * step
                end = start + step
                parts.append(_unflatten(values[start:end], dims[1:]))
            return parts

        return _unflatten(self.data, self.shape)


class _SafeOpen:
    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        self._records: Dict[str, _TensorRecord] = {}
        for name, payload in raw.get("tensors", {}).items():
            shape = tuple(int(dim) for dim in payload["shape"])
            dtype = str(payload.get("dtype", "f32"))
            data = [float(x) for x in payload["data"]]
            self._records[name] = _TensorRecord(shape=shape, dtype=dtype, data=data)

    def __enter__(self) -> "_SafeOpen":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to clean up
        return None

    def keys(self) -> Iterable[str]:
        return self._records.keys()

    def get_tensor(self, key: str):
        record = self._records[key]
        return record.to_nested()


def safe_open(path: str | Path, framework: str = "python", device: str | None = None) -> _SafeOpen:  # noqa: ARG001
    return _SafeOpen(Path(path))

