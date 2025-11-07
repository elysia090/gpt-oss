"""A tiny subset of :mod:`numpy` used by the Sera reference implementation."""

from __future__ import annotations

import math
import random as _py_random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple, Union

Number = Union[int, float]
Scalar = Union[Number, bool]
bool_ = bool
Shape = Tuple[int, ...]


def isscalar(value: object) -> bool:
    """Return ``True`` when *value* can be treated as a scalar."""

    return isinstance(value, (int, float, bool))


def _coerce_scalar(value: Scalar) -> Scalar:
    if isinstance(value, bool):
        return value
    return float(value)


def _deep_copy(data: Union[Scalar, List]) -> Union[Scalar, List]:
    if isinstance(data, list):
        return [_deep_copy(item) for item in data]
    return data


def _infer_shape(data: Union[Scalar, List]) -> Shape:
    if isinstance(data, list):
        if not data:
            return (0,)
        inner = data[0]
        if isinstance(inner, list):
            inner_shape = _infer_shape(inner)
            return (len(data),) + inner_shape
        return (len(data),)
    return ()


def _ensure_list(data: Union[Scalar, List, "ndarray"]) -> Union[Scalar, List]:
    if isinstance(data, ndarray):
        return _deep_copy(data._data)
    if isinstance(data, list):
        return [_ensure_list(item) if isinstance(item, (list, ndarray)) else _coerce_scalar(item) for item in data]
    if isinstance(data, tuple):
        return [_ensure_list(item) if isinstance(item, (list, ndarray)) else _coerce_scalar(item) for item in data]
    return _coerce_scalar(data)


def _apply_scalar(data: Union[Scalar, List], scalar: Scalar, op: Callable[[float, float], float]) -> Union[Scalar, List]:
    if isinstance(data, list):
        return [_apply_scalar(item, scalar, op) for item in data]
    return op(float(data), float(scalar))


def _apply_pair(a: Union[Scalar, List], b: Union[Scalar, List], op: Callable[[float, float], float]) -> Union[Scalar, List]:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Shape mismatch for elementwise operation")
        return [_apply_pair(x, y, op) for x, y in zip(a, b)]
    if isinstance(a, list) or isinstance(b, list):
        raise ValueError("Shape mismatch for elementwise operation")
    return op(float(a), float(b))


def _flatten(data: Union[Scalar, List]) -> List[float]:
    if isinstance(data, list):
        result: List[float] = []
        for item in data:
            result.extend(_flatten(item))
        return result
    return [float(data)]


def _compare_pair(a: Union[Scalar, List], b: Union[Scalar, List], op: Callable[[float, float], bool]) -> Union[bool, List]:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Shape mismatch for comparison")
        return [_compare_pair(x, y, op) for x, y in zip(a, b)]
    if isinstance(a, list) or isinstance(b, list):
        raise ValueError("Shape mismatch for comparison")
    return op(float(a), float(b))


def _compare_scalar(data: Union[Scalar, List], scalar: float, op: Callable[[float, float], bool]) -> Union[bool, List]:
    if isinstance(data, list):
        return [_compare_scalar(item, scalar, op) for item in data]
    return op(float(data), scalar)


class ndarray:
    """A minimal array type implementing the operations used by the tests."""

    def __init__(self, data: Union[Scalar, Sequence, "ndarray"]) -> None:
        coerced = _ensure_list(data)
        if isinstance(coerced, list):
            self._data = coerced
        else:
            self._data = [coerced]
        self._update_shape()

    def _update_shape(self) -> None:
        shape = _infer_shape(self._data)
        self._shape = shape

    # Basic protocol -------------------------------------------------
    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        prod = 1
        for dim in self._shape:
            prod *= dim
        return prod

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def tolist(self) -> List:
        return _deep_copy(self._data)

    def __iter__(self) -> Iterator:
        if self.ndim <= 1:
            return iter(self._data)
        return (ndarray(item) for item in self._data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            data = self._data
            for part in idx:
                data = data[part]
            if isinstance(data, list):
                return ndarray(data)
            return data
        data = self._data[idx]
        if isinstance(data, list):
            return ndarray(data)
        return data

    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, tuple):
            data = self._data
            for part in idx[:-1]:
                data = data[part]
            data[idx[-1]] = _ensure_list(value)
        else:
            self._data[idx] = _ensure_list(value)
        self._update_shape()

    def copy(self) -> "ndarray":
        return ndarray(_deep_copy(self._data))

    # Arithmetic -----------------------------------------------------
    def _binary(self, other, op: Callable[[float, float], float]) -> "ndarray":
        other_data = _ensure_list(other)
        if isinstance(other_data, list):
            result = _apply_pair(self._data, other_data, op)
        else:
            result = _apply_scalar(self._data, other_data, op)
        return ndarray(result)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self._data = self.__add__(other)._data
        self._update_shape()
        return self

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        other_data = _ensure_list(other)
        if isinstance(other_data, list):
            result = _apply_pair(other_data, self._data, lambda a, b: a - b)
        else:
            result = _apply_scalar(self._data, other_data, lambda a, b: other_data - a)
        return ndarray(result)

    def __isub__(self, other):
        self._data = self.__sub__(other)._data
        self._update_shape()
        return self

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self._data = self.__mul__(other)._data
        self._update_shape()
        return self

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        other_data = _ensure_list(other)
        if isinstance(other_data, list):
            result = _apply_pair(other_data, self._data, lambda a, b: a / b)
        else:
            result = _apply_scalar(self._data, other_data, lambda a, b: other_data / a)
        return ndarray(result)

    def __itruediv__(self, other):
        self._data = self.__truediv__(other)._data
        self._update_shape()
        return self

    def __neg__(self):
        return self * -1.0

    def __pow__(self, power: float):
        return self._binary(power, lambda a, b: a**b)

    # Comparisons ----------------------------------------------------
    def _compare(self, other, op: Callable[[float, float], bool]) -> "ndarray":
        other_data = _ensure_list(other)
        if isinstance(other_data, list):
            result = _compare_pair(self._data, other_data, op)
        else:
            result = _compare_scalar(self._data, float(other_data), op)
        return ndarray(result)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    # Linear algebra -------------------------------------------------
    def __matmul__(self, other):
        other_arr = asarray(other)
        if self.ndim == 1 and other_arr.ndim == 1:
            return float(dot(self, other_arr))
        if self.ndim == 1 and other_arr.ndim == 2:
            rows, cols = other_arr.shape
            if rows != len(self._data):
                raise ValueError("Shape mismatch for matmul")
            result = []
            for j in range(cols):
                column = [other_arr[i, j] for i in range(rows)]
                result.append(float(dot(self, ndarray(column))))
            return ndarray(result)
        if self.ndim == 2 and other_arr.ndim == 1:
            rows, cols = self.shape
            if cols != len(other_arr._data):
                raise ValueError("Shape mismatch for matmul")
            result = []
            for row in self:
                result.append(float(dot(row, other_arr)))
            return ndarray(result)
        if self.ndim == 2 and other_arr.ndim == 2:
            rows, cols = self.shape
            o_rows, o_cols = other_arr.shape
            if cols != o_rows:
                raise ValueError("Shape mismatch for matmul")
            data = []
            for i in range(rows):
                row = []
                for j in range(o_cols):
                    column = [other_arr[k, j] for k in range(o_rows)]
                    row.append(float(dot(self[i], ndarray(column))))
                data.append(row)
            return ndarray(data)
        raise ValueError("matmul only supports 1D and 2D arrays")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ndarray({self.tolist()!r})"


# Array constructors -------------------------------------------------

def array(data: Union[Scalar, Sequence, ndarray], dtype=None) -> ndarray:
    del dtype
    return ndarray(data)


def asarray(data: Union[Scalar, Sequence, ndarray], dtype=None) -> ndarray:
    del dtype
    if isinstance(data, ndarray):
        return data
    return ndarray(data)


def zeros(shape: Union[int, Sequence[int]], dtype=None) -> ndarray:
    del dtype
    if isinstance(shape, int):
        data = [0.0 for _ in range(shape)]
    else:
        dims = list(shape)
        if len(dims) == 1:
            data = [0.0 for _ in range(dims[0])]
        elif len(dims) == 2:
            data = [[0.0 for _ in range(dims[1])] for _ in range(dims[0])]
        else:
            raise ValueError("Only 1D or 2D zeros supported")
    return ndarray(data)


def ones(shape: Union[int, Sequence[int]], dtype=None) -> ndarray:
    del dtype
    if isinstance(shape, int):
        data = [1.0 for _ in range(shape)]
    else:
        dims = list(shape)
        if len(dims) == 1:
            data = [1.0 for _ in range(dims[0])]
        elif len(dims) == 2:
            data = [[1.0 for _ in range(dims[1])] for _ in range(dims[0])]
        else:
            raise ValueError("Only 1D or 2D ones supported")
    return ndarray(data)


def zeros_like(arr: ndarray, dtype=None) -> ndarray:
    del dtype
    shape = arr.shape
    return zeros(shape)


def ones_like(arr: ndarray, dtype=None) -> ndarray:
    del dtype
    shape = arr.shape
    return ones(shape)


def eye(n: int, dtype=None) -> ndarray:
    del dtype
    data = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        data[i][i] = 1.0
    return ndarray(data)


# Statistical helpers -----------------------------------------------

def mean(arr: Union[ndarray, Sequence[float]]) -> float:
    values = _flatten(_ensure_list(arr))
    if not values:
        return 0.0
    return sum(values) / len(values)


def dot(a: Union[ndarray, Sequence[float]], b: Union[ndarray, Sequence[float]]) -> float:
    list_a = _flatten(_ensure_list(a))
    list_b = _flatten(_ensure_list(b))
    if len(list_a) != len(list_b):
        raise ValueError("dot requires vectors of equal length")
    return sum(x * y for x, y in zip(list_a, list_b))


def outer(a: Union[ndarray, Sequence[float]], b: Union[ndarray, Sequence[float]]) -> ndarray:
    list_a = _flatten(_ensure_list(a))
    list_b = _flatten(_ensure_list(b))
    data = [[x * y for y in list_b] for x in list_a]
    return ndarray(data)


def sqrt(x: Union[ndarray, float]) -> Union[ndarray, float]:
    if isinstance(x, ndarray):
        return x._binary(1.0, lambda a, _: math.sqrt(a))
    return math.sqrt(float(x))


def exp(x: Union[ndarray, float]) -> Union[ndarray, float]:
    if isinstance(x, ndarray):
        return x._binary(1.0, lambda a, _: math.exp(a))
    return math.exp(float(x))


def abs(x: Union[ndarray, float]) -> Union[ndarray, float]:  # type: ignore[override]
    if isinstance(x, ndarray):
        return x._binary(1.0, lambda a, _: math.fabs(a))
    return math.fabs(float(x))


def clip(x: ndarray, a_min: float, a_max: float) -> ndarray:
    def _clip_value(value: float) -> float:
        if a_min is not None and value < a_min:
            return a_min
        if a_max is not None and value > a_max:
            return a_max
        return value

    def _apply(data: Union[Scalar, List]) -> Union[Scalar, List]:
        if isinstance(data, list):
            return [_apply(item) for item in data]
        return _clip_value(float(data))

    return ndarray(_apply(x._data))


def count_nonzero(x: Union[ndarray, Sequence[float]]) -> int:
    values = _flatten(_ensure_list(x))
    return sum(1 for value in values if float(value) != 0.0)


def reshape(arr: ndarray, shape: Shape) -> ndarray:
    values = _flatten(arr._data)
    if len(shape) == 1:
        if len(values) != shape[0]:
            raise ValueError("Invalid reshape")
        return ndarray(values)
    if len(shape) == 2:
        rows, cols = shape
        if len(values) != rows * cols:
            raise ValueError("Invalid reshape")
        data = []
        idx = 0
        for _ in range(rows):
            data.append(values[idx : idx + cols])
            idx += cols
        return ndarray(data)
    raise ValueError("reshape only supports 1D or 2D")


ndarray.reshape = reshape  # type: ignore[attr-defined]


# Linear algebra helpers --------------------------------------------


class _LinalgModule:
    def norm(self, arr: Union[ndarray, Sequence[float]]) -> float:
        values = _flatten(_ensure_list(arr))
        return math.sqrt(sum(value * value for value in values))


linalg = _LinalgModule()


# Random number generator -------------------------------------------


@dataclass
class _DefaultRNG:
    seed: int

    def __post_init__(self) -> None:
        self._rng = _py_random.Random(self.seed)

    def standard_normal(self, shape: Union[int, Sequence[int]]) -> ndarray:
        if isinstance(shape, int):
            data = [self._rng.gauss(0.0, 1.0) for _ in range(shape)]
            return ndarray(data)
        dims = list(shape)
        if len(dims) == 1:
            data = [self._rng.gauss(0.0, 1.0) for _ in range(dims[0])]
            return ndarray(data)
        if len(dims) == 2:
            data = [
                [self._rng.gauss(0.0, 1.0) for _ in range(dims[1])] for _ in range(dims[0])
            ]
            return ndarray(data)
        raise ValueError("standard_normal only supports up to 2 dimensions")


class _RandomModule:
    def default_rng(self, seed: int) -> _DefaultRNG:
        return _DefaultRNG(seed)

    Random = _py_random.Random


random = _RandomModule()


__all__ = [
    "array",
    "asarray",
    "bool_",
    "isscalar",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "eye",
    "mean",
    "dot",
    "outer",
    "sqrt",
    "exp",
    "abs",
    "clip",
    "count_nonzero",
    "linalg",
    "random",
    "ndarray",
    "__version__",
]

__version__ = "0.0-test"
