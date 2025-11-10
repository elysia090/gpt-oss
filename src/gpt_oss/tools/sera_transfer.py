"""Deterministic conversion of checkpoints into Sera Transfer Kit artefacts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import shutil
import pickle
import re
import struct
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping as _Mapping, MutableMapping
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

_SAFETENSORS_MISSING_MSG = (
    "The `safetensors` package is required for Sera conversion. "
    "Install it with 'pip install safetensors'."
)

if importlib.util.find_spec("safetensors") is None:  # pragma: no cover - deterministic import guard
    raise ModuleNotFoundError(_SAFETENSORS_MISSING_MSG)
from safetensors import SafetensorError

_NUMPY_MISSING_MSG = (
    "The numpy package is required for Sera conversion. Install it with 'pip install numpy'."
)

if importlib.util.find_spec("numpy") is None:  # pragma: no cover - deterministic import guard
    raise ModuleNotFoundError(_NUMPY_MISSING_MSG)
import numpy as _np

if not getattr(_np, "__gpt_oss_numpy_stub__", False):
    _missing_numpy_apis = [name for name in ("stack", "ldexp", "split") if not hasattr(_np, name)]
    if _missing_numpy_apis:
        raise ModuleNotFoundError(_NUMPY_MISSING_MSG)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from numpy import ndarray as _NDArray
    from torch import Tensor as _TorchTensor

    TensorLike = list | _NDArray | _TorchTensor
else:  # pragma: no cover - runtime alias
    TensorLike = Any

_TORCH_IMPORT_ATTEMPTED = False
_TORCH_MODULE = None


def _get_torch():
    global _TORCH_IMPORT_ATTEMPTED, _TORCH_MODULE
    if not _TORCH_IMPORT_ATTEMPTED:
        _TORCH_IMPORT_ATTEMPTED = True
        try:  # pragma: no cover - optional dependency
            import torch as torch_module
        except Exception:  # pragma: no cover - defensive
            _TORCH_MODULE = None
        else:
            _TORCH_MODULE = torch_module
    return _TORCH_MODULE


def _is_torch_tensor(value: object) -> bool:
    if value is None or not _TORCH_IMPORT_ATTEMPTED:
        return False
    torch_module = _TORCH_MODULE
    return torch_module is not None and isinstance(value, torch_module.Tensor)


@dataclass
class DeterminismSettings:
    """Runtime configuration describing deterministic kernel requirements."""

    mxfp4_backend: str = "python"
    torch_threads: int | None = None
    numpy_threads: int | None = None
    assume_ftz: bool = False
    assume_fma: bool = False

    def allow_torch(self) -> bool:
        return (
            self.mxfp4_backend == "torch"
            and self.torch_threads == 1
            and self.assume_ftz
            and self.assume_fma
        )

    def allow_numpy(self) -> bool:
        return (
            self.mxfp4_backend == "numpy"
            and self.numpy_threads == 1
            and self.assume_ftz
            and self.assume_fma
        )


@dataclass
class DeterminismObserver:
    """Capture deterministic execution decisions for auditing."""

    mxfp4_backends: set[str] = field(default_factory=set)

    def record_backend(self, backend: str) -> None:
        self.mxfp4_backends.add(backend)


_CURRENT_DETERMINISM = DeterminismSettings()
_DETERMINISM_OBSERVER: DeterminismObserver | None = None


def safe_open(*args, **kwargs):
    """Proxy ``safetensors.safe_open`` so it can be safely monkeypatched in tests."""

    from safetensors import safe_open as _safe_open

    return _safe_open(*args, **kwargs)


def _resolve_callable(name: str, fallback):
    proxy = sys.modules.get("gpt_oss.tools.sera_transfer")
    if proxy is not None:
        candidate = getattr(proxy, name, None)
        if callable(candidate):
            return candidate
    return fallback


def _resolve_safe_open():
    return _resolve_callable("safe_open", safe_open)


def _resolve_load_tensors():
    return _resolve_callable("load_tensors", load_tensors)


def _load_sera_common_module():
    module_name = "_sera_common"
    sera_common_path = Path(__file__).resolve().parents[1] / "inference" / "sera_common.py"
    spec = importlib.util.spec_from_file_location(module_name, sera_common_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime helpers")
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
    else:
        module.__spec__ = spec
    spec.loader.exec_module(module)
    return module


_sera_common = _load_sera_common_module()
ARRAY_MAGIC = _sera_common.ARRAY_MAGIC
JSON_BYTES_PREFIX = _sera_common.JSON_BYTES_PREFIX
encode_snapshot_blob = _sera_common.encode_snapshot_blob
ensure_bytes = _sera_common.ensure_bytes

if getattr(_np, "__gpt_oss_numpy_stub__", False):
    _NP_ARRAY_TYPES: tuple[type, ...] = ()
else:
    _NP_ARRAY_TYPES = (_np.ndarray,) if _np is not None else ()

_MXFP4_SUFFIXES = (".blocks", ".scales")

_MXFP4_FP4_VALUES = (
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


_QKV_COMPONENT_MARKER = "__sera_qkv__"


def _qkv_component_name(base: str, role: str) -> str:
    return f"{base}{_QKV_COMPONENT_MARKER}{role}"


def _parse_qkv_component(name: str) -> Tuple[str, str] | None:
    if not isinstance(name, str) or _QKV_COMPONENT_MARKER not in name:
        return None
    base, role = name.rsplit(_QKV_COMPONENT_MARKER, 1)
    return base, role
def _load_sera_runtime():
    sera_path = Path(__file__).resolve().parents[1] / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_runtime", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sera, module.SeraConfig


def _strip_private_fields(blob):
    if isinstance(blob, dict):
        return {
            key: _strip_private_fields(value)
            for key, value in blob.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(blob, list):
        return [_strip_private_fields(value) for value in blob]
    return blob


def _config_to_dict(config):
    if dataclasses.is_dataclass(config):
        result = {}
        for field in dataclasses.fields(config):
            name = field.name
            if name.startswith("_") or not field.init:
                continue
            result[name] = _config_to_dict(getattr(config, name))
        return result
    if hasattr(config, "__dict__") and not isinstance(config, (str, bytes)):
        return {key: _config_to_dict(value) for key, value in vars(config).items()}
    if isinstance(config, dict):
        return {key: _config_to_dict(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_config_to_dict(value) for value in config]
    return config

logger = logging.getLogger(__name__)


def _active_determinism() -> DeterminismSettings:
    return _CURRENT_DETERMINISM


def _set_determinism(settings: DeterminismSettings) -> None:
    global _CURRENT_DETERMINISM
    _CURRENT_DETERMINISM = settings
    _apply_thread_configuration(settings)


def _set_determinism_observer(observer: DeterminismObserver | None) -> None:
    global _DETERMINISM_OBSERVER
    _DETERMINISM_OBSERVER = observer


def _record_mxfp4_backend(backend: str) -> None:
    observer = _DETERMINISM_OBSERVER
    if observer is not None:
        observer.record_backend(backend)


def _apply_thread_configuration(settings: DeterminismSettings) -> None:
    if settings.torch_threads is not None:
        torch_module = _get_torch()
        if torch_module is not None:
            try:
                torch_module.set_num_threads(settings.torch_threads)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Unable to set torch thread count to %s", settings.torch_threads
                )
            if settings.assume_ftz and hasattr(torch_module, "set_flush_denormal"):
                try:
                    torch_module.set_flush_denormal(True)
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("Unable to request torch flush-to-zero behaviour")
    if settings.numpy_threads is not None:
        thread_value = str(settings.numpy_threads)
        for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[env_name] = thread_value
        if hasattr(_np, "set_num_threads"):
            try:  # pragma: no cover - optional API
                _np.set_num_threads(settings.numpy_threads)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Unable to set numpy thread count to %s", thread_value)


_MODEL_CONFIG_LOG_ORDER: Tuple[str, ...] = (
    "architectures",
    "attention_bias",
    "attention_dropout",
    "eos_token_id",
    "experts_per_token",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initial_context_length",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_local_experts",
    "output_router_logits",
    "pad_token_id",
    "quantization_config",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "router_aux_loss_coef",
    "sliding_window",
    "swiglu_limit",
    "tie_word_embeddings",
    "transformers_version",
    "use_cache",
    "vocab_size",
)


def _format_model_config_keys(config: Mapping[str, object]) -> str:
    if not config:
        return "<none>"
    ordered = [key for key in _MODEL_CONFIG_LOG_ORDER if key in config]
    remaining = [key for key in config if key not in _MODEL_CONFIG_LOG_ORDER]
    keys = ordered + remaining
    return ", ".join(keys) if keys else "<none>"


def _safe_exists(path: Path) -> bool:
    """Return ``True`` if *path* exists, suppressing ``OSError`` failures."""

    try:
        return path.exists()
    except OSError:  # pragma: no cover - defensive
        # HF symlink/xet が変なタイプでも「足りない」とみなして再取得させる
        return False


def _normalise_path(path: Path) -> Path:
    """Return an absolute version of *path* tolerant of exotic links."""

    path = path.expanduser()
    try:
        return path.resolve()
    except OSError:  # pragma: no cover - exercised on Windows
        return path.absolute()


__all__ = [
    "ArrayHeader",
    "ArrayInfo",
    "ConversionSummary",
    "SnapshotInfo",
    "crc32c",
    "format_summary",
    "render_summary",
    "run_interactive_cli",
    "sha256_low64",
    "SplitMix64",
    "convert",
    "main",
]


MAGIC_SERA_ARRAY = ARRAY_MAGIC


def _json_key(value: object) -> str:
    encoded = encode_snapshot_blob(value)
    if isinstance(encoded, str):
        return encoded
    if isinstance(encoded, (bytes, bytearray, memoryview)):
        return JSON_BYTES_PREFIX + ensure_bytes(encoded).hex()
    return str(encoded)


def _json_value(value: object):
    encoded = encode_snapshot_blob(value)
    if isinstance(encoded, dict):
        return {
            _json_key(key): _json_value(item)
            for key, item in encoded.items()
        }
    if isinstance(encoded, list):
        return [_json_value(item) for item in encoded]
    if hasattr(encoded, "__dict__") and not isinstance(encoded, (str, bytes, bytearray)):
        return _json_value(vars(encoded))
    return encoded


def _flatten(values) -> List[float]:
    result: List[float] = []

    def _recurse(item) -> None:
        if isinstance(item, (list, tuple)):
            for sub in item:
                _recurse(sub)
            return
        if isinstance(item, _Bfloat16Tensor):
            _recurse(item.to_float32())
            return
        if _NP_ARRAY_TYPES and isinstance(item, _NP_ARRAY_TYPES):
            ndim = getattr(item, "ndim", 0)
            if ndim == 0:
                result.append(float(item))
            else:
                for sub in item:
                    _recurse(sub)
            return
        if _is_torch_tensor(item):
            if getattr(item, "ndim", 0) == 0:
                result.append(float(item))
            else:
                for sub in item:
                    _recurse(sub)
            return
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes, bytearray)):
            for sub in item:
                _recurse(sub)
            return
        result.append(float(item))

    _recurse(values)
    return result


def _infer_shape(values) -> Tuple[int, ...]:
    if isinstance(values, (int, float)):
        return ()
    if isinstance(values, list):
        if not values:
            return (0,)
        inner = _infer_shape(values[0])
        return (len(values),) + inner
    if isinstance(values, tuple):
        if not values:
            return (0,)
        inner = _infer_shape(values[0])
        return (len(values),) + inner
    if isinstance(values, _Bfloat16Tensor):
        return values.shape
    if _NP_ARRAY_TYPES and isinstance(values, _NP_ARRAY_TYPES):
        return tuple(int(dim) for dim in getattr(values, "shape", ()))
    if _is_torch_tensor(values):
        return tuple(int(dim) for dim in getattr(values, "shape", ()))
    raise TypeError(f"Unsupported tensor type: {type(values)!r}")


def _tensor_len(value) -> int:
    try:
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        if isinstance(value, _Bfloat16Tensor):
            shape = value.shape
            return int(shape[0]) if shape else 0
        if _NP_ARRAY_TYPES and isinstance(value, _NP_ARRAY_TYPES):
            shape = getattr(value, "shape", ())
            return int(shape[0]) if shape else 0
        if _is_torch_tensor(value):
            shape = getattr(value, "shape", ())
            return int(shape[0]) if len(shape) else 0
        return 0


# ---------------------------------------------------------------------------
# CRC utilities


CRC32C_TABLE: List[int] = []


def _init_crc32c_table() -> None:
    if CRC32C_TABLE:
        return
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1
        CRC32C_TABLE.append(crc & 0xFFFFFFFF)


def crc32c(data: bytes) -> int:
    _init_crc32c_table()
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def sha256_low64(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[-8:], "little")


@dataclass
class ArrayHeader:
    magic: int
    dtype_code: int
    rank: int
    dims: Tuple[int, int, int, int, int]
    byte_len: int
    crc32c: int
    sha256_low64: int
    flags: int
    reserved: int = 0

    HEADER_STRUCT = struct.Struct("<I H H 5Q Q Q Q I I")

    def to_bytes(self) -> bytes:
        return self.HEADER_STRUCT.pack(
            self.magic,
            self.dtype_code,
            self.rank,
            *self.dims,
            self.byte_len,
            self.crc32c,
            self.sha256_low64,
            self.flags,
            self.reserved,
        )


# ---------------------------------------------------------------------------
# Array serialisation


_DTYPE_INFO: Mapping[str, Tuple[int, str]] = {
    "f64": (1, "d"),
    "f32": (2, "f"),
    "i32": (3, "i"),
    "i16": (4, "h"),
    "i8": (5, "b"),
    "u8": (6, "B"),
}


def _pack_values(data, fmt: str) -> bytes:
    from array import array

    arr = array(fmt)
    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (list, tuple)):
            flat = _flatten(data)
            if fmt in {"f", "d"}:
                arr.extend(flat)
            else:
                arr.extend(int(round(x)) for x in flat)
        else:
            arr.extend(float(x) if fmt in {"f", "d"} else int(round(x)) for x in data)
    else:
        arr.append(float(data) if fmt in {"f", "d"} else int(round(data)))
    return arr.tobytes()


def write_array(path: Path, data, dtype: str, flags: int = 0x1) -> bytes:
    if dtype not in _DTYPE_INFO:
        raise ValueError(f"Unsupported dtype {dtype}")
    code, fmt = _DTYPE_INFO[dtype]
    payload = _pack_values(data, fmt)
    shape = _infer_shape(data)
    dims = list(shape[:5])
    while len(dims) < 5:
        dims.append(1)
    header = ArrayHeader(
        magic=MAGIC_SERA_ARRAY,
        dtype_code=code,
        rank=len(shape),
        dims=tuple(dims),
        byte_len=len(payload),
        crc32c=crc32c(payload),
        sha256_low64=sha256_low64(payload),
        flags=flags,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.to_bytes())
        f.write(payload)
    return payload


@dataclass(frozen=True)
class ArrayDigest:
    """Digest information required to describe an array payload."""

    bytes: int
    sha256: bytes


@dataclass(frozen=True)
class ArrayInfo:
    """Lightweight description of an array artefact produced during conversion."""

    name: str
    path: Path
    dtype: str
    shape: Tuple[int, ...]
    bytes: int
    sha256: str


@dataclass(frozen=True)
class SnapshotInfo:
    """Metadata for auxiliary snapshot files persisted during conversion."""

    kind: str
    path: Path


@dataclass(frozen=True)
class ConversionSummary:
    """Summary of artefacts written by :func:`convert`."""

    source: Path
    output: Path
    arrays: Tuple[ArrayInfo, ...]
    manifest_path: Path
    metadata_keys: Tuple[str, ...]
    snapshots: Tuple[SnapshotInfo, ...]
    total_bytes: int
    log_path: Path | None

    @property
    def array_count(self) -> int:
        return len(self.arrays)

    def to_dict(self) -> Dict[str, object]:
        """Serialise the summary into JSON-serialisable primitives."""

        return {
            "source": str(self.source),
            "output": str(self.output),
            "manifest_path": str(self.manifest_path),
            "metadata_keys": list(self.metadata_keys),
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "arrays": [
                {
                    "name": info.name,
                    "path": str(info.path),
                    "dtype": info.dtype,
                    "shape": list(info.shape),
                    "bytes": info.bytes,
                    "sha256": info.sha256,
                }
                for info in self.arrays
            ],
            "snapshots": [
                {"kind": snapshot.kind, "path": str(snapshot.path)}
                for snapshot in self.snapshots
            ],
            "total_bytes": self.total_bytes,
        }


def format_summary(summary: ConversionSummary) -> str:
    """Return a human-friendly multi-line summary of conversion outputs."""

    header = "Sera Transfer Conversion Summary"
    lines = [header, "=" * len(header)]
    lines.append(f"Source : {summary.source}")
    lines.append(f"Output : {summary.output}")
    lines.append(f"Manifest: {summary.manifest_path}")
    if summary.log_path:
        lines.append(f"Log file: {summary.log_path}")

    if summary.metadata_keys:
        lines.append(
            "Metadata sections: " + ", ".join(sorted(summary.metadata_keys))
        )

    lines.append("")
    lines.append("Arrays:")
    if summary.arrays:
        name_width = max(len(info.name) for info in summary.arrays)
        dtype_width = max(len(info.dtype) for info in summary.arrays)
        header_row = f"  {'Name'.ljust(name_width)}  {'DType'.ljust(dtype_width)}  Shape        Bytes"
        lines.append(header_row)
        lines.append("  " + "-" * (len(header_row) - 2))
        for info in summary.arrays:
            if info.shape:
                shape_text = " × ".join(str(dim) for dim in info.shape)
            else:
                shape_text = "scalar"
            lines.append(
                "  "
                + f"{info.name.ljust(name_width)}  {info.dtype.ljust(dtype_width)}  {shape_text:<11}  {info.bytes:>8}"
            )
    else:
        lines.append("  <no arrays written>")

    if summary.snapshots:
        lines.append("")
        lines.append("Snapshots:")
        for snapshot in summary.snapshots:
            lines.append(f"  {snapshot.kind:<8} {snapshot.path}")

    lines.append("")
    lines.append(
        f"Total arrays: {summary.array_count} | Total payload bytes: {summary.total_bytes}"
    )
    return "\n".join(lines)


def render_summary(summary: ConversionSummary, *, format: str = "table") -> str:
    """Serialise ``summary`` into the requested format.

    Parameters
    ----------
    summary:
        The conversion summary produced by :func:`convert`.
    format:
        Either ``"table"`` for the human-readable report or ``"json"`` for a
        machine-friendly representation. The check is case-insensitive.
    """

    normalized = format.lower()
    if normalized == "table":
        return format_summary(summary)
    if normalized == "json":
        return json.dumps(summary.to_dict(), indent=2, sort_keys=True)
    raise ValueError(f"Unsupported summary format: {format}")


# ---------------------------------------------------------------------------
# Deterministic PRNG


class SplitMix64:
    def __init__(self, seed: int = 0) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def gaussian_pair(self) -> Tuple[float, float]:
        u = ((self.next() >> 11) & ((1 << 53) - 1)) / float(1 << 53)
        v = ((self.next() >> 11) & ((1 << 53) - 1)) / float(1 << 53)
        u = min(max(u, 2.0 ** -52), 1.0 - 2.0 ** -52)
        radius = math.sqrt(-2.0 * math.log(u))
        angle = 2.0 * math.pi * v
        return radius * math.cos(angle), radius * math.sin(angle)


def gaussian_vector(prng: SplitMix64, length: int) -> List[float]:
    values: List[float] = []
    while len(values) < length:
        g0, g1 = prng.gaussian_pair()
        values.append(g0)
        if len(values) < length:
            values.append(g1)
    return values


# ---------------------------------------------------------------------------
# Hash helpers shared across conversion stages
# ---------------------------------------------------------------------------


def _mix64(value: int, seed: int = 0) -> int:
    z = (value + seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _tensor_bytes(tensor: TensorLike) -> bytes:
    from array import array

    flat = _flatten(tensor)
    buf = array("d")
    buf.extend(float(x) for x in flat)
    return buf.tobytes()


def _hash_bytes(data: bytes) -> int:
    """Return the low 64 bits of the SHA-256 digest of ``data``."""

    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little", signed=False)


def _assign_positions(
    options: Sequence[Tuple[int, int]], occupied: Sequence[bool]
) -> Optional[List[int]]:
    placement: List[int] = []
    used: set[int] = set()
    for pos0, pos1 in options:
        choice: Optional[int] = None
        if 0 <= pos0 < len(occupied) and not occupied[pos0] and pos0 not in used:
            choice = pos0
        elif 0 <= pos1 < len(occupied) and not occupied[pos1] and pos1 not in used:
            choice = pos1
        if choice is None:
            return None
        used.add(choice)
        placement.append(choice)
    return placement


def _build_two_level_mph(keys: Sequence[int]) -> Tuple[List[int], List[int], Dict[int, int]]:
    """Construct a deterministic two-level minimal perfect hash (spec §4.4).

    The helper mirrors the construction used in both the tokenizer and the
    sparse linear dictionary.  ``keys`` are u64 identifiers.  The function
    returns a tuple ``(seeds, ordered_keys, slot_lookup)`` where ``seeds`` is
    the table of bucket seeds, ``ordered_keys`` are the keys arranged in slot
    order, and ``slot_lookup`` maps each key to its slot index.
    """

    capacity = len(keys)
    if capacity == 0:
        return [], [], {}

    table_size = max(1, int(math.ceil(1.23 * capacity)))
    seeds = [0 for _ in range(table_size)]
    slots: List[Optional[int]] = [None for _ in range(capacity)]
    occupied = [False for _ in range(capacity)]

    buckets: Dict[int, List[int]] = {}
    for key in keys:
        bucket = _mix64(key) % table_size
        buckets.setdefault(bucket, []).append(key)

    slot_lookup: Dict[int, int] = {}
    ordered_buckets = sorted(
        buckets.items(), key=lambda item: (len(item[1]), item[0]), reverse=True
    )
    for bucket, bucket_keys in ordered_buckets:
        local_keys = sorted(bucket_keys)
        for seed_candidate in range(1 << 16):
            options: List[Tuple[int, int]] = []
            for key in local_keys:
                pos0 = _mix64(key, seed_candidate) % capacity
                pos1 = _mix64(key, seed_candidate + 1) % capacity
                options.append((pos0, pos1))
            placement = _assign_positions(options, occupied)
            if placement is None:
                continue
            seeds[bucket] = seed_candidate
            for key, slot in zip(local_keys, placement):
                slots[slot] = key
                occupied[slot] = True
                slot_lookup[key] = slot
            break
        else:  # pragma: no cover - deterministic bound ensures termination
            raise RuntimeError("Unable to build minimal perfect hash (seed cap reached)")

    ordered_keys = [key for key in slots if key is not None]
    return seeds, ordered_keys, slot_lookup


# ---------------------------------------------------------------------------
# Linear algebra helpers


_HOUSEHOLDER_TOLERANCE = 1e-12


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    if not A or not B:
        return []
    rows = len(A)
    cols = len(B[0])
    inner = len(B)
    result: List[List[float]] = []
    for i in range(rows):
        row: List[float] = []
        for j in range(cols):
            s = 0.0
            for k in range(inner):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result


def _prepare_householder_column(column: Iterable[float], rows: int) -> List[float]:
    values: List[float] = []
    for idx, value in enumerate(column):
        if idx >= rows:
            break
        values.append(float(value))
    if len(values) < rows:
        values.extend([0.0] * (rows - len(values)))
    return values


def _assemble_columns(columns: Sequence[List[float]], rows: int) -> List[List[float]]:
    if rows <= 0 or not columns:
        return []
    matrix: List[List[float]] = []
    for row_idx in range(rows):
        matrix.append([column[row_idx] for column in columns])
    return matrix


def _householder_qr(
    matrix: List[List[float]], eps: float = _HOUSEHOLDER_TOLERANCE
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows and matrix[0] else 0
    Q: List[List[float]] = [
        [1.0 if i == j else 0.0 for j in range(rows)] for i in range(rows)
    ]
    R: List[List[float]] = [row[:] for row in matrix]
    diag: List[float] = []
    for k in range(min(rows, cols)):
        norm_x = 0.0
        for i in range(k, rows):
            value = R[i][k]
            norm_x += value * value
        norm_x = math.sqrt(norm_x)
        if norm_x < eps:
            diag.append(0.0)
            continue
        x0 = R[k][k]
        sign = -1.0 if x0 >= 0.0 else 1.0
        alpha = sign * norm_x
        v = [0.0] * rows
        v[k] = x0 - alpha
        for i in range(k + 1, rows):
            v[i] = R[i][k]
        norm_v = math.sqrt(sum(v[i] * v[i] for i in range(k, rows)))
        if norm_v < eps:
            diag.append(0.0)
            continue
        inv_norm_v = 1.0 / norm_v
        for i in range(k, rows):
            v[i] *= inv_norm_v
        for j in range(k, cols):
            dot = 0.0
            for i in range(k, rows):
                dot += v[i] * R[i][j]
            for i in range(k, rows):
                R[i][j] -= 2.0 * v[i] * dot
        for j in range(rows):
            dot = 0.0
            for i in range(k, rows):
                dot += v[i] * Q[i][j]
            for i in range(k, rows):
                Q[i][j] -= 2.0 * v[i] * dot
        diag.append(R[k][k])
    return Q, R, diag


def orthonormal_matrix(
    rows: int, cols: int, generator: Iterable[Iterable[float]]
) -> List[List[float]]:
    if rows <= 0 or cols <= 0:
        return []
    selected: List[List[float]] = []
    basis_q: List[List[float]] | None = None
    for column in generator:
        vec = _prepare_householder_column(column, rows)
        candidate = selected + [vec]
        matrix = _assemble_columns(candidate, rows)
        q_matrix, _r_matrix, diag = _householder_qr(matrix)
        diag_index = len(candidate) - 1
        if diag_index >= len(diag):
            raise ValueError(
                "Unable to construct orthonormal basis (columns exceed row count)"
            )
        if abs(diag[diag_index]) <= _HOUSEHOLDER_TOLERANCE:
            continue
        selected.append(vec)
        basis_q = q_matrix
        if len(selected) == cols:
            break
    if len(selected) < cols or basis_q is None:
        raise ValueError(
            "Unable to construct orthonormal basis (insufficient independent vectors)"
        )
    q_full = transpose(basis_q)
    return [row[:cols] for row in q_full]


# ---------------------------------------------------------------------------
# Model configuration


@dataclass
class LayerConfig:
    name: str
    w_q: str
    w_k: str
    w_v: str
    w_o: str
    w1: str
    w2: str
    b1: str
    b2: str


def _tensor_shape(tensor: object) -> Tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is not None:
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    if isinstance(tensor, (list, tuple)):
        try:
            return _infer_shape(tensor)
        except TypeError:  # pragma: no cover - defensive
            return ()
    return ()


def _mxfp4_components(name: str) -> Tuple[str, str, str] | None:
    if not isinstance(name, str):
        return None

    for suffix in _MXFP4_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            return base, f"{base}.blocks", f"{base}.scales"
    return None


def _as_nested_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return [_as_nested_list(item) for item in value]
    if _NP_ARRAY_TYPES and isinstance(value, _NP_ARRAY_TYPES):
        return value.tolist()
    if _is_torch_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _decode_mxfp4_torch(blocks: "_TorchTensor", scales: TensorLike):
    torch_module = _get_torch()
    if torch_module is None:
        raise RuntimeError("Torch support is not available for MXFP4 decoding")
    blocks_tensor = blocks.to(dtype=torch_module.uint8)
    if _is_torch_tensor(scales):
        scales_tensor = scales.to(dtype=torch_module.int32, device=blocks_tensor.device)
    else:
        scales_tensor = torch_module.as_tensor(
            scales, dtype=torch_module.int32, device=blocks_tensor.device
        )
    scales_tensor = scales_tensor - 127
    lut = torch_module.tensor(
        _MXFP4_FP4_VALUES, dtype=torch_module.float32, device=blocks_tensor.device
    )
    prefix_shape = blocks_tensor.shape[:-1]
    idx_lo = (blocks_tensor & 0x0F).to(dtype=torch_module.long)
    idx_hi = (blocks_tensor >> 4).to(dtype=torch_module.long)
    decoded = torch_module.stack((lut[idx_lo], lut[idx_hi]), dim=-1)
    decoded = torch_module.ldexp(decoded, scales_tensor.reshape(*prefix_shape, 1, 1))
    return decoded.reshape(*prefix_shape, blocks_tensor.shape[-1] * 2)


def _decode_mxfp4_numpy(blocks: "_NDArray", scales: TensorLike):
    blocks_array = _np.asarray(blocks, dtype=_np.uint8)
    scales_array = _np.asarray(scales, dtype=_np.int32) - 127
    lut = _np.asarray(_MXFP4_FP4_VALUES, dtype=_np.float32)
    prefix_shape = blocks_array.shape[:-1]
    idx_lo = blocks_array & 0x0F
    idx_hi = blocks_array >> 4
    decoded = _np.stack((lut[idx_lo], lut[idx_hi]), axis=-1)
    decoded = _np.ldexp(decoded, scales_array.reshape(*prefix_shape, 1, 1))
    return decoded.reshape(*prefix_shape, blocks_array.shape[-1] * 2)


def _decode_mxfp4_python(blocks: TensorLike, scales: TensorLike):
    blocks_list = _as_nested_list(blocks)
    scales_list = _as_nested_list(scales)

    def _decode_recursive(block_part, scale_part):
        if isinstance(block_part, list) and block_part and isinstance(block_part[0], list):
            return [
                _decode_recursive(sub_block, sub_scale)
                for sub_block, sub_scale in zip(block_part, scale_part)
            ]
        if not isinstance(block_part, list):
            return []
        scale_value = int(scale_part) - 127
        row: List[float] = []
        for byte in block_part:
            byte_value = int(byte) & 0xFF
            lo = _MXFP4_FP4_VALUES[byte_value & 0x0F]
            hi = _MXFP4_FP4_VALUES[(byte_value >> 4) & 0x0F]
            row.append(math.ldexp(lo, scale_value))
            row.append(math.ldexp(hi, scale_value))
        return row

    return _decode_recursive(blocks_list, scales_list)


def _decode_mxfp4_pair(blocks: TensorLike, scales: TensorLike):
    settings = _active_determinism()
    torch_module = None
    if settings.allow_torch():
        torch_module = _get_torch()
    if torch_module is not None and isinstance(blocks, torch_module.Tensor):
        _record_mxfp4_backend("torch")
        return _decode_mxfp4_torch(blocks, scales)
    if _is_torch_tensor(blocks):
        blocks = _as_nested_list(blocks)
        scales = _as_nested_list(scales)
    if _NP_ARRAY_TYPES and isinstance(blocks, _NP_ARRAY_TYPES):
        if settings.allow_numpy():
            _record_mxfp4_backend("numpy")
            return _decode_mxfp4_numpy(blocks, scales)
        blocks = _as_nested_list(blocks)
        scales = _as_nested_list(scales)
    if _is_torch_tensor(scales):
        scales = _as_nested_list(scales)
    if _NP_ARRAY_TYPES and isinstance(scales, _NP_ARRAY_TYPES):
        scales = _as_nested_list(scales)
    _record_mxfp4_backend("python")
    return _decode_mxfp4_python(blocks, scales)


def _split_qkv_tensor(
    tensor: TensorLike,
    q_dim: int,
    k_dim: int,
    v_dim: int,
):
    if min(q_dim, k_dim, v_dim) <= 0:
        raise ValueError("Q, K, and V projection dimensions must be positive for split")

    shape = _tensor_shape(tensor)
    if len(shape) < 2:
        raise ValueError("QKV tensor must be at least 2D to split into components")

    rows, cols = shape[0], shape[1]
    total = q_dim + k_dim + v_dim
    axis: int | None = None

    if rows == total:
        axis = 0
    elif cols == total:
        axis = 1

    if axis is None:
        raise ValueError(
            "Unable to split QKV tensor: expected dimensions compatible with split "
            f"[{q_dim}, {k_dim}, {v_dim}] but received shape {shape}"
        )

    if _is_torch_tensor(tensor):
        chunks = tensor.split((q_dim, k_dim, v_dim), dim=axis)
        if len(chunks) != 3:
            raise ValueError("QKV tensor did not split into 3 components as expected")
        return tuple(chunks)

    if _NP_ARRAY_TYPES and isinstance(tensor, _NP_ARRAY_TYPES):
        indices = (q_dim, q_dim + k_dim)
        chunks = _np.split(tensor, indices, axis=axis)
        if len(chunks) != 3:
            raise ValueError("QKV tensor did not split into 3 components as expected")
        return tuple(chunks)

    matrix = _as_nested_list(tensor)
    if axis == 0:
        q_rows = matrix[:q_dim]
        k_rows = matrix[q_dim : q_dim + k_dim]
        v_rows = matrix[q_dim + k_dim : total]
        return (
            [row[:] for row in q_rows],
            [row[:] for row in k_rows],
            [row[:] for row in v_rows],
        )

    if not matrix or not matrix[0]:
        raise ValueError("Unable to split empty QKV tensor")

    return (
        [row[0:q_dim] for row in matrix],
        [row[q_dim : q_dim + k_dim] for row in matrix],
        [row[q_dim + k_dim : total] for row in matrix],
    )


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    head_dim: int
    vocab_size: int
    tau: float
    layers: List[LayerConfig]
    rope_theta: float | None = None
    num_hidden_layers: int | None = None
    num_experts: int | None = None
    experts_per_token: int | None = None
    intermediate_size: int | None = None
    swiglu_limit: float | None = None
    num_key_value_heads: int | None = None
    sliding_window: int | None = None
    initial_context_length: int | None = None
    rope_scaling_factor: float | None = None
    rope_ntk_alpha: float | None = None
    rope_ntk_beta: float | None = None

    @staticmethod
    def from_dict(
        data: Mapping[str, object],
        tensors: Mapping[str, object] | None = None,
    ) -> "ModelConfig":
        raw = dict(data)

        def _select(*keys: str) -> object | None:
            for key in keys:
                if key in raw:
                    return raw[key]
            return None

        def _coerce_int(name: str, value: object | None) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Model config field '{name}' must be an integer, got {value!r}"
                ) from exc

        def _coerce_float(name: str, value: object | None) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Model config field '{name}' must be a float, got {value!r}"
                ) from exc

        d_model = _coerce_int(
            "d_model",
            raw.get("d_model")
            or _select("hidden_size", "n_embd", "model_dim", "dim_model"),
        )

        n_heads = _coerce_int(
            "n_heads",
            raw.get("n_heads")
            or _select(
                "num_attention_heads",
                "num_heads",
                "n_head",
                "decoder_attention_heads",
            ),
        )

        kv_heads = _coerce_int(
            "num_key_value_heads",
            _select("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
        )

        head_dim = _coerce_int(
            "head_dim",
            raw.get("head_dim") or _select("head_dim", "attention_head_size"),
        )

        if n_heads == 0:
            raise ValueError("Model config field 'n_heads' must be greater than zero")

        if head_dim is None and d_model is not None and n_heads is not None and n_heads:
            if d_model % n_heads == 0:
                head_dim = d_model // n_heads
            elif kv_heads:
                if d_model % kv_heads == 0:
                    head_dim = d_model // kv_heads
        if head_dim is None and d_model is not None and n_heads is not None:
            doc_hint = "docs/operations/sera-transfer.md"
            raise ValueError(
                "Model config does not provide 'head_dim'. Provide the Sera schema "
                "(d_model/n_heads/head_dim) or a Hugging Face config with "
                "hidden_size/num_attention_heads/head_dim. See "
                f"{doc_hint} for details."
            )

        if d_model is None or n_heads is None or head_dim is None:
            doc_hint = "docs/operations/sera-transfer.md"
            raise ValueError(
                "Model config must define d_model, n_heads, and head_dim (or provide "
                "their Hugging Face equivalents such as hidden_size, "
                "num_attention_heads, and head_dim). See "
                f"{doc_hint} for the expected schema."
            )

        vocab_size = int(data.get("vocab_size", 0) or 0)
        tau = float(data.get("tau", 1.0))
        rope_theta = data.get("rope_theta")
        num_hidden_layers = _coerce_int("num_hidden_layers", raw.get("num_hidden_layers"))
        num_experts = _coerce_int("num_experts", raw.get("num_experts"))
        experts_per_token = _coerce_int("experts_per_token", raw.get("experts_per_token"))
        intermediate_size = _coerce_int("intermediate_size", raw.get("intermediate_size"))
        swiglu_limit = _coerce_float("swiglu_limit", raw.get("swiglu_limit"))
        sliding_window = _coerce_int("sliding_window", raw.get("sliding_window"))
        initial_context_length = _coerce_int(
            "initial_context_length", raw.get("initial_context_length")
        )
        rope_scaling_factor = _coerce_float(
            "rope_scaling_factor", raw.get("rope_scaling_factor")
        )
        rope_ntk_alpha = _coerce_float("rope_ntk_alpha", raw.get("rope_ntk_alpha"))
        rope_ntk_beta = _coerce_float("rope_ntk_beta", raw.get("rope_ntk_beta"))
        if num_hidden_layers is None and "n_layers" in raw:
            num_hidden_layers = _coerce_int("n_layers", raw.get("n_layers"))
        if num_hidden_layers is None and "num_layers" in raw:
            num_hidden_layers = _coerce_int("num_layers", raw.get("num_layers"))
        if intermediate_size is None and "ffn_dim" in raw:
            intermediate_size = _coerce_int("ffn_dim", raw.get("ffn_dim"))

        layer_entries = list(data.get("layers", []) or [])
        layers: List[LayerConfig]
        if layer_entries:
            layers = []
            for idx, layer in enumerate(layer_entries):
                layer_data = layer  # type: ignore[assignment]
                layers.append(
                    LayerConfig(
                        name=str(layer_data.get("name", f"layer_{idx}")),
                        w_q=str(layer_data.get("W_Q", layer_data["W_K"])),
                        w_k=str(layer_data["W_K"]),
                        w_v=str(layer_data.get("W_V", layer_data["W_K"])),
                        w_o=str(layer_data["W_O"]),
                        w1=str(layer_data["FFN_W1"]),
                        w2=str(layer_data["FFN_W2"]),
                        b1=str(layer_data["FFN_B1"]),
                        b2=str(layer_data["FFN_B2"]),
                    )
                )
        else:
            if tensors is None:
                raise ValueError(
                    "Model config is missing 'layers' and no tensor map was provided"
                )
            layers = ModelConfig._infer_layers_from_tensors(
                tensors,
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
            )

        if tensors is not None:
            head_dim, kv_heads = ModelConfig._materialize_qkv_layers(
                layers,
                tensors,
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                num_key_value_heads=kv_heads,
            )
            ModelConfig._materialize_mxfp4_layers(layers, tensors)

        if num_hidden_layers is None:
            num_hidden_layers = len(layers)
        elif num_hidden_layers != len(layers):
            logger.warning(
                "Model config reports %s hidden layers but %s were inferred",
                num_hidden_layers,
                len(layers),
            )

        return ModelConfig(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            tau=tau,
            layers=layers,
            rope_theta=float(rope_theta) if rope_theta is not None else None,
            num_hidden_layers=num_hidden_layers,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            intermediate_size=intermediate_size,
            swiglu_limit=swiglu_limit,
            num_key_value_heads=kv_heads,
            sliding_window=sliding_window,
            initial_context_length=initial_context_length,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=rope_ntk_alpha,
            rope_ntk_beta=rope_ntk_beta,
        )

    @staticmethod
    def _infer_layers_from_tensors(
        tensors: Mapping[str, object],
        *,
        d_model: int,
        n_heads: int,
        head_dim: int,
    ) -> List[LayerConfig]:
        role_suffixes = {
            "W_Q": (
                "attn.q.weight",
                "attn.wq.weight",
                "attn.q_proj.weight",
                "attn.qproj.weight",
                "attention.q.weight",
                "attention.wq.weight",
                "attention.q_proj.weight",
                "attention.qproj.weight",
                "self_attn.q_proj.weight",
                "self_attn.qproj.weight",
                "q_proj.weight",
                "qproj.weight",
                "query_proj.weight",
                "queryproj.weight",
                "query.weight",
                "q_linear.weight",
                "q.weight",
            ),
            "W_K": (
                "attn.k.weight",
                "attn.wk.weight",
                "attn.k_proj.weight",
                "attn.kproj.weight",
                "attention.k.weight",
                "attention.wk.weight",
                "attention.k_proj.weight",
                "attention.kproj.weight",
                "self_attn.k_proj.weight",
                "self_attn.kproj.weight",
                "k_proj.weight",
                "kproj.weight",
                "key_proj.weight",
                "keyproj.weight",
                "key.weight",
                "k_linear.weight",
                "k.weight",
            ),
            "W_V": (
                "attn.v.weight",
                "attn.wv.weight",
                "attn.v_proj.weight",
                "attn.vproj.weight",
                "attention.v.weight",
                "attention.wv.weight",
                "attention.v_proj.weight",
                "attention.vproj.weight",
                "self_attn.v_proj.weight",
                "self_attn.vproj.weight",
                "v_proj.weight",
                "vproj.weight",
                "value_proj.weight",
                "valueproj.weight",
                "value.weight",
                "v_linear.weight",
                "v.weight",
            ),
            "W_O": (
                "attn.o.weight",
                "attn.wo.weight",
                "attn.o_proj.weight",
                "attn.oproj.weight",
                "attn.out.weight",
                "attention.o.weight",
                "attention.wo.weight",
                "attention.o_proj.weight",
                "attention.out_proj.weight",
                "attention.out.weight",
                "attention.oproj.weight",
                "attention.outproj.weight",
                "self_attn.o_proj.weight",
                "self_attn.oproj.weight",
                "o_proj.weight",
                "oproj.weight",
                "out_proj.weight",
                "outproj.weight",
                "c_proj.weight",
                "cproj.weight",
                "o_linear.weight",
                "o.weight",
            ),
            "FFN_W1": (
                "ffn.w1.weight",
                "mlp.w1.weight",
                "ffn.gate_proj.weight",
                "mlp.gate_proj.weight",
                "ffn.up_proj.weight",
                "mlp.up_proj.weight",
                "feed_forward.w1.weight",
                "feed_forward.up_proj.weight",
                "ffn.wi.weight",
                "mlp.wi.weight",
                "ffn.gate.weight",
                "mlp.gate.weight",
                "ffn.upproj.weight",
                "mlp.upproj.weight",
                "up_proj.weight",
                "upproj.weight",
                "gate_proj.weight",
                "gate.weight",
                "w1.weight",
                "wi.weight",
                "mlp.mlp1_weight.blocks",
                "mlp.mlp1_weight.scales",
            ),
            "FFN_W2": (
                "ffn.w2.weight",
                "mlp.w2.weight",
                "ffn.down_proj.weight",
                "mlp.down_proj.weight",
                "ffn.proj.weight",
                "feed_forward.w2.weight",
                "feed_forward.down_proj.weight",
                "ffn.wo.weight",
                "mlp.wo.weight",
                "ffn.downproj.weight",
                "mlp.downproj.weight",
                "down_proj.weight",
                "downproj.weight",
                "proj.weight",
                "w2.weight",
                "wo.weight",
                "mlp.mlp2_weight.blocks",
                "mlp.mlp2_weight.scales",
            ),
            "FFN_B1": (
                "ffn.w1.bias",
                "mlp.w1.bias",
                "ffn.gate_proj.bias",
                "mlp.gate_proj.bias",
                "ffn.up_proj.bias",
                "mlp.up_proj.bias",
                "feed_forward.w1.bias",
                "feed_forward.up_proj.bias",
                "ffn.wi.bias",
                "mlp.wi.bias",
                "ffn.gate.bias",
                "mlp.gate.bias",
                "up_proj.bias",
                "gate_proj.bias",
                "gate.bias",
                "w1.bias",
                "wi.bias",
                "mlp.mlp1_bias",
            ),
            "FFN_B2": (
                "ffn.w2.bias",
                "mlp.w2.bias",
                "ffn.down_proj.bias",
                "mlp.down_proj.bias",
                "ffn.proj.bias",
                "feed_forward.w2.bias",
                "feed_forward.down_proj.bias",
                "ffn.wo.bias",
                "mlp.wo.bias",
                "down_proj.bias",
                "w2.bias",
                "wo.bias",
                "mlp.mlp2_bias",
            ),
        }

        expected_proj = n_heads * head_dim
        layer_map: Dict[str, Dict[str, str]] = {}
        role_priorities: Dict[str, Dict[str, int]] = {}

        qkv_suffixes = (
            "attn.qkv.weight",
            "attention.qkv.weight",
            "self_attn.qkv.weight",
            "qkv.weight",
            "qkv_proj.weight",
        )
        qkv_suffix_tokens = [
            tuple(suffix.lower().split(".")) for suffix in qkv_suffixes
        ]
        qkv_sources: Dict[str, str] = {}

        role_suffix_tokens = {
            role: [tuple(suffix.lower().split(".")) for suffix in suffixes]
            for role, suffixes in role_suffixes.items()
        }
        mxfp_priority_suffix_tokens = {
            "FFN_W1": {
                tuple("mlp.mlp1_weight.blocks".split(".")),
                tuple("mlp.mlp1_weight.scales".split(".")),
            },
            "FFN_W2": {
                tuple("mlp.mlp2_weight.blocks".split(".")),
                tuple("mlp.mlp2_weight.scales".split(".")),
            },
            "FFN_B1": {tuple("mlp.mlp1_bias".split("."))},
            "FFN_B2": {tuple("mlp.mlp2_bias".split("."))},
        }

        for name, tensor in tensors.items():
            if not isinstance(name, str):
                continue

            shape = _tensor_shape(tensor)
            name_tokens = name.split(".")
            lower_tokens = [token.lower() for token in name_tokens]

            matched_qkv = False
            for suffix_tokens in qkv_suffix_tokens:
                if len(lower_tokens) < len(suffix_tokens):
                    continue
                tail_tokens = lower_tokens[-len(suffix_tokens) :]
                if not ModelConfig._tokens_match(tail_tokens, suffix_tokens):
                    continue
                prefix_tokens = name_tokens[: -len(suffix_tokens)]
                prefix = ".".join(prefix_tokens).rstrip(".")
                if prefix:
                    qkv_sources.setdefault(prefix, name)
                matched_qkv = True
                break

            if matched_qkv:
                continue

            for role, suffix_token_sets in role_suffix_tokens.items():
                matched = False
                for suffix_tokens in suffix_token_sets:
                    if len(lower_tokens) < len(suffix_tokens):
                        continue

                    tail_tokens = lower_tokens[-len(suffix_tokens) :]
                    if not ModelConfig._tokens_allow_role(
                        role, tail_tokens, suffix_tokens, shape, expected_proj
                    ):
                        continue

                    prefix_tokens = name_tokens[: -len(suffix_tokens)]
                    prefix = ".".join(prefix_tokens).rstrip(".")
                    if not prefix:
                        continue

                    layer_roles = layer_map.setdefault(prefix, {})
                    priorities = role_priorities.setdefault(prefix, {})
                    suffix_key = tuple(suffix_tokens)
                    priority = (
                        1
                        if suffix_key in mxfp_priority_suffix_tokens.get(role, set())
                        else 0
                    )
                    existing_priority = priorities.get(role, -1)
                    if priority > existing_priority:
                        layer_roles[role] = name
                        priorities[role] = priority
                    matched = True
                    break

                if matched:
                    break

        for prefix, base_name in qkv_sources.items():
            layer_roles = layer_map.setdefault(prefix, {})
            for role in ("W_Q", "W_K", "W_V"):
                layer_roles.setdefault(role, _qkv_component_name(base_name, role))

        required_roles = {
            "W_Q",
            "W_K",
            "W_V",
            "W_O",
            "FFN_W1",
            "FFN_W2",
            "FFN_B1",
            "FFN_B2",
        }
        incomplete = [
            prefix
            for prefix, roles in layer_map.items()
            if not required_roles.issubset(roles)
        ]
        if incomplete:
            missing_details = {
                prefix: sorted(required_roles - layer_map[prefix].keys())
                for prefix in incomplete
            }
            raise ValueError(
                "Unable to infer layer configuration from tensors; missing roles: "
                f"{missing_details}"
            )

        if not layer_map:
            raise ValueError("No transformer layers could be inferred from tensor names")

        def sort_key(item: Tuple[str, Dict[str, str]]) -> Tuple[int, str]:
            prefix, _ = item
            match = re.search(r"(?:^|\.)(\d+)(?!.*\d)", prefix)
            if match:
                return (int(match.group(1)), prefix)
            return (math.inf, prefix)

        layers: List[LayerConfig] = []
        for prefix, roles in sorted(layer_map.items(), key=sort_key):
            layers.append(
                LayerConfig(
                    name=prefix.replace(".", "_"),
                    w_q=roles["W_Q"],
                    w_k=roles["W_K"],
                    w_v=roles["W_V"],
                    w_o=roles["W_O"],
                    w1=roles["FFN_W1"],
                    w2=roles["FFN_W2"],
                    b1=roles["FFN_B1"],
                    b2=roles["FFN_B2"],
                )
            )
        return layers

    @staticmethod
    def _materialize_qkv_layers(
        layers: Sequence[LayerConfig],
        tensors: Mapping[str, TensorLike],
        *,
        d_model: int,
        n_heads: int,
        head_dim: int,
        num_key_value_heads: int | None,
    ) -> Tuple[int, int | None]:
        if not isinstance(tensors, MutableMapping):
            raise TypeError(
                "Tensor mapping must be mutable to decode fused QKV weights"
            )

        kv_heads = num_key_value_heads
        head_dim_local = head_dim
        cache: Dict[str, Dict[str, TensorLike]] = {}

        for layer in layers:
            for attr, role in (("w_q", "W_Q"), ("w_k", "W_K"), ("w_v", "W_V")):
                name = getattr(layer, attr)
                parsed = _parse_qkv_component(name)
                if parsed is None:
                    continue
                base, parsed_role = parsed
                components = cache.get(base)
                if components is None:
                    tensor = tensors.get(base)
                    if tensor is None:
                        raise ValueError(
                            "Unable to decode fused QKV tensor; missing source "
                            f"'{base}' for role '{parsed_role}'"
                        )
                    direct_split = (
                        kv_heads is not None
                        and kv_heads > 0
                        and head_dim_local is not None
                        and head_dim_local > 0
                    )
                    if direct_split:
                        q_dim = n_heads * head_dim_local
                        kv_dim = kv_heads * head_dim_local
                        try:
                            q, k, v = _split_qkv_tensor(
                                tensor, q_dim, kv_dim, kv_dim
                            )
                        except ValueError:
                            direct_split = False
                    if not direct_split:
                        inferred = ModelConfig._infer_qkv_split_dims(
                            tensor,
                            d_model=d_model,
                            n_heads=n_heads,
                            kv_heads=kv_heads,
                            head_dim_hint=head_dim_local,
                        )
                        if inferred is None:
                            raise
                        q_dim, kv_dim, head_dim_local, inferred_kv_heads = inferred
                        kv_heads = inferred_kv_heads
                        q, k, v = _split_qkv_tensor(tensor, q_dim, kv_dim, kv_dim)
                    components = {"W_Q": q, "W_K": k, "W_V": v}
                    cache[base] = components
                component = components.get(parsed_role)
                if component is None:
                    raise ValueError(
                        "Fused QKV tensor did not provide component for role "
                        f"'{parsed_role}'"
                    )
                component_name = _qkv_component_name(base, parsed_role)
                tensors[component_name] = component
                setattr(layer, attr, component_name)

        return head_dim_local, kv_heads

    @staticmethod
    def _infer_qkv_split_dims(
        tensor: TensorLike,
        *,
        d_model: int,
        n_heads: int,
        kv_heads: int | None,
        head_dim_hint: int | None,
    ) -> Tuple[int, int, int, int] | None:
        shape = _tensor_shape(tensor)
        if len(shape) < 2:
            return None

        dims = [shape[0], shape[1]]
        fused_dim: int | None = None
        for dim in dims:
            if dim != d_model:
                fused_dim = dim
                break
        if fused_dim is None:
            fused_dim = dims[0]

        candidates: List[int] = []
        seen: set[int] = set()

        if kv_heads is not None and kv_heads > 0:
            candidates.append(kv_heads)
            seen.add(kv_heads)

        limit = int(math.sqrt(n_heads))
        divisor_candidates: List[int] = []
        for divisor in range(1, limit + 1):
            if n_heads % divisor == 0:
                divisor_candidates.append(divisor)
                complement = n_heads // divisor
                if complement != divisor:
                    divisor_candidates.append(complement)

        for candidate in sorted(divisor_candidates):
            if candidate <= 0 or candidate in seen:
                continue
            candidates.append(candidate)
            seen.add(candidate)

        fallback: List[Tuple[int, int, int, int]] = []

        for candidate in candidates:
            total_heads = n_heads + 2 * candidate
            if total_heads <= 0:
                continue
            if fused_dim % total_heads != 0:
                continue

            head_dim = fused_dim // total_heads
            if head_dim <= 0:
                continue

            q_dim = head_dim * n_heads
            kv_dim = head_dim * candidate

            try:
                _split_qkv_tensor(tensor, q_dim, kv_dim, kv_dim)
            except ValueError:
                continue

            result = (q_dim, kv_dim, head_dim, candidate)
            if head_dim_hint is None or head_dim == head_dim_hint:
                return result
            fallback.append(result)

        if fallback:
            return fallback[0]

        return None

    @staticmethod
    def _materialize_mxfp4_layers(
        layers: Sequence[LayerConfig], tensors: Mapping[str, TensorLike]
    ) -> None:
        if not isinstance(tensors, MutableMapping):
            raise TypeError(
                "Tensor mapping must be mutable to decode MXFP4 weights"
            )

        cache: Dict[str, TensorLike] = {}
        for layer in layers:
            for attr in ("w1", "w2"):
                name = getattr(layer, attr)
                components = _mxfp4_components(name)
                if components is None:
                    continue
                base, blocks_name, scales_name = components
                if base not in cache:
                    blocks = tensors.get(blocks_name)
                    scales = tensors.get(scales_name)
                    if blocks is None or scales is None:
                        raise ValueError(
                            "Unable to decode MXFP4 tensors for role "
                            f"'{name}'; expected both '{blocks_name}' and "
                            f"'{scales_name}'"
                        )
                    cache[base] = _decode_mxfp4_pair(blocks, scales)
                    tensors[base] = cache[base]
                setattr(layer, attr, base)

    @staticmethod
    def _tokens_allow_role(
        role: str,
        tail_tokens: Sequence[str],
        suffix_tokens: Sequence[str],
        shape: Tuple[int, ...],
        expected_proj: int,
    ) -> bool:
        if not ModelConfig._tokens_match(tail_tokens, suffix_tokens):
            return False

        if not ModelConfig._shape_allows_role(role, shape, expected_proj):
            return False

        return True

    @staticmethod
    def _tokens_match(
        actual_tokens: Sequence[str],
        expected_tokens: Sequence[str],
    ) -> bool:
        if len(actual_tokens) != len(expected_tokens):
            return False

        return all(
            ModelConfig._normalize_token(actual) == ModelConfig._normalize_token(expected)
            for actual, expected in zip(actual_tokens, expected_tokens)
        )

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.lower().replace("_", "").replace("-", "")

    @staticmethod
    def _shape_allows_role(
        role: str,
        shape: Tuple[int, ...],
        expected_proj: int,
    ) -> bool:
        if len(shape) < 2:
            return True
        rows, cols = shape[0], shape[1]
        if role in {"W_Q", "W_K", "W_V"}:
            return expected_proj in {rows, cols}
        if role == "W_O":
            return expected_proj in {rows, cols}
        return True


# ---------------------------------------------------------------------------
# Tensor utilities


_SAFE_TENSOR_NUMPY_DTYPES = {
    "F64": _np.float64,
    "F32": _np.float32,
    "F16": _np.float16,
    "I64": _np.int64,
    "U64": _np.uint64,
    "I32": _np.int32,
    "U32": _np.uint32,
    "I16": _np.int16,
    "U16": _np.uint16,
    "I8": _np.int8,
    "U8": _np.uint8,
    "BOOL": _np.bool_,
}


class _Bfloat16Tensor:
    """Lazy view over a BF16 safetensors payload."""

    __slots__ = ("_shape", "_buffer", "_uint16", "_float32_cache")
    __array_priority__ = 1000

    def __init__(self, data: memoryview, shape: Sequence[int]) -> None:
        self._shape = tuple(int(dim) for dim in shape)
        if data.format != "B":
            data = memoryview(data).cast("B")
        self._buffer = data
        self._uint16: _np.ndarray | None = _np.frombuffer(data, dtype=_np.uint16).reshape(
            self._shape
        )
        self._float32_cache: _np.ndarray | None = None

    def _ensure_float32(self) -> _np.ndarray:
        cached = self._float32_cache
        if cached is not None:
            return cached
        if self._uint16 is None:
            raise RuntimeError("BF16 tensor payload has already been released")
        raw = self._uint16.astype(_np.uint32)
        raw <<= 16
        float32 = raw.view(_np.float32).reshape(self._shape)
        self._float32_cache = float32
        self._uint16 = None
        try:
            self._buffer.release()  # type: ignore[attr-defined]
        except AttributeError:
            pass
        except BufferError:
            pass
        self._buffer = None  # type: ignore[assignment]
        return float32

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        count = 1
        for dim in self._shape:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        return self.size * 2

    def to_float32(self) -> _np.ndarray:
        return self._ensure_float32()

    def astype(self, dtype) -> _np.ndarray:
        return self._ensure_float32().astype(dtype)

    def tolist(self):
        return self._ensure_float32().tolist()

    def __array__(self, dtype=None):
        arr = self._ensure_float32()
        if dtype is not None and arr.dtype != dtype:
            return arr.astype(dtype)
        return arr

    def __iter__(self):
        return iter(self._ensure_float32())

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of unsized object")
        return int(self._shape[0])

    def __getitem__(self, item):
        return self._ensure_float32()[item]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"_Bfloat16Tensor(shape={self._shape})"


def _decode_safetensor_entry(dtype: str, shape: Sequence[int], data: memoryview) -> TensorLike:
    if dtype == "BF16":
        if not isinstance(data, memoryview):
            data = memoryview(data)
        return _Bfloat16Tensor(data, shape)
    np_dtype = _SAFE_TENSOR_NUMPY_DTYPES.get(dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported safetensors dtype: {dtype}")
    array = _np.frombuffer(data, dtype=np_dtype)
    return array.reshape(tuple(int(dim) for dim in shape))


def _read_safetensors_index(path: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    with path.open("rb") as handle:
        header_len_raw = handle.read(8)
        if len(header_len_raw) != 8:
            raise ValueError("Invalid safetensors header")
        header_len = int.from_bytes(header_len_raw, "little")
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("Incomplete safetensors header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Unable to parse safetensors header") from exc

    if not isinstance(header, dict):
        raise TypeError("Safetensors header must be a JSON object")

    tensor_index: Dict[str, Dict[str, Any]] = {}
    for name, entry in header.items():
        if not isinstance(name, str):
            continue
        if name.startswith("__"):
            continue
        if not isinstance(entry, dict):
            continue
        if not {"dtype", "shape", "data_offsets"}.issubset(entry):
            continue
        tensor_index[name] = {
            "dtype": entry["dtype"],
            "shape": tuple(int(dim) for dim in entry["shape"]),
            "data_offsets": tuple(int(offset) for offset in entry["data_offsets"]),
        }

    data_start = 8 + header_len
    return tensor_index, data_start


def _read_tensor_payload(
    handle, base_offset: int, entry: Mapping[str, Any]
) -> memoryview:
    start, end = entry["data_offsets"]
    length = end - start
    handle.seek(base_offset + start)
    buffer = bytearray(length)
    view = memoryview(buffer)
    read = handle.readinto(view)
    if read != length:
        raise IOError("Unexpected end of safetensors payload")
    return view


def _load_tensors_from_deserialize(
    path: Path, keys: Optional[Sequence[str]] = None
) -> Dict[str, TensorLike]:
    index, base_offset = _read_safetensors_index(path)
    wanted = set(keys) if keys is not None else None
    tensors: Dict[str, TensorLike] = {}

    with path.open("rb") as handle:
        for name, entry in index.items():
            if wanted is not None and name not in wanted:
                continue
            payload = _read_tensor_payload(handle, base_offset, entry)
            tensors[name] = _decode_safetensor_entry(
                entry["dtype"], entry["shape"], payload
            )

    if wanted is not None and len(tensors) != len(wanted):
        missing = sorted(set(wanted) - set(tensors))
        raise KeyError(f"Missing tensors in safetensors file: {', '.join(missing)}")
    return tensors


def load_tensors(path: Path) -> Dict[str, TensorLike]:
    index: Dict[str, Dict[str, Any]] | None = None
    base_offset = 0
    tensors: Dict[str, TensorLike] = {}
    fallback_entries: List[Tuple[str, Dict[str, Any]]] = []
    opener = _resolve_safe_open()

    try:
        with opener(path, framework="numpy") as tensor_file:
            index, base_offset = _read_safetensors_index(path)
            for key in tensor_file.keys():
                entry = index.get(key)
                if entry is None:
                    continue
                dtype = entry["dtype"]
                if dtype == "BF16":
                    fallback_entries.append((key, entry))
                    continue
                try:
                    array = tensor_file.get_tensor(key)
                    if _NP_ARRAY_TYPES and isinstance(array, _NP_ARRAY_TYPES):
                        array = array.copy()
                    elif _is_torch_tensor(array):
                        array = array.clone()
                    tensors[key] = array
                except TypeError:
                    fallback_entries.append((key, entry))
    except SafetensorError as exc:
        if "bf16" not in str(exc).lower():
            raise
        return _load_tensors_from_deserialize(path)

    if fallback_entries:
        if index is None:
            index, base_offset = _read_safetensors_index(path)
        with path.open("rb") as handle:
            for key, entry in fallback_entries:
                payload = _read_tensor_payload(handle, base_offset, entry)
                tensors[key] = _decode_safetensor_entry(
                    entry["dtype"], entry["shape"], payload
                )
    return tensors


# ---------------------------------------------------------------------------
# PRF generation


def _column_means_squared(matrix: List[List[float]]) -> List[float]:
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    result: List[float] = []
    for j in range(cols):
        acc = 0.0
        for i in range(rows):
            value = matrix[i][j]
            acc += value * value
        result.append(max(acc / rows, 1e-8))
    return result


def compute_prf(cfg: ModelConfig, tensors: Mapping[str, TensorLike], r: int) -> Dict[str, List]:
    digest = hashlib.sha256()
    diag_accum = [0.0 for _ in range(cfg.d_model)]
    counts = [0 for _ in range(cfg.d_model)]
    for layer in cfg.layers:
        matrix = tensors.get(layer.w_k)
        if matrix is None:
            continue
        digest.update(layer.w_k.encode("utf-8"))
        tensor_bytes = _tensor_bytes(matrix)
        digest.update(tensor_bytes)
        for row in matrix:
            for idx, value in enumerate(row[: cfg.d_model]):
                diag_accum[idx] += float(value) * float(value)
                counts[idx] += 1

    diag: List[float] = []
    for idx in range(cfg.d_model):
        if counts[idx] == 0:
            diag.append(1e-8)
        else:
            diag.append(max(diag_accum[idx] / counts[idx], 1e-8))

    prng_seed = int.from_bytes(digest.digest()[:8], "little", signed=False)
    prng = SplitMix64(seed=prng_seed)
    prf_W: List[List[float]] = []
    whitening_sig2: List[float] = []
    total_diag = sum(diag) / max(1, len(diag))
    for _ in range(r):
        gaussian = gaussian_vector(prng, cfg.d_model)
        scaled = [g * math.sqrt(diag[j]) for j, g in enumerate(gaussian)]
        variance = sum(value * value for value in scaled) / max(1, len(scaled))
        whitening_sig2.append(max(variance, 1e-6))
        prf_W.append(scaled)

    r_init = []
    for i in range(r):
        scale = 1.0 / float(i + 1)
        r_init.append([diag[j] * scale for j in range(cfg.d_model)])

    s_init = [total_diag for _ in range(r)]
    whitening_mu = [0.0 for _ in range(r)]
    return {
        "prf_W": prf_W,
        "R_init": r_init,
        "s_init": s_init,
        "whitening_mu": whitening_mu,
        "whitening_sig2": whitening_sig2,
    }


# ---------------------------------------------------------------------------
# Overlay generation


def hadamard_generator(prng: SplitMix64, rows: int, cols: int) -> Iterable[List[float]]:
    produced = 0
    while True:
        column = [1.0 if (prng.next() & 1) == 0 else -1.0 for _ in range(rows)]
        produced += 1
        yield column
        if cols > 0 and produced >= cols:
            cols = 0  # allow additional columns if the caller needs more


def gaussian_matrix(prng: SplitMix64, rows: int, cols: int) -> List[List[float]]:
    matrix: List[List[float]] = []
    for _ in range(rows):
        matrix.append(gaussian_vector(prng, cols))
    return matrix


def compute_overlays(
    cfg: ModelConfig, tensors: Mapping[str, TensorLike], r: int, r_v: int
) -> Dict[str, List]:
    accumulator: Optional[List[List[float]]] = None
    layer_count = 0
    digest = hashlib.sha256()
    for layer in cfg.layers:
        w_o = tensors.get(layer.w_o)
        if w_o is None:
            continue
        digest.update(layer.w_o.encode("utf-8"))
        digest.update(_tensor_bytes(w_o))
        layer_rows = _tensor_len(w_o)
        layer_cols = _tensor_len(w_o[0]) if layer_rows else 0
        if accumulator is None:
            accumulator = [[0.0 for _ in range(layer_cols)] for _ in range(layer_rows)]
        for i in range(min(layer_rows, len(accumulator))):
            row = accumulator[i]
            layer_row = w_o[i]
            for j in range(min(layer_cols, len(row))):
                row[j] += float(layer_row[j])
        layer_count += 1

    if accumulator is None or layer_count == 0:
        raise ValueError("No W_O tensors available to compute overlays")

    for i in range(len(accumulator)):
        row = accumulator[i]
        accumulator[i] = [value / layer_count for value in row]

    prng_seed = int.from_bytes(digest.digest()[:8], "little", signed=False)
    prng = SplitMix64(seed=prng_seed)
    rows = len(accumulator)
    cols = len(accumulator[0]) if accumulator else 0
    feature_dim = min(r, rows)
    value_dim = min(r_v, cols) if cols else r_v
    if feature_dim == 0 or value_dim == 0:
        raise ValueError("Overlay dimensions must be positive")

    H = orthonormal_matrix(feature_dim, r_v, hadamard_generator(prng, feature_dim, r_v))
    Omega = gaussian_matrix(prng, rows, r_v)
    Y = matmul(transpose(accumulator), Omega)
    U = orthonormal_matrix(cols, r_v, transpose(Y))
    Z = matmul(transpose(U), matmul(accumulator, H))
    return {
        "overlays_H": H,
        "overlays_U": U,
        "overlays_DeltaW": Z,
    }


# ---------------------------------------------------------------------------
# FFN collapse


def matrix_abs(matrix) -> List[List[float]]:
    result: List[List[float]] = []
    for row in matrix:
        result.append([abs(float(value)) for value in row])
    return result


def sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def top_indices(values: List[float], count: int) -> List[int]:
    if len(values) <= count:
        return list(range(len(values)))
    return [idx for idx, _ in sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:count]]


_CUCKOO_ENTRY = struct.Struct("<QIf")


def _serialize_cuckoo(entries: Sequence[Tuple[int, int, float]]) -> List[int]:
    if not entries:
        return []
    buf = io.BytesIO()
    for key, hidden_idx, value in entries:
        buf.write(_CUCKOO_ENTRY.pack(key, hidden_idx, float(value)))
    return list(buf.getvalue())


def collapse_ffn(
    cfg: ModelConfig, tensors: Mapping[str, TensorLike], top_l: int
) -> Tuple[Dict[str, List], Dict[str, object]]:
    weight_map: Dict[int, float] = {}
    residual_entries: List[Tuple[int, int, float]] = []
    base_bias = 0.0

    for layer_index, layer in enumerate(cfg.layers):
        W1 = tensors[layer.w1]
        W2 = tensors[layer.w2]
        b1 = tensors[layer.b1]
        b2 = tensors[layer.b2]
        hidden_dim = len(W1)

        base_bias += sum(float(x) for x in b2)
        base_bias += sum(max(0.0, float(x)) for x in b1)

        abs_W1 = matrix_abs(W1)
        abs_W2 = matrix_abs(W2)

        for feature in range(cfg.d_model):
            scores: List[float] = []
            for h in range(hidden_dim):
                col_sum = sum(abs_W2[out_idx][h] for out_idx in range(len(abs_W2)))
                scores.append(col_sum * abs_W1[h][feature])
            idxs = set(top_indices(scores, top_l))
            effect = 0.0
            for h in idxs:
                sign_sum = sum(
                    float(W2[out_idx][h]) for out_idx in range(len(W2))
                )
                effect += sign(sign_sum) * max(0.0, float(W1[h][feature]))
            key = (layer_index << 32) | feature
            weight_map[key] = effect

            for h in range(hidden_dim):
                if h in idxs:
                    continue
                sign_sum = sum(
                    float(W2[out_idx][h]) for out_idx in range(len(W2))
                )
                residual = sign(sign_sum) * max(0.0, float(W1[h][feature]))
                if abs(residual) > 0.0:
                    residual_entries.append((key, h, residual))

    slot_keys = list(weight_map.keys())
    seeds, ordered_keys, slot_lookup = _build_two_level_mph(slot_keys)
    mphf = [[(seed >> (8 * i)) & 0xFF for i in range(4)] for seed in seeds]
    key_bytes = [[(key >> (8 * i)) & 0xFF for i in range(8)] for key in ordered_keys]
    weights = [weight_map[key] for key in ordered_keys]

    cuckoo_blob = _serialize_cuckoo(residual_entries)

    arrays = {
        "linear_mphf": mphf,
        "linear_keys": key_bytes,
        "linear_weights": weights,
        "linear_bias": [base_bias],
        "cuckoo_delta": cuckoo_blob,
    }
    metadata = {
        "keys": ordered_keys,
        "weights": weights,
        "bias": base_bias,
        "residuals": len(residual_entries),
        "slot_lookup": slot_lookup,
    }
    return arrays, metadata


# ---------------------------------------------------------------------------
# Memory and bridge records


def memory_coefficients(cfg: ModelConfig) -> Tuple[Dict[str, List], Dict[str, object]]:
    theta = cfg.rope_theta if cfg.rope_theta is not None else 10000.0
    rho = 0.995
    layer_count = max(1, len(cfg.layers))
    coeffs: List[List[float]] = []
    delay = []
    for layer_index in range(layer_count):
        angle = (layer_index + 1) / layer_count * (1.0 / theta)
        coeffs.append([2 * rho * math.cos(angle), -(rho ** 2), 1.0])
        delay.append(0.0)
    arrays = {"memory_coeff": coeffs, "delaybuf_init": delay}
    metadata = {"rho": rho, "theta": theta, "layers": layer_count}
    return arrays, metadata


def _layer_seed(layer: LayerConfig, tensors: Mapping[str, TensorLike]) -> int:
    digest = hashlib.sha256()
    for name in (
        layer.w_q,
        layer.w_k,
        layer.w_v,
        layer.w_o,
        layer.w1,
        layer.w2,
        layer.b1,
        layer.b2,
    ):
        tensor = tensors.get(name)
        if tensor is None:
            continue
        digest.update(_tensor_bytes(tensor))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _quantize_q8_8(value: float) -> Tuple[List[int], float]:
    magnitude = abs(value)
    if magnitude == 0:
        scale = 1.0
    else:
        scale = max(1e-6, magnitude / 0.832767)
    quantised = int(round(value / scale * 256.0))
    quantised = max(-32768, min(32767, quantised))
    scale_q = int(round(scale * 256.0))
    scale_q = max(1, min(32767, scale_q))
    return [scale_q, quantised], scale


def bridge_records(
    cfg: ModelConfig, tensors: Mapping[str, TensorLike], vocab_size: int, W: int = 2
) -> Tuple[Dict[str, List], Dict[str, object]]:
    vocab = vocab_size or cfg.d_model or 16
    vocab = max(1, vocab)
    hubs: List[List[int]] = []
    qdin: List[List[int]] = []
    qdout: List[List[int]] = []
    peers: List[int] = []
    in_scales: List[float] = []
    out_scales: List[float] = []

    aggregated_in = [0.0 for _ in range(vocab)]
    aggregated_out = [0.0 for _ in range(vocab)]
    seeds: List[int] = []
    digest = hashlib.sha256()
    for layer in cfg.layers:
        layer_seed = _layer_seed(layer, tensors)
        seeds.append(layer_seed)
        digest.update(struct.pack("<Q", layer_seed))
        W1 = tensors[layer.w1]
        W2 = tensors[layer.w2]
        b1 = tensors[layer.b1]
        b2 = tensors[layer.b2]
        hidden_dim = len(W1)
        out_dim = len(W2)
        for token in range(vocab):
            feature = token % cfg.d_model
            feature_h = feature % max(1, hidden_dim)
            feature_out = feature % max(1, out_dim)
            avg_w1 = sum(float(W1[h][feature % len(W1[h])]) for h in range(hidden_dim)) / max(1, hidden_dim)
            avg_w2 = sum(float(value) for value in W2[feature_out]) / max(1, len(W2[feature_out]))
            b1_len = _tensor_len(b1)
            b2_len = _tensor_len(b2)
            bias1 = float(b1[feature_h % b1_len]) if b1_len else 0.0
            bias2 = float(b2[feature_out % b2_len]) if b2_len else 0.0
            aggregated_in[token] += avg_w1 + bias1
            aggregated_out[token] += avg_w2 + bias2

    global_seed = int.from_bytes(digest.digest()[:8], "little", signed=False)

    for token in range(vocab):
        token_digest = hashlib.sha256()
        token_digest.update(struct.pack("<I", token))
        token_digest.update(struct.pack("<f", float(aggregated_in[token])))
        token_digest.update(struct.pack("<f", float(aggregated_out[token])))
        token_seed = int.from_bytes(token_digest.digest()[:8], "little", signed=False)
        prng = SplitMix64(global_seed ^ token_seed)
        row: List[int] = []
        for _ in range(W):
            bits = prng.next()
            row.extend([(bits >> (8 * i)) & 0xFF for i in range(8)])
        hubs.append(row)

        qdin_entry, in_scale = _quantize_q8_8(aggregated_in[token])
        qdout_entry, out_scale = _quantize_q8_8(aggregated_out[token])
        qdin.append(qdin_entry)
        qdout.append(qdout_entry)
        in_scales.append(in_scale)
        out_scales.append(out_scale)

        jitter = ((prng.next() >> 8) & 0xFF) / 512.0 - 0.25
        peer_score = int(round((0.5 + jitter) * 256.0))
        peer_score = max(-32768, min(32767, peer_score))
        peers.append(peer_score)

    arrays = {
        "bridge_hubs": hubs,
        "bridge_qDin": qdin,
        "bridge_qDout": qdout,
        "peer_scores": peers,
    }
    metadata = {
        "in_scales": in_scales,
        "out_scales": out_scales,
        "seed": global_seed,
        "legs": W,
        "layer_seeds": seeds,
    }
    return arrays, metadata


# ---------------------------------------------------------------------------
# Tokenizer placeholders


def tokenizer_arrays(
    cfg: ModelConfig, tensors: Mapping[str, TensorLike], max_len: int = 4
) -> Tuple[Dict[str, List], Dict[str, object]]:
    vocab = cfg.vocab_size or 256
    vocab = max(1, vocab)

    tensor_digest = hashlib.sha256()
    embedding: Optional[TensorLike] = None
    for name in sorted(tensors.keys()):
        tensor = tensors[name]
        try:
            tensor_bytes = _tensor_bytes(tensor)
        except Exception:  # pragma: no cover - defensive
            continue
        tensor_digest.update(name.encode("utf-8"))
        tensor_digest.update(tensor_bytes)
        if embedding is None:
            rows = _tensor_len(tensor)
            if rows >= vocab:
                try:
                    first_row = tensor[0]  # type: ignore[index]
                except Exception:
                    first_row = None
                if first_row is not None and _tensor_len(first_row) > 0:
                    embedding = tensor

    global_seed = int.from_bytes(tensor_digest.digest()[:8], "little", signed=False)

    used_pieces: List[bytes] = []
    pieces: List[Tuple[bytes, int]] = []

    def has_prefix_conflict(candidate: bytes) -> bool:
        for existing in used_pieces:
            if existing.startswith(candidate) or candidate.startswith(existing):
                return True
        return False

    for token in range(vocab):
        attempt = 0
        base_seed = _mix64(global_seed, token + 1)
        row_seed = 0
        if embedding is not None:
            row = embedding[token][: cfg.d_model]
            row_seed = _hash_bytes(_tensor_bytes([row]))
        while True:
            local_seed = _mix64(base_seed ^ row_seed, attempt + 1)
            local = SplitMix64(local_seed)
            length = 1 + (local.next() % max_len)
            piece = bytes((local.next() & 0xFF) for _ in range(length))
            if piece not in used_pieces and not has_prefix_conflict(piece):
                used_pieces.append(piece)
                pieces.append((piece, token))
                break
            attempt += 1

    class _TrieNode(dict):
        pass

    root: _TrieNode = _TrieNode()
    node_ids: Dict[int, int] = {id(root): 0}
    next_state = 1
    transitions: List[Tuple[int, int, int, int]] = []

    for piece, token in pieces:
        node = root
        current = 0
        for idx, byte in enumerate(piece):
            child = node.get(byte)
            if child is None:
                child = _TrieNode()
                node[byte] = child
            if id(child) not in node_ids and idx < len(piece) - 1:
                node_ids[id(child)] = next_state
                next_state += 1
            if idx == len(piece) - 1:
                dest = 0
                output = token
            else:
                dest = node_ids[id(child)]
                output = 0
            transitions.append((current, dest, byte, output))
            node = child
            current = dest

    lines = ["% byte level fst derived from checkpoint", "start 0", "end 0"]
    for src, dst, byte, output in transitions:
        lines.append(f"{src} {dst} {byte} {output}")
    fst_text = "\n".join(lines).encode("utf-8")

    tables: Dict[str, List[int]] = {}
    mph_meta: Dict[int, Dict[str, object]] = {}
    modulus = 1 << 64
    for length in range(1, max_len + 1):
        factor = pow(257, length - 1, modulus)
        table_bytes: List[int] = []
        for byte in range(256):
            value = (factor * byte + global_seed) & (modulus - 1)
            table_bytes.extend(struct.pack("<Q", value))
        tables[f"T_{length}"] = table_bytes

        length_pieces = [piece for piece, _ in pieces if len(piece) == length]
        key_hashes = [_hash_bytes(piece) for piece in length_pieces]
        seeds, ordered_keys, _ = _build_two_level_mph(key_hashes)
        mph_meta[length] = {
            "table_size": len(seeds),
            "seeds": seeds,
            "key_hashes": ordered_keys,
        }

    arrays = {"tokenizer_fst": list(fst_text)}
    arrays.update(tables)
    metadata = {
        "pieces": pieces,
        "max_piece_length": max_len,
        "seed": global_seed,
        "mph": mph_meta,
    }
    return arrays, metadata


# ---------------------------------------------------------------------------
# Manifest writing


def write_manifest(
    path: Path,
    cfg: ModelConfig,
    artefacts: Mapping[str, ArrayDigest],
    *,
    r: int,
    r_v: int,
    vocab_size: int,
) -> None:
    schema_path = Path("docs/specs/Sera-Transfer.txt")
    schema_digest = (
        hashlib.sha256(schema_path.read_bytes()).digest()
        if schema_path.exists()
        else hashlib.sha256(b"sera").digest()
    )
    seed_digest = hashlib.sha256(b"sera-transfer").digest()

    empty_digest = hashlib.sha256(b"").digest()

    def _digest(name: str) -> bytes:
        info = artefacts.get(name)
        return info.sha256 if info is not None else empty_digest

    def _length(name: str) -> int:
        info = artefacts.get(name)
        return info.bytes if info is not None else 0

    with path.open("wb") as f:
        f.write(struct.pack("<I", 0x5345524D))
        f.write(struct.pack("<I", 0x3))
        f.write(seed_digest)
        f.write(schema_digest)

        L_tok = 4
        S_norm = 1.0
        L_norm = 1.0
        P_gen = vocab_size
        f.write(struct.pack("<IffI", L_tok, S_norm, L_norm, P_gen))
        f.write(_digest("tokenizer_fst"))
        _write_utf8(f, "byte_level")

        f.write(struct.pack("<Ifdd", r, cfg.tau, 1e-8, 3.0))
        f.write(struct.pack("<d", 1e-3))
        f.write(hashlib.sha256(b"lambda").digest())

        C = _length("linear_weights") // 4
        f.write(struct.pack("<5Idd", C, cfg.d_model, 2, 1, 8, 0.1, 1.0))

        f.write(struct.pack("<6Id", 1, 2, 1, 1, 32, 4, 0.95))
        f.write(struct.pack("<III", cfg.n_heads, cfg.head_dim, r_v))
        _write_utf8(f, "l2")
        f.write(struct.pack("<fi", 0.5, 4))

        f.write(struct.pack("<III", vocab_size, 2, cfg.d_model))
        f.write(_digest("bridge_hubs"))
        f.write(struct.pack("<ff", 0.1, 1.0))
        f.write(struct.pack("<II", 0, 0))
        f.write(struct.pack("<III", 4, 2, 8))

        f.write(struct.pack("<IIIIfff", 8, 16, 4, 2, 1.3, 0.01, 0.05))
        f.write(struct.pack("<f f f f", 0.5, 20.0, 0.1, 0.05))
        f.write(_digest("linear_mphf"))
        f.write(_digest("linear_keys"))
        f.write(hashlib.sha256(b"prev").digest())
        f.write(hashlib.sha256(b"curr").digest())


def _write_utf8(f: io.BufferedWriter, value: str) -> None:
    data = value.encode("utf-8")
    f.write(struct.pack("<H", len(data)))
    f.write(data)


def _determinism_metadata(
    settings: DeterminismSettings, observer: DeterminismObserver | None
) -> Dict[str, object]:
    observed = sorted(observer.mxfp4_backends) if observer is not None else []
    if not observed:
        observed = ["python"]
    return {
        "householder_qr": {
            "algorithm": "householder",
            "tolerance": _HOUSEHOLDER_TOLERANCE,
            "sign_rule": "negative_if_nonnegative_pivot",
        },
        "mxfp4_decoder": {
            "configured_backend": settings.mxfp4_backend,
            "observed_backends": observed,
            "torch_threads": settings.torch_threads,
            "numpy_threads": settings.numpy_threads,
            "assume_ftz": settings.assume_ftz,
            "assume_fma": settings.assume_fma,
        },
    }


# ---------------------------------------------------------------------------
# Conversion driver


def convert(
    source: Path,
    output: Path,
    *,
    r: int = 64,
    r_v: int = 8,
    top_l: int = 8,
    original_subdir: str | Path | None = None,
    verbose: bool = False,
    determinism: DeterminismSettings | None = None,
) -> ConversionSummary:
    """Convert a checkpoint directory into a Sera Transfer Kit artefact.

    The conversion expects a ``config.json`` and ``model.safetensors`` file.
    When these files are missing from ``source`` directly, the function probes
    a small set of common Hugging Face layouts – ``source/original`` and
    ``source/original/model`` – before failing. Callers that rely on the
    quickstart helper must therefore retain the root-level metadata alongside
    the ``original/`` directory. Advanced users can supply an explicit
    ``original_subdir`` to search an arbitrary additional location.
    """

    source = _normalise_path(source)
    output = _normalise_path(output)

    try:
        created_output_dir = not output.exists()
    except OSError:
        created_output_dir = True
    output.mkdir(parents=True, exist_ok=True)

    previous_settings = dataclasses.replace(_active_determinism())
    active_settings = (
        dataclasses.replace(determinism)
        if determinism is not None
        else previous_settings
    )
    observer = DeterminismObserver()
    previous_observer = _DETERMINISM_OBSERVER
    _set_determinism(active_settings)
    _set_determinism_observer(observer)

    if verbose and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    log_path = output / "conversion.log"
    file_handler: logging.Handler | None = None
    previous_level = logger.level

    cleanup_on_failure = False

    try:
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        def log_verbose(message: str, *args: object) -> None:
            if verbose:
                logger.info(message, *args)
            else:
                logger.debug(message, *args)

        def log_notice(message: str, *args: object) -> None:
            logger.info(message, *args)

        log_notice("Writing conversion log to %s", log_path)
        log_notice("Resolved source path: %s", source)
        log_notice("Resolved output path: %s", output)

        search_roots: List[Path] = []

        def add_root(candidate: Path | str) -> None:
            path = candidate if isinstance(candidate, Path) else Path(candidate)
            if not path.is_absolute():
                path = source / path
            if path not in search_roots:
                search_roots.append(path)
                log_verbose("Registered search root: %s", path)

        add_root(source)
        if original_subdir is not None:
            add_root(original_subdir)
        else:
            expected_files = ("config.json", "model.safetensors")
            need_probe = any(
                not _safe_exists(source / filename)
                for filename in expected_files
            )
            if need_probe:
                add_root("original")
                add_root(Path("original") / "model")

        def find_file(filename: str) -> Path:
            for root in search_roots:
                candidate = root / filename
                log_verbose("Probing %s", candidate)
                if _safe_exists(candidate):
                    log_notice("Located %s at %s", filename, candidate)
                    return candidate
            search_list = ", ".join(str(root) for root in search_roots)
            message = f"Unable to locate {filename!r}; searched: {search_list or source}"
            logger.error(message)
            raise FileNotFoundError(message)

        def _convert_inner() -> ConversionSummary:
            determinism_meta: Dict[str, object] = {}
            config_path = find_file("config.json")
            model_path = find_file("model.safetensors")
            log_notice("Reading model configuration from %s", config_path)
            config_data = json.loads(config_path.read_text())
            if isinstance(config_data, Mapping):
                config_keys = _format_model_config_keys(config_data)
                log_verbose("Model config keys: %s", config_keys)
            load_fn = _resolve_load_tensors()
            tensors = load_fn(model_path)
            tokenizer_fn = _resolve_callable("tokenizer_arrays", tokenizer_arrays)
            prf_fn = _resolve_callable("compute_prf", compute_prf)
            overlays_fn = _resolve_callable("compute_overlays", compute_overlays)
            collapse_fn = _resolve_callable("collapse_ffn", collapse_ffn)
            memory_fn = _resolve_callable("memory_coefficients", memory_coefficients)
            bridge_fn = _resolve_callable("bridge_records", bridge_records)
            write_array_fn = _resolve_callable("write_array", write_array)
            write_manifest_fn = _resolve_callable("write_manifest", write_manifest)
            runtime_loader = _resolve_callable("_load_sera_runtime", _load_sera_runtime)
            cfg = ModelConfig.from_dict(config_data, tensors=tensors)

            local_r = min(r, cfg.d_model)
            local_r_v = min(r_v, local_r)

            arrays_dir = output / "arrays"
            arrays_dir.mkdir(parents=True, exist_ok=True)

            artefact_digests: Dict[str, ArrayDigest] = {}
            artefact_records: Dict[str, Dict[str, object]] = {}
            metadata: Dict[str, object] = {}
            array_infos: List[ArrayInfo] = []
            snapshot_infos: List[SnapshotInfo] = []

            def store_array(name: str, data, dtype: str) -> None:
                path = arrays_dir / f"{name}.bin"
                payload = write_array_fn(path, data, dtype)
                shape = _infer_shape(data)
                payload_len = len(payload)
                digest = hashlib.sha256(payload).digest()
                artefact_digests[name] = ArrayDigest(bytes=payload_len, sha256=digest)
                artefact_records[name] = {
                    "path": str(path),
                    "dtype": dtype,
                    "shape": shape,
                    "bytes": payload_len,
                    "sha256": digest.hex(),
                }
                array_infos.append(
                    ArrayInfo(
                        name=name,
                        path=path,
                        dtype=dtype,
                        shape=shape,
                        bytes=payload_len,
                        sha256=digest.hex(),
                    )
                )
                del payload

            log_notice("Generating tokenizer arrays")
            tokenizer_data, tokenizer_meta = tokenizer_fn(cfg, tensors)
            for name, data in tokenizer_data.items():
                store_array(name, data, "u8")
            metadata["tokenizer"] = tokenizer_meta

            log_notice("Generating PRF arrays")
            prf = prf_fn(cfg, tensors, local_r)
            for name, data in prf.items():
                store_array(name, data, "f32")
            attention_meta = {
                "features": local_r,
                "tau": getattr(cfg, "tau", 1.0),
                "whitening_sig2": prf.get("whitening_sig2", []),
            }
            optional_attention = {
                "rope_theta": cfg.rope_theta,
                "num_key_value_heads": cfg.num_key_value_heads,
                "sliding_window": cfg.sliding_window,
                "initial_context_length": cfg.initial_context_length,
                "rope_scaling_factor": cfg.rope_scaling_factor,
                "rope_ntk_alpha": cfg.rope_ntk_alpha,
                "rope_ntk_beta": cfg.rope_ntk_beta,
            }
            for key, value in optional_attention.items():
                if value is not None:
                    attention_meta[key] = value
            metadata["attention"] = attention_meta

            log_notice("Computing attention overlays")
            overlays = overlays_fn(cfg, tensors, local_r, local_r_v)
            for name, data in overlays.items():
                store_array(name, data, "f32")
            metadata["overlays"] = {
                "rank": local_r,
                "rank_value": local_r_v,
                "rows": len(overlays.get("overlays_H", [])),
                "cols": len(overlays.get("overlays_U", [])),
            }

            log_notice("Collapsing FFN weights")
            linear_data, linear_meta = collapse_fn(cfg, tensors, top_l)
            store_array("linear_mphf", linear_data["linear_mphf"], "u8")
            store_array("linear_keys", linear_data["linear_keys"], "u8")
            store_array("linear_weights", linear_data["linear_weights"], "f32")
            store_array("linear_bias", linear_data["linear_bias"], "f32")
            store_array("cuckoo_delta", linear_data["cuckoo_delta"], "u8")
            linear_meta = dict(linear_meta)
            optional_linear = {
                "num_experts": cfg.num_experts,
                "experts_per_token": cfg.experts_per_token,
                "intermediate_size": cfg.intermediate_size,
                "swiglu_limit": cfg.swiglu_limit,
            }
            for key, value in optional_linear.items():
                if value is not None:
                    linear_meta[key] = value
            metadata["linear"] = linear_meta

            log_notice("Computing memory coefficients")
            memory_data, memory_meta = memory_fn(cfg)
            store_array("memory_coeff", memory_data["memory_coeff"], "f64")
            store_array("delaybuf_init", memory_data["delaybuf_init"], "f32")
            metadata["memory"] = memory_meta

            log_notice("Constructing bridge records")
            bridge_data, bridge_meta = bridge_fn(cfg, tensors, cfg.vocab_size)
            store_array("bridge_hubs", bridge_data["bridge_hubs"], "u8")
            store_array("bridge_qDin", bridge_data["bridge_qDin"], "i16")
            store_array("bridge_qDout", bridge_data["bridge_qDout"], "i16")
            store_array("peer_scores", bridge_data["peer_scores"], "i16")
            metadata["bridge"] = bridge_meta
            determinism_meta = _determinism_metadata(active_settings, observer)
            metadata["determinism"] = determinism_meta

            manifest_path = output / "sera_manifest.bin"
            write_manifest_fn(
                manifest_path,
                cfg,
                artefact_digests,
                r=local_r,
                r_v=local_r_v,
                vocab_size=cfg.vocab_size or 16,
            )
            determinism_meta["manifest_path"] = manifest_path.name
            try:
                determinism_meta["manifest_digest"] = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
            except OSError as exc:  # pragma: no cover - IO errors uncommon
                determinism_meta["manifest_digest_error"] = str(exc)

            runtime_snapshot_error: Dict[str, str] | None = None
            try:
                sera_cls, sera_config_cls = runtime_loader()
            except Exception as exc:  # pragma: no cover - defensive continuation
                logger.warning(
                    "Failed to load Sera runtime; continuing without runtime state (%s: %s)",
                    exc.__class__.__name__,
                    exc,
                    exc_info=True,
                )
                runtime_snapshot_error = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
                runtime_snapshot = {
                    "status": "error",
                    "error": runtime_snapshot_error,
                }
            else:
                try:
                    base_config = sera_config_cls()
                except Exception:  # pragma: no cover - extremely defensive
                    base_config = None

                try:
                    if base_config is not None and dataclasses.is_dataclass(base_config):
                        pieces = tokenizer_meta.get("pieces", [])
                        tokenizer_vocab = {piece: token for piece, token in pieces}
                        max_piece_length = tokenizer_meta.get(
                            "max_piece_length", base_config.tokenizer.max_piece_length
                        )
                        tokenizer_config = dataclasses.replace(
                            base_config.tokenizer,
                            vocabulary=tokenizer_vocab,
                            max_piece_length=max_piece_length,
                        )
                        whitening_sig2 = metadata["attention"].get("whitening_sig2") or [1.0]
                        beta_floor = max(1e-3, min(float(value) for value in whitening_sig2))
                        value_dim = len(overlays.get("overlays_U", [])) or cfg.d_model
                        attention_config = dataclasses.replace(
                            base_config.attention,
                            dim=cfg.d_model,
                            value_dim=value_dim,
                            features=local_r,
                            tau=getattr(cfg, "tau", base_config.attention.tau),
                            beta_floor=beta_floor,
                        )
                        linear_capacity = max(
                            len(linear_meta.get("keys", [])), base_config.linear.capacity
                        )
                        linear_config = dataclasses.replace(
                            base_config.linear, capacity=linear_capacity
                        )
                        bridge_config = dataclasses.replace(
                            base_config.bridge,
                            hub_window=max(
                                base_config.bridge.hub_window, bridge_meta.get("legs", 0) * 4
                            ),
                        )
                        sera_config = dataclasses.replace(
                            base_config,
                            tokenizer=tokenizer_config,
                            attention=attention_config,
                            linear=linear_config,
                            bridge=bridge_config,
                        )
                        sera_model = sera_cls(sera_config)
                        runtime_snapshot = sera_model.snapshot()
                    else:
                        config_instance = base_config if base_config is not None else object()
                        sera_model = sera_cls(config_instance)
                        runtime_snapshot = sera_model.snapshot()
                except Exception as exc:  # pragma: no cover - defensive continuation
                    logger.warning(
                        "Failed to materialise Sera runtime snapshot; continuing without runtime state (%s: %s)",
                        exc.__class__.__name__,
                        exc,
                        exc_info=True,
                    )
                    runtime_snapshot_error = {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                    runtime_snapshot = {
                        "status": "error",
                        "error": runtime_snapshot_error,
                    }
            if isinstance(runtime_snapshot, dict):
                runtime_meta = runtime_snapshot.get("metadata")
                if not isinstance(runtime_meta, dict):
                    runtime_meta = {}
                    runtime_snapshot["metadata"] = runtime_meta
                runtime_meta["determinism"] = determinism_meta
            if runtime_snapshot_error is not None:
                metadata["runtime_snapshot_error"] = runtime_snapshot_error

            snapshot_path = output / "sera_state.pkl"
            snapshot = {
                "model_config": _config_to_dict(cfg),
                "artefacts": artefact_records,
                "metadata": metadata,
                "manifest_path": str(output / "sera_manifest.bin"),
                "sera_snapshot": runtime_snapshot,
            }
            with snapshot_path.open("wb") as fh:
                pickle.dump(snapshot, fh)
            snapshot_infos.append(SnapshotInfo("pickle", snapshot_path))

            json_snapshot = _json_value(snapshot)
            json_path = output / "sera_state.json"
            with json_path.open("w", encoding="utf-8") as fh:
                json.dump(json_snapshot, fh, indent=2, sort_keys=True)
                fh.write("\n")
            snapshot_infos.append(SnapshotInfo("json", json_path))

            try:  # pragma: no cover - msgpack is optional
                import msgpack  # type: ignore
            except ModuleNotFoundError:  # pragma: no cover - exercised when msgpack missing
                pass
            else:
                msgpack_path = output / "sera_state.msgpack"
                with msgpack_path.open("wb") as fh:
                    msgpack.pack(json_snapshot, fh, use_bin_type=True)
                snapshot_infos.append(SnapshotInfo("msgpack", msgpack_path))

            total_bytes = sum(info.bytes for info in array_infos)
            summary = ConversionSummary(
                source=source,
                output=output,
                arrays=tuple(array_infos),
                manifest_path=manifest_path,
                metadata_keys=tuple(sorted(metadata)),
                snapshots=tuple(snapshot_infos),
                total_bytes=total_bytes,
                log_path=log_path,
            )
            formatted = format_summary(summary)
            for line in formatted.splitlines():
                log_verbose(line)
            return summary

        try:
            return _convert_inner()
        except ModuleNotFoundError:
            cleanup_on_failure = created_output_dir
            raise
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            logger.exception("Conversion failed: %s", exc)
            raise
    finally:
        _set_determinism_observer(previous_observer)
        _set_determinism(previous_settings)
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()
        if previous_level:
            logger.setLevel(previous_level)
        else:
            logger.setLevel(logging.NOTSET)

        if cleanup_on_failure:
            try:
                shutil.rmtree(output)
            except OSError:
                logger.debug("Failed to remove output directory after error: %s", output)


def run_interactive_cli(
    default_source: Path | None = None,
    default_output: Path | None = None,
    *,
    r: int = 64,
    r_v: int = 8,
    top_l: int = 8,
    original_subdir: str | Path | None = None,
    verbose: bool = True,
    determinism: DeterminismSettings | None = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> ConversionSummary:
    """Guide the user through conversion via a lightweight text UI."""

    def _display(message: str = "") -> None:
        output_func(message)

    def _prompt_path(
        prompt: str,
        default: Path | None,
        *,
        must_exist: bool,
    ) -> Path:
        while True:
            suffix = f" [{default}]" if default is not None else ""
            response = input_func(f"{prompt}{suffix}: ").strip()
            if not response:
                if default is None:
                    _display("Please provide a path.")
                    continue
                path = default
            else:
                path = Path(response)
            path = path.expanduser()
            if must_exist and not path.exists():
                _display(f"Path {path} does not exist. Please try again.")
                continue
            return path

    def _prompt_int(prompt: str, default: int) -> int:
        while True:
            response = input_func(f"{prompt} [{default}]: ").strip()
            if not response:
                return default
            try:
                value = int(response)
            except ValueError:
                _display("Please enter an integer value.")
                continue
            if value <= 0:
                _display("Please provide a positive integer.")
                continue
            return value

    def _prompt_optional_string(prompt: str, default: str | None) -> str | None:
        suffix = f" [{default}]" if default else ""
        response = input_func(f"{prompt}{suffix}: ").strip()
        if not response:
            return default
        return response

    def _prompt_confirm(prompt: str, default: bool = True) -> bool:
        choice = "Y/n" if default else "y/N"
        while True:
            response = input_func(f"{prompt} ({choice}): ").strip().lower()
            if not response:
                return default
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            _display("Please answer 'y' or 'n'.")

    _display("Sera Transfer Conversion Assistant")
    _display("=" * 36)
    _display("Press Ctrl+C at any time to abort.\n")

    source = _prompt_path(
        "Checkpoint directory",
        default_source or Path.cwd(),
        must_exist=True,
    )
    output = _prompt_path(
        "Output directory",
        default_output or (source.parent / f"{source.name}-sera"),
        must_exist=False,
    )
    selected_r = _prompt_int("Rank parameter r", r)
    selected_rv = _prompt_int("Value rank rv", r_v)
    selected_topl = _prompt_int("Top-L parameter", top_l)
    selected_original = _prompt_optional_string(
        "Original subdirectory (optional)",
        str(original_subdir) if original_subdir is not None else None,
    )

    _display("\nConfiguration summary:")
    summary_block = textwrap.dedent(
        f"""
        Source : {source}
        Output : {output}
        r / rv / topL : {selected_r} / {selected_rv} / {selected_topl}
        Original subdir: {selected_original or '<auto>'}
        """
    ).strip("\n")
    _display(summary_block)

    if not _prompt_confirm("Proceed with conversion?", True):
        raise KeyboardInterrupt("Conversion cancelled by user")

    summary = convert(
        source,
        output,
        r=selected_r,
        r_v=selected_rv,
        top_l=selected_topl,
        original_subdir=selected_original,
        verbose=verbose,
        determinism=determinism,
    )
    _display("")
    _display(render_summary(summary, format="table"))
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert checkpoints into Sera Transfer Kit artefacts. "
            "Supports the official safetensors wheel for binary payloads; "
            "falls back to the repository's JSON stub for tests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to the checkpoint directory containing config.json and model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory where the converted artefacts should be written",
    )
    parser.add_argument("--r", type=int, default=64, help="Rank compression parameter")
    parser.add_argument(
        "--rv",
        type=int,
        default=8,
        help="Value compression rank used for rotary features",
    )
    parser.add_argument("--topL", type=int, default=8, help="Top-L sparse FFN parameter")
    parser.add_argument(
        "--original-subdir",
        type=str,
        default=None,
        help="Optional subdirectory that contains config.json and model.safetensors",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=None,
        help="Enable verbose logging (can also set SERA_TRANSFER_VERBOSE=1)",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Disable verbose logging",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive text UI to guide the conversion",
    )
    parser.add_argument(
        "--no-summary",
        dest="print_summary",
        action="store_false",
        help="Do not print the conversion summary table",
    )
    parser.add_argument(
        "--summary-format",
        choices=("table", "json"),
        default="table",
        help="Format to use when rendering the conversion summary",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional path to write the conversion summary to",
    )
    parser.add_argument(
        "--mxfp4-backend",
        choices=("python", "numpy", "torch"),
        default="python",
        help=(
            "Backend to use when decoding MXFP4 tensors. "
            "Non-python options require deterministic attestations."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="Pin the Torch thread count used for deterministic MXFP4 decoding.",
    )
    parser.add_argument(
        "--numpy-threads",
        type=int,
        help="Pin the NumPy thread count used for deterministic MXFP4 decoding.",
    )
    parser.add_argument(
        "--assume-ftz",
        action="store_true",
        help="Attest that the environment flushes denormal floats to zero (FTZ).",
    )
    parser.add_argument(
        "--assume-fma",
        action="store_true",
        help="Attest that fused multiply-add behaviour is deterministic.",
    )
    parser.set_defaults(print_summary=True)

    args = parser.parse_args(argv)

    def _expand_path(value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser()

    args.source = _expand_path(args.source)
    args.output = _expand_path(args.output)
    args.summary_output = _expand_path(args.summary_output)

    if args.original_subdir is not None:
        args.original_subdir = Path(args.original_subdir).expanduser()

    if args.verbose is None:
        env_value = os.environ.get("SERA_TRANSFER_VERBOSE")
        if env_value is None:
            args.verbose = False
        else:
            args.verbose = env_value.lower() not in {"", "0", "false", "no"}

    if not args.interactive:
        missing: List[str] = []
        if args.source is None:
            missing.append("--source")
        if args.output is None:
            missing.append("--output")
        if missing:
            parser.error(
                " and ".join(missing) + " required unless --interactive is supplied"
            )

    if args.torch_threads is not None and args.torch_threads <= 0:
        parser.error("--torch-threads must be a positive integer")
    if args.numpy_threads is not None and args.numpy_threads <= 0:
        parser.error("--numpy-threads must be a positive integer")

    backend = args.mxfp4_backend
    if backend == "torch":
        if args.torch_threads is None:
            parser.error("--torch-threads=1 is required when using --mxfp4-backend=torch")
        if args.torch_threads != 1:
            parser.error("Deterministic torch MXFP4 decoding requires --torch-threads=1")
        if not args.assume_ftz or not args.assume_fma:
            parser.error(
                "Torch MXFP4 decoding requires --assume-ftz and --assume-fma attestations"
            )
    elif backend == "numpy":
        if args.numpy_threads is None:
            parser.error("--numpy-threads=1 is required when using --mxfp4-backend=numpy")
        if args.numpy_threads != 1:
            parser.error("Deterministic NumPy MXFP4 decoding requires --numpy-threads=1")
        if not args.assume_ftz or not args.assume_fma:
            parser.error(
                "NumPy MXFP4 decoding requires --assume-ftz and --assume-fma attestations"
            )
    else:
        if args.torch_threads not in (None, 1):
            parser.error("--torch-threads must be 1 when specified with --mxfp4-backend=python")
        if args.numpy_threads not in (None, 1):
            parser.error("--numpy-threads must be 1 when specified with --mxfp4-backend=python")

    args.determinism = DeterminismSettings(
        mxfp4_backend=backend,
        torch_threads=args.torch_threads,
        numpy_threads=args.numpy_threads,
        assume_ftz=args.assume_ftz,
        assume_fma=args.assume_fma,
    )

    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.interactive:
        run_interactive_cli(
            args.source,
            args.output,
            r=args.r,
            r_v=args.rv,
            top_l=args.topL,
            original_subdir=args.original_subdir,
            verbose=args.verbose,
            determinism=args.determinism,
        )
        return

    try:
        summary = convert(
            args.source,
            args.output,
            r=args.r,
            r_v=args.rv,
            top_l=args.topL,
            original_subdir=args.original_subdir,
            verbose=args.verbose,
            determinism=args.determinism,
        )
    except ModuleNotFoundError as exc:
        message = str(exc) or _SAFETENSORS_MISSING_MSG
        hint = ""
        if args.output is not None:
            hint = f" No artefacts were written to {args.output}."
        print(f"sera_transfer: {message}{hint}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        rendered = render_summary(summary, format=args.summary_format)
    except ValueError as exc:  # pragma: no cover - defensive
        raise SystemExit(str(exc)) from exc

    if args.summary_output is not None:
        summary_path = args.summary_output.expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        text = rendered if rendered.endswith("\n") else rendered + "\n"
        summary_path.write_text(text, encoding="utf-8")

    if args.print_summary:
        print(rendered)


if __name__ == "__main__":
    main()
