"""Deterministic conversion of checkpoints into Sera Transfer Kit artefacts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import importlib.util
import io
import json
import logging
import math
import os
import pickle
import re
import struct
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised indirectly in environments with safetensors
    from safetensors import safe_open
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without safetensors
    safe_open = None  # type: ignore[assignment]


def _looks_like_repo_stub_safe_open(func) -> bool:
    if func is None:
        return False

    module = inspect.getmodule(func)
    if module is None:
        return False

    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False

    try:
        module_path = Path(module_file).resolve()
    except OSError:  # pragma: no cover - defensive
        return False

    try:
        repo_root = Path(__file__).resolve().parents[3]
    except IndexError:  # pragma: no cover - defensive
        return False

    stub_path = (repo_root / "safetensors" / "__init__.py").resolve()
    if module_path == stub_path:
        return True

    if module_path.name == "__init__.py" and module_path.parent.name == "safetensors":
        try:
            module_path.relative_to(repo_root)
        except ValueError:
            return False
        return True

    return False

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


def _safe_exists(path: Path, *, verbose: bool = False) -> bool:
    """Return ``True`` if *path* exists, suppressing ``OSError`` failures."""

    try:
        return path.exists()
    except OSError as exc:  # pragma: no cover - defensive logging branch
        if verbose:
            logger.warning("Failed to stat %s: %s", path, exc)
        return False


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


MAGIC_SERA_ARRAY = 0x53455241
JSON_BYTES_PREFIX = "__sera_bytes__:"


def _json_key(value: object) -> str:
    if isinstance(value, bytes):
        return JSON_BYTES_PREFIX + value.hex()
    if isinstance(value, str):
        return value
    return str(value)


def _json_value(value: object):
    if isinstance(value, dict):
        return {
            _json_key(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bytes):
        return JSON_BYTES_PREFIX + value.hex()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return _json_value(vars(value))
    return value


def _flatten(values: Iterable) -> List[float]:
    result: List[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            result.extend(_flatten(item))
        else:
            result.append(float(item))
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
    raise TypeError(f"Unsupported tensor type: {type(values)!r}")


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


def _tensor_bytes(tensor: List) -> bytes:
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


def gram_schmidt(columns: List[List[float]]) -> List[List[float]]:
    orth: List[List[float]] = []
    for vec in columns:
        v = list(vec)
        for basis in orth:
            dot = sum(a * b for a, b in zip(v, basis))
            v = [a - dot * b for a, b in zip(v, basis)]
        norm = math.sqrt(sum(a * a for a in v))
        if norm < 1e-12:
            raise ValueError("Vectors are linearly dependent")
        v = [a / norm for a in v]
        orth.append(v)
    return orth


def orthonormal_matrix(rows: int, cols: int, generator: Iterable[List[float]]) -> List[List[float]]:
    columns = []
    for column in generator:
        columns.append(column[:rows])
        if len(columns) == cols:
            break
    orth = gram_schmidt(columns)
    return transpose(orth)


# ---------------------------------------------------------------------------
# Model configuration


@dataclass
class LayerConfig:
    name: str
    w_k: str
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


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    head_dim: int
    vocab_size: int
    tau: float
    layers: List[LayerConfig]
    rope_theta: float | None = None

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
            doc_hint = "docs/howto-sera-transfer.md"
            raise ValueError(
                "Model config does not provide 'head_dim'. Provide the Sera schema "
                "(d_model/n_heads/head_dim) or a Hugging Face config with "
                "hidden_size/num_attention_heads/head_dim. See "
                f"{doc_hint} for details."
            )

        if d_model is None or n_heads is None or head_dim is None:
            doc_hint = "docs/howto-sera-transfer.md"
            raise ValueError(
                "Model config must define d_model, n_heads, and head_dim (or provide "
                "their Hugging Face equivalents such as hidden_size, "
                "num_attention_heads, and head_dim). See "
                f"{doc_hint} for the expected schema."
            )

        vocab_size = int(data.get("vocab_size", 0) or 0)
        tau = float(data.get("tau", 1.0))
        rope_theta = data.get("rope_theta")

        layer_entries = list(data.get("layers", []) or [])
        layers: List[LayerConfig]
        if layer_entries:
            layers = []
            for idx, layer in enumerate(layer_entries):
                layer_data = layer  # type: ignore[assignment]
                layers.append(
                    LayerConfig(
                        name=str(layer_data.get("name", f"layer_{idx}")),
                        w_k=str(layer_data["W_K"]),
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

        return ModelConfig(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            tau=tau,
            layers=layers,
            rope_theta=float(rope_theta) if rope_theta is not None else None,
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
            "W_K": (
                "attn.k.weight",
                "attn.wk.weight",
                "attn.k_proj.weight",
                "attention.k.weight",
                "attention.wk.weight",
                "attention.k_proj.weight",
                "self_attn.k_proj.weight",
            ),
            "W_O": (
                "attn.o.weight",
                "attn.wo.weight",
                "attn.o_proj.weight",
                "attention.o.weight",
                "attention.wo.weight",
                "attention.o_proj.weight",
                "attention.out_proj.weight",
                "self_attn.o_proj.weight",
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
            ),
            "FFN_W2": (
                "ffn.w2.weight",
                "mlp.w2.weight",
                "ffn.down_proj.weight",
                "mlp.down_proj.weight",
                "ffn.proj.weight",
                "feed_forward.w2.weight",
                "feed_forward.down_proj.weight",
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
            ),
            "FFN_B2": (
                "ffn.w2.bias",
                "mlp.w2.bias",
                "ffn.down_proj.bias",
                "mlp.down_proj.bias",
                "ffn.proj.bias",
                "feed_forward.w2.bias",
                "feed_forward.down_proj.bias",
            ),
        }

        expected_proj = n_heads * head_dim
        layer_map: Dict[str, Dict[str, str]] = {}

        for name, tensor in tensors.items():
            if not isinstance(name, str):
                continue
            lowered = name.lower()
            shape = _tensor_shape(tensor)
            for role, suffixes in role_suffixes.items():
                matched = False
                for suffix in suffixes:
                    if lowered.endswith(suffix):
                        if not ModelConfig._shape_allows_role(
                            role, shape, expected_proj
                        ):
                            continue
                        prefix = name[: len(name) - len(suffix)].rstrip(".")
                        if not prefix:
                            break
                        layer_roles = layer_map.setdefault(prefix, {})
                        layer_roles.setdefault(role, name)
                        matched = True
                        break
                if matched:
                    break

        required_roles = {"W_K", "W_O", "FFN_W1", "FFN_W2", "FFN_B1", "FFN_B2"}
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
                    w_k=roles["W_K"],
                    w_o=roles["W_O"],
                    w1=roles["FFN_W1"],
                    w2=roles["FFN_W2"],
                    b1=roles["FFN_B1"],
                    b2=roles["FFN_B2"],
                )
            )
        return layers

    @staticmethod
    def _shape_allows_role(
        role: str,
        shape: Tuple[int, ...],
        expected_proj: int,
    ) -> bool:
        if len(shape) < 2:
            return True
        rows, cols = shape[0], shape[1]
        if role == "W_K":
            return expected_proj in {rows, cols}
        if role == "W_O":
            return expected_proj in {rows, cols}
        return True


# ---------------------------------------------------------------------------
# Tensor utilities


_SAFETENSORS_MISSING_MSG = (
    "The `safetensors` package is required to load model checkpoints. "
    "Install it via `pip install safetensors` or provide a preloaded tensor map."
)


def _resolve_safe_open():
    global safe_open
    if safe_open is not None:
        return safe_open
    try:  # pragma: no cover - exercised when lazy import succeeds
        from safetensors import safe_open as imported_safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise ModuleNotFoundError(_SAFETENSORS_MISSING_MSG) from exc
    safe_open = imported_safe_open
    return safe_open


def load_tensors(path: Path) -> Dict[str, List]:
    safe_open_fn = _resolve_safe_open()

    safe_open_is_stub = _looks_like_repo_stub_safe_open(safe_open_fn)
    stub_hint = (
        "The repository includes a JSON-only stub of `safetensors`. "
        "Install the official `safetensors` wheel (for example, via `pip install safetensors`) "
        "to load binary checkpoints."
    )

    if safe_open_is_stub:
        try:
            with path.open("rb") as fh:
                sample = fh.read(1024)
        except OSError:
            sample = b""

        if sample and not sample.lstrip().startswith(b"{"):
            raise ModuleNotFoundError(stub_hint)

    tensors: Dict[str, List] = {}
    try:
        with safe_open_fn(path, framework="python") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                try:
                    tensors[key] = tensor.tolist()
                except AttributeError:
                    tensors[key] = tensor  # pragma: no cover - defensive fallback
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if safe_open_is_stub:
            raise ModuleNotFoundError(stub_hint) from exc
        raise
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


def compute_prf(cfg: ModelConfig, tensors: Mapping[str, List], r: int) -> Dict[str, List]:
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
    for _ in range(cols):
        column = [1.0 if (prng.next() & 1) == 0 else -1.0 for _ in range(rows)]
        yield column


def gaussian_matrix(prng: SplitMix64, rows: int, cols: int) -> List[List[float]]:
    matrix: List[List[float]] = []
    for _ in range(rows):
        matrix.append(gaussian_vector(prng, cols))
    return matrix


def compute_overlays(cfg: ModelConfig, tensors: Mapping[str, List], r: int, r_v: int) -> Dict[str, List]:
    accumulator: Optional[List[List[float]]] = None
    layer_count = 0
    digest = hashlib.sha256()
    for layer in cfg.layers:
        w_o = tensors.get(layer.w_o)
        if w_o is None:
            continue
        digest.update(layer.w_o.encode("utf-8"))
        digest.update(_tensor_bytes(w_o))
        layer_rows = len(w_o)
        layer_cols = len(w_o[0]) if w_o else 0
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


def matrix_abs(matrix: List[List[float]]) -> List[List[float]]:
    return [[abs(value) for value in row] for row in matrix]


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
    cfg: ModelConfig, tensors: Mapping[str, List], top_l: int
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
                sign_sum = sum(W2[out_idx][h] for out_idx in range(len(W2)))
                effect += sign(sign_sum) * max(0.0, float(W1[h][feature]))
            key = (layer_index << 32) | feature
            weight_map[key] = effect

            for h in range(hidden_dim):
                if h in idxs:
                    continue
                sign_sum = sum(W2[out_idx][h] for out_idx in range(len(W2)))
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


def _layer_seed(layer: LayerConfig, tensors: Mapping[str, List]) -> int:
    digest = hashlib.sha256()
    for name in (layer.w_k, layer.w_o, layer.w1, layer.w2, layer.b1, layer.b2):
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
    cfg: ModelConfig, tensors: Mapping[str, List], vocab_size: int, W: int = 2
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
            bias1 = float(b1[feature_h % len(b1)]) if b1 else 0.0
            bias2 = float(b2[feature_out % len(b2)]) if b2 else 0.0
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
    cfg: ModelConfig, tensors: Mapping[str, List], max_len: int = 4
) -> Tuple[Dict[str, List], Dict[str, object]]:
    vocab = cfg.vocab_size or 256
    vocab = max(1, vocab)

    tensor_digest = hashlib.sha256()
    embedding: Optional[List[List[float]]] = None
    for name in sorted(tensors.keys()):
        tensor = tensors[name]
        try:
            tensor_bytes = _tensor_bytes(tensor)
        except Exception:  # pragma: no cover - defensive
            continue
        tensor_digest.update(name.encode("utf-8"))
        tensor_digest.update(tensor_bytes)
        if (
            embedding is None
            and isinstance(tensor, list)
            and tensor
            and isinstance(tensor[0], list)
            and len(tensor) >= vocab
        ):
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


def write_manifest(path: Path, cfg: ModelConfig, artefacts: Mapping[str, bytes], *, r: int, r_v: int, vocab_size: int) -> None:
    schema_path = Path("docs/specs/Sera-Transfer.txt")
    schema_digest = hashlib.sha256(schema_path.read_bytes()).digest() if schema_path.exists() else hashlib.sha256(b"sera").digest()
    seed_digest = hashlib.sha256(b"sera-transfer").digest()

    with path.open("wb") as f:
        f.write(struct.pack("<I", 0x5345524D))
        f.write(struct.pack("<I", 0x3))
        f.write(seed_digest)
        f.write(schema_digest)

        L_tok = 4
        S_norm = 1.0
        L_norm = 1.0
        P_gen = vocab_size
        sp_cert = hashlib.sha256(artefacts.get("tokenizer_fst", b""))
        f.write(struct.pack("<IffI", L_tok, S_norm, L_norm, P_gen))
        f.write(sp_cert.digest())
        _write_utf8(f, "byte_level")

        f.write(struct.pack("<Ifdd", r, cfg.tau, 1e-8, 3.0))
        f.write(struct.pack("<d", 1e-3))
        f.write(hashlib.sha256(b"lambda").digest())

        C = len(artefacts.get("linear_weights", b"")) // 4 if artefacts.get("linear_weights") else 0
        f.write(struct.pack("<5Idd", C, cfg.d_model, 2, 1, 8, 0.1, 1.0))

        f.write(struct.pack("<6Id", 1, 2, 1, 1, 32, 4, 0.95))
        f.write(struct.pack("<III", cfg.n_heads, cfg.head_dim, r_v))
        _write_utf8(f, "l2")
        f.write(struct.pack("<fi", 0.5, 4))

        proj_digest = hashlib.sha256(artefacts.get("bridge_hubs", b""))
        f.write(struct.pack("<III", vocab_size, 2, cfg.d_model))
        f.write(proj_digest.digest())
        f.write(struct.pack("<ff", 0.1, 1.0))
        f.write(struct.pack("<II", 0, 0))
        f.write(struct.pack("<III", 4, 2, 8))

        f.write(struct.pack("<IIIIfff", 8, 16, 4, 2, 1.3, 0.01, 0.05))
        f.write(struct.pack("<f f f f", 0.5, 20.0, 0.1, 0.05))
        f.write(hashlib.sha256(artefacts.get("linear_mphf", b"" )).digest())
        f.write(hashlib.sha256(artefacts.get("linear_keys", b"" )).digest())
        f.write(hashlib.sha256(b"prev").digest())
        f.write(hashlib.sha256(b"curr").digest())


def _write_utf8(f: io.BufferedWriter, value: str) -> None:
    data = value.encode("utf-8")
    f.write(struct.pack("<H", len(data)))
    f.write(data)


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
) -> ConversionSummary:
    """Convert a checkpoint directory into a Sera Transfer Kit artefact.

    The conversion expects a ``config.json`` and ``model.safetensors`` file. When
    these files are missing from ``source`` directly, the function probes a small
    set of common Hugging Face layouts – ``source/original`` and
    ``source/original/model`` – before failing. Advanced users can supply an
    explicit ``original_subdir`` to search an arbitrary additional location.
    """

    source = source.resolve()
    output = output.resolve()

    if verbose and not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    if verbose:
        logger.info("Resolved source path: %s", source)
        logger.info("Resolved output path: %s", output)

    search_roots: List[Path] = []

    def add_root(candidate: Path | str) -> None:
        path = candidate if isinstance(candidate, Path) else Path(candidate)
        if not path.is_absolute():
            path = source / path
        if path not in search_roots:
            search_roots.append(path)
            if verbose:
                logger.info("Registered search root: %s", path)

    add_root(source)
    if original_subdir is not None:
        add_root(original_subdir)
    else:
        need_probe = any(
            not _safe_exists(source / filename, verbose=verbose)
            for filename in ("config.json", "model.safetensors")
        )
        if need_probe:
            add_root("original")
            add_root(Path("original") / "model")

    def find_file(filename: str) -> Path:
        for root in search_roots:
            candidate = root / filename
            if verbose:
                logger.info("Probing %s", candidate)
            if _safe_exists(candidate, verbose=verbose):
                if verbose:
                    logger.info("Located %s at %s", filename, candidate)
                return candidate
        search_list = ", ".join(str(root) for root in search_roots)
        message = f"Unable to locate {filename!r}; searched: {search_list or source}"
        if verbose:
            logger.error(message)
        raise FileNotFoundError(message)

    def _convert_inner() -> ConversionSummary:
        config_path = find_file("config.json")
        model_path = find_file("model.safetensors")
        if verbose:
            logger.info("Reading model configuration from %s", config_path)
        config_data = json.loads(config_path.read_text())
        if verbose and isinstance(config_data, dict):
            config_keys = ", ".join(sorted(config_data)) or "<none>"
            logger.info("Model config keys: %s", config_keys)
        tensors = load_tensors(model_path)
        cfg = ModelConfig.from_dict(config_data, tensors=tensors)

        local_r = min(r, cfg.d_model)
        local_r_v = min(r_v, local_r)

        arrays_dir = output / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        artefact_payloads: Dict[str, bytes] = {}
        artefact_records: Dict[str, Dict[str, object]] = {}
        metadata: Dict[str, object] = {}
        array_infos: List[ArrayInfo] = []
        snapshot_infos: List[SnapshotInfo] = []

        def store_array(name: str, data, dtype: str) -> None:
            path = arrays_dir / f"{name}.bin"
            payload = write_array(path, data, dtype)
            shape = _infer_shape(data)
            payload_len = len(payload)
            artefact_payloads[name] = payload
            artefact_records[name] = {
                "path": str(path),
                "dtype": dtype,
                "shape": shape,
                "bytes": payload_len,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            array_infos.append(
                ArrayInfo(
                    name=name,
                    path=path,
                    dtype=dtype,
                    shape=shape,
                    bytes=payload_len,
                    sha256=artefact_records[name]["sha256"],
                )
            )

        tokenizer_data, tokenizer_meta = tokenizer_arrays(cfg, tensors)
        for name, data in tokenizer_data.items():
            store_array(name, data, "u8")
        metadata["tokenizer"] = tokenizer_meta

        prf = compute_prf(cfg, tensors, local_r)
        for name, data in prf.items():
            store_array(name, data, "f32")
        metadata["attention"] = {
            "features": local_r,
            "tau": getattr(cfg, "tau", 1.0),
            "whitening_sig2": prf.get("whitening_sig2", []),
        }

        overlays = compute_overlays(cfg, tensors, local_r, local_r_v)
        for name, data in overlays.items():
            store_array(name, data, "f32")
        metadata["overlays"] = {
            "rank": local_r,
            "rank_value": local_r_v,
            "rows": len(overlays.get("overlays_H", [])),
            "cols": len(overlays.get("overlays_U", [])),
        }

        linear_data, linear_meta = collapse_ffn(cfg, tensors, top_l)
        store_array("linear_mphf", linear_data["linear_mphf"], "u8")
        store_array("linear_keys", linear_data["linear_keys"], "u8")
        store_array("linear_weights", linear_data["linear_weights"], "f32")
        store_array("linear_bias", linear_data["linear_bias"], "f32")
        store_array("cuckoo_delta", linear_data["cuckoo_delta"], "u8")
        metadata["linear"] = linear_meta

        memory_data, memory_meta = memory_coefficients(cfg)
        store_array("memory_coeff", memory_data["memory_coeff"], "f64")
        store_array("delaybuf_init", memory_data["delaybuf_init"], "f32")
        metadata["memory"] = memory_meta

        bridge_data, bridge_meta = bridge_records(cfg, tensors, cfg.vocab_size)
        store_array("bridge_hubs", bridge_data["bridge_hubs"], "u8")
        store_array("bridge_qDin", bridge_data["bridge_qDin"], "i16")
        store_array("bridge_qDout", bridge_data["bridge_qDout"], "i16")
        store_array("peer_scores", bridge_data["peer_scores"], "i16")
        metadata["bridge"] = bridge_meta

        manifest_path = output / "sera_manifest.bin"
        write_manifest(
            manifest_path,
            cfg,
            artefact_payloads,
            r=local_r,
            r_v=local_r_v,
            vocab_size=cfg.vocab_size or 16,
        )

        sera_cls, sera_config_cls = _load_sera_runtime()
        try:
            base_config = sera_config_cls()
        except Exception:  # pragma: no cover - extremely defensive
            base_config = None

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
            linear_config = dataclasses.replace(base_config.linear, capacity=linear_capacity)
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
        )
        if verbose:
            formatted = format_summary(summary)
            for line in formatted.splitlines():
                logger.info(line)
        return summary

    try:
        return _convert_inner()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        if verbose:
            logger.error("Conversion failed: %s", exc)
        raise


def run_interactive_cli(
    default_source: Path | None = None,
    default_output: Path | None = None,
    *,
    r: int = 64,
    r_v: int = 8,
    top_l: int = 8,
    original_subdir: str | Path | None = None,
    verbose: bool = True,
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
    )
    _display("")
    _display(render_summary(summary, format="table"))
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert checkpoints into Sera Transfer Kit artefacts",
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
    parser.set_defaults(print_summary=True)

    args = parser.parse_args(argv)

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
        )
        return

    summary = convert(
        args.source,
        args.output,
        r=args.r,
        r_v=args.rv,
        top_l=args.topL,
        original_subdir=args.original_subdir,
        verbose=args.verbose,
    )

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
