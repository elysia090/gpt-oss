"""Deterministic conversion of checkpoints into Sera Transfer Kit artefacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pickle
import math
import struct
import textwrap
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import sys

try:  # pragma: no cover - exercised indirectly in environments with safetensors
    from safetensors import safe_open
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without safetensors
    safe_open = None  # type: ignore[assignment]

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
    if isinstance(config, dict):
        return {key: _config_to_dict(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_config_to_dict(value) for value in config]
    return config

__all__ = [
    "ArrayHeader",
    "crc32c",
    "sha256_low64",
    "SplitMix64",
    "convert",
    "main",
]


MAGIC_SERA_ARRAY = 0x53455241


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
    def from_dict(data: Mapping[str, object]) -> "ModelConfig":
        layers = []
        for idx, layer in enumerate(data["layers"]):
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
        return ModelConfig(
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            head_dim=int(data["head_dim"]),
            vocab_size=int(data.get("vocab_size", 0) or 0),
            tau=float(data.get("tau", 1.0)),
            layers=layers,
            rope_theta=float(data.get("rope_theta")) if data.get("rope_theta") is not None else None,
        )


# ---------------------------------------------------------------------------
# Tensor utilities


def load_tensors(path: Path) -> Dict[str, List]:
    if safe_open is None:
        raise ModuleNotFoundError(
            "The `safetensors` package is required to load model checkpoints. "
            "Install it via `pip install safetensors` or provide a preloaded tensor map."
        )
    tensors: Dict[str, List] = {}
    with safe_open(path, framework="python") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            try:
                tensors[key] = tensor.tolist()
            except AttributeError:
                tensors[key] = tensor  # pragma: no cover - defensive fallback
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


def compute_prf(layer: LayerConfig, cfg: ModelConfig, tensors: Mapping[str, List], r: int) -> Dict[str, List]:
    w_k = tensors[layer.w_k]
    diag = _column_means_squared(w_k)
    prng = SplitMix64(seed=0xC0FFEE)
    prf_W: List[List[float]] = []
    for _ in range(r):
        gaussian = gaussian_vector(prng, cfg.d_model)
        scaled = [g * math.sqrt(diag[j % len(diag)]) for j, g in enumerate(gaussian)]
        prf_W.append(scaled)
    whitening_mu = [0.0 for _ in range(r)]
    whitening_sig2 = [1.0 for _ in range(r)]
    r_init = [[0.0 for _ in range(cfg.d_model)] for _ in range(r)]
    s_init = [0.0 for _ in range(r)]
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


def compute_overlays(layer: LayerConfig, cfg: ModelConfig, tensors: Mapping[str, List], r: int, r_v: int) -> Dict[str, List]:
    w_o = tensors[layer.w_o]
    prng = SplitMix64(seed=0x5EED)
    H = orthonormal_matrix(r, r_v, hadamard_generator(prng, r, r_v))
    Omega = gaussian_matrix(prng, len(w_o), r_v)
    Y = matmul(transpose(w_o), Omega)
    U = orthonormal_matrix(len(w_o[0]), r_v, transpose(Y))
    Z = matmul(transpose(U), matmul(w_o, H))
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


def collapse_ffn(
    cfg: ModelConfig, tensors: Mapping[str, List], top_l: int
) -> Tuple[Dict[str, List], Dict[str, object]]:
    weight_map: Dict[int, float] = {}
    key_order: List[int] = []
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
            idxs = top_indices(scores, top_l)
            effect = 0.0
            for h in idxs:
                sign_sum = sum(W2[out_idx][h] for out_idx in range(len(W2)))
                effect += sign(sign_sum) * max(0.0, float(W1[h][feature]))
            key = (layer_index << 32) | feature
            weight_map[key] = effect
            key_order.append(key)

    capacity = len(weight_map)
    table_size = max(1, int(math.ceil(1.23 * capacity)))
    slots: List[Optional[int]] = [None] * capacity
    occupied = [False] * capacity
    seeds: List[int] = [0] * table_size
    buckets: Dict[int, List[int]] = {}
    for key in key_order:
        bucket = _mix64(key) % table_size
        buckets.setdefault(bucket, []).append(key)

    def assign_positions(options: List[Tuple[int, int]]) -> Optional[List[int]]:
        placement: List[int] = []
        used: set[int] = set()
        for pos0, pos1 in options:
            choice = None
            if not occupied[pos0] and pos0 not in used:
                choice = pos0
            elif not occupied[pos1] and pos1 not in used:
                choice = pos1
            if choice is None:
                return None
            used.add(choice)
            placement.append(choice)
        return placement

    ordered = sorted(buckets.items(), key=lambda item: (len(item[1]), item[0]), reverse=True)
    for bucket, bucket_keys in ordered:
        local_keys = sorted(bucket_keys)
        for seed_candidate in range(1 << 16):
            options: List[Tuple[int, int]] = []
            for key in local_keys:
                pos0 = _mix64(key, seed_candidate) % capacity
                pos1 = _mix64(key, seed_candidate + 1) % capacity
                options.append((pos0, pos1))
            assignment = assign_positions(options)
            if assignment is not None:
                seeds[bucket] = seed_candidate
                for key, slot in zip(local_keys, assignment):
                    slots[slot] = key
                    occupied[slot] = True
                break
        else:  # pragma: no cover - deterministic bound ensures termination
            raise RuntimeError("Unable to build FFN minimal perfect hash")

    slot_keys: List[int] = [key for key in slots if key is not None]
    mphf = [[(seed >> (8 * i)) & 0xFF for i in range(4)] for seed in seeds]
    key_bytes = [[(key >> (8 * i)) & 0xFF for i in range(8)] for key in slot_keys]
    weights = [weight_map[key] for key in slot_keys]

    arrays = {
        "linear_mphf": mphf,
        "linear_keys": key_bytes,
        "linear_weights": weights,
        "linear_bias": [base_bias],
        "cuckoo_delta": [],
    }
    metadata = {
        "keys": slot_keys,
        "weights": weights,
        "bias": base_bias,
        "table_size": table_size,
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
    for layer in cfg.layers:
        seeds.append(_layer_seed(layer, tensors))
        W1 = tensors[layer.w1]
        W2 = tensors[layer.w2]
        b1 = tensors[layer.b1]
        b2 = tensors[layer.b2]
        hidden_dim = len(W1)
        for token in range(vocab):
            feature = token % cfg.d_model
            avg_w1 = sum(float(W1[h][feature]) for h in range(hidden_dim)) / max(1, hidden_dim)
            avg_w2 = sum(float(value) for value in W2[feature]) / max(1, len(W2[feature]))
            bias1 = float(b1[feature % len(b1)]) if b1 else 0.0
            bias2 = float(b2[feature % len(b2)]) if b2 else 0.0
            aggregated_in[token] += avg_w1 + bias1
            aggregated_out[token] += avg_w2 + bias2

    global_seed = 0
    for idx, seed in enumerate(seeds):
        global_seed ^= _mix64(seed, idx + 1)

    for token in range(vocab):
        prng = SplitMix64(global_seed ^ token)
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
    }
    return arrays, metadata


# ---------------------------------------------------------------------------
# Tokenizer placeholders


def tokenizer_arrays(
    cfg: ModelConfig, tensors: Mapping[str, List], max_len: int = 4
) -> Tuple[Dict[str, List], Dict[str, object]]:
    vocab = cfg.vocab_size or 256
    vocab = max(1, vocab)
    pieces: List[Tuple[bytes, int]] = []

    layer_seeds = [_layer_seed(layer, tensors) for layer in cfg.layers]
    global_seed = 0
    for idx, seed in enumerate(layer_seeds):
        global_seed ^= _mix64(seed, idx + 1)
    global_seed ^= _mix64(vocab, len(layer_seeds) + 1)

    prng = SplitMix64(global_seed)
    sequences: List[Tuple[int, bytes]] = []
    for token in range(vocab):
        seed = prng.next()
        local = SplitMix64(seed)
        length = 1 + (local.next() % max_len)
        seq = bytes((local.next() & 0xFF) for _ in range(length))
        sequences.append((token, seq))
        pieces.append((seq, token))

    lines = ["% byte level fst derived from checkpoint", "start 0", "end 0"]
    next_state = 1
    for token, piece in sequences:
        current = 0
        for idx, byte in enumerate(piece):
            last = idx == len(piece) - 1
            if last:
                dest = 0
                output = token
            else:
                dest = next_state
                next_state += 1
                output = 0
            lines.append(f"{current} {dest} {byte} {output}")
            current = dest
    fst_text = "\n".join(lines).encode("utf-8")

    tables: Dict[str, List[int]] = {}
    modulus = 1 << 64
    for length in range(1, max_len + 1):
        factor = pow(257, length - 1, modulus)
        table = []
        for byte in range(256):
            value = (factor * byte + global_seed) & 0xFFFFFFFFFFFFFFFF
            table.append(value & 0xFF)
        tables[f"T_{length}"] = table

    arrays = {"tokenizer_fst": list(fst_text)}
    arrays.update(tables)
    metadata = {
        "pieces": pieces,
        "max_piece_length": max_len,
        "seed": global_seed,
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
) -> None:
    """Convert a checkpoint directory into a Sera Transfer Kit artefact.

    The conversion expects a ``config.json`` and ``model.safetensors`` file.  When
    these files are not present in ``source`` directly, the function probes a
    small set of common Hugging Face layouts – ``source/original`` and
    ``source/original/model`` – before failing.  Advanced users can supply an
    explicit ``original_subdir`` to search an arbitrary additional location.
    """

    search_roots: List[Path] = []

    def add_root(candidate: Path | str) -> None:
        path = candidate if isinstance(candidate, Path) else Path(candidate)
        if not path.is_absolute():
            path = source / path
        if path not in search_roots:
            search_roots.append(path)

    add_root(source)
    if original_subdir is not None:
        add_root(original_subdir)
    else:
        add_root("original")
        add_root(Path("original") / "model")

    def find_file(filename: str) -> Path:
        for root in search_roots:
            candidate = root / filename
            if candidate.exists():
                return candidate
        search_list = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            f"Unable to locate {filename!r}; searched: {search_list or source}"
        )

    config_path = find_file("config.json")
    model_path = find_file("model.safetensors")
    cfg = ModelConfig.from_dict(json.loads(config_path.read_text()))
    tensors = load_tensors(model_path)

    r = min(r, cfg.d_model)
    r_v = min(r_v, r)

    arrays_dir = output / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    artefact_payloads: Dict[str, bytes] = {}
    artefact_records: Dict[str, Dict[str, object]] = {}
    metadata: Dict[str, object] = {}

    def store_array(name: str, data, dtype: str) -> None:
        path = arrays_dir / f"{name}.bin"
        payload = write_array(path, data, dtype)
        artefact_payloads[name] = payload
        artefact_records[name] = {
            "path": str(path),
            "dtype": dtype,
            "shape": _infer_shape(data),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    tokenizer_data, tokenizer_meta = tokenizer_arrays(cfg, tensors)
    for name, data in tokenizer_data.items():
        store_array(name, data, "u8")
    metadata["tokenizer"] = tokenizer_meta

    prf = compute_prf(cfg.layers[0], cfg, tensors, r)
    for name, data in prf.items():
        store_array(name, data, "f32")

    overlays = compute_overlays(cfg.layers[0], cfg, tensors, r, r_v)
    for name, data in overlays.items():
        store_array(name, data, "f32")

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

    write_manifest(output / "sera_manifest.bin", cfg, artefact_payloads, r=r, r_v=r_v, vocab_size=cfg.vocab_size or 16)

    snapshot_path = output / "sera_state.pkl"
    snapshot = {
        "model_config": _config_to_dict(cfg),
        "artefacts": artefact_records,
        "metadata": metadata,
        "manifest_path": str(output / "sera_manifest.bin"),
    }
    with snapshot_path.open("wb") as fh:
        pickle.dump(snapshot, fh)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert checkpoints into Sera Transfer Kit artefacts")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--rv", type=int, default=8)
    parser.add_argument("--topL", type=int, default=8)
    parser.add_argument(
        "--original-subdir",
        type=str,
        default=None,
        help="Optional subdirectory that contains config.json and model.safetensors",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    convert(
        args.source,
        args.output,
        r=args.r,
        r_v=args.rv,
        top_l=args.topL,
        original_subdir=args.original_subdir,
    )


if __name__ == "__main__":
    main()
