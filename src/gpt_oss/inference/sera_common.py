"""Shared Sera runtime helpers used across the CLI and tooling."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Sequence, Tuple, Union

JSON_BYTES_PREFIX = "__sera_bytes__:"
ARRAY_MAGIC = 0x53455241
MANIFEST_MAGIC = 0x5345524D
MANIFEST_VERSION = 0x3
ARRAY_HEADER_STRUCT = struct.Struct("<I H H 5Q Q Q Q I I")
DTYPE_CODES: Dict[int, str] = {
    1: "f64",
    2: "f32",
    3: "i32",
    4: "i16",
    5: "i8",
    6: "u8",
    7: "q8_8",
    8: "q4_12",
    9: "bf16",
}
DEFAULT_STATE_FILENAMES: Tuple[str, ...] = (
    "sera_state.json",
    "sera_state.msgpack",
    "sera_state.pkl",
)
PICKLE_SUFFIXES = {".pkl", ".pickle"}


@dataclass(frozen=True)
class SeraArrayHeader:
    """Metadata describing a single Sera array payload."""

    dtype_code: int
    dtype: str
    rank: int
    dims: Tuple[int, ...]
    shape: Tuple[int, ...]
    byte_len: int
    crc32c: int
    sha256_low64: int
    flags: int
    reserved: int


class SeraArrayError(RuntimeError):
    """Raised when a Sera array payload fails validation."""


class SeraSnapshotFormatError(RuntimeError):
    """Raised when a runtime snapshot uses an unsupported format."""


def parse_array_header(values: Sequence[int]) -> SeraArrayHeader:
    """Parse the packed header stored alongside Sera arrays."""

    magic = int(values[0])
    if magic != ARRAY_MAGIC:
        raise SeraArrayError("Invalid Sera array header magic")
    dtype_code = int(values[1])
    rank = int(values[2])
    dims = tuple(int(dim) for dim in values[3:8])
    shape = tuple(dim for dim in dims[:rank]) if rank > 0 else tuple()
    header = SeraArrayHeader(
        dtype_code=dtype_code,
        dtype=DTYPE_CODES.get(dtype_code, f"code{dtype_code}"),
        rank=rank,
        dims=dims,
        shape=shape,
        byte_len=int(values[8]),
        crc32c=int(values[9]),
        sha256_low64=int(values[10]),
        flags=int(values[11]),
        reserved=int(values[12]),
    )
    return header


def load_array_file(path: Path) -> tuple[SeraArrayHeader, bytes]:
    """Read and validate the array payload stored on disk."""

    with path.open("rb") as fh:
        header_raw = fh.read(ARRAY_HEADER_STRUCT.size)
        if len(header_raw) != ARRAY_HEADER_STRUCT.size:
            raise SeraArrayError(f"Array file {path} is truncated")
        values = ARRAY_HEADER_STRUCT.unpack(header_raw)
        payload = fh.read()
    header = parse_array_header(values)
    if len(payload) != header.byte_len:
        raise SeraArrayError(
            f"Array payload length mismatch for {path}: "
            f"expected {header.byte_len}, got {len(payload)}"
        )
    return header, payload


def encode_snapshot_blob(value: object) -> object:
    """Encode binary payloads so they survive round-trips through JSON."""

    if isinstance(value, dict):
        return {
            encode_snapshot_blob(key): encode_snapshot_blob(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [encode_snapshot_blob(item) for item in value]
    if isinstance(value, tuple):
        return [encode_snapshot_blob(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return JSON_BYTES_PREFIX + bytes(value).hex()
    if isinstance(value, Path):
        return str(value)
    return value


def decode_snapshot_blob(blob: object) -> object:
    """Decode payloads produced by :func:`encode_snapshot_blob`."""

    if isinstance(blob, str) and blob.startswith(JSON_BYTES_PREFIX):
        return bytes.fromhex(blob[len(JSON_BYTES_PREFIX) :])
    if isinstance(blob, list):
        return [decode_snapshot_blob(item) for item in blob]
    if isinstance(blob, dict):
        decoded: MutableMapping[object, object] = {}
        for key, value in blob.items():
            decoded_key = decode_snapshot_blob(key) if isinstance(key, str) else key
            decoded[decoded_key] = decode_snapshot_blob(value)
        return dict(decoded)
    return blob


def ensure_bytes(value: Union[str, bytes, bytearray, memoryview, Iterable[int]]) -> bytes:
    """Coerce the provided value into a ``bytes`` instance."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return bytes(value.tobytes())
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(int(part) & 0xFF for part in value)


__all__ = [
    "ARRAY_HEADER_STRUCT",
    "ARRAY_MAGIC",
    "DEFAULT_STATE_FILENAMES",
    "DTYPE_CODES",
    "JSON_BYTES_PREFIX",
    "MANIFEST_MAGIC",
    "MANIFEST_VERSION",
    "PICKLE_SUFFIXES",
    "SeraArrayError",
    "SeraArrayHeader",
    "SeraSnapshotFormatError",
    "decode_snapshot_blob",
    "encode_snapshot_blob",
    "ensure_bytes",
    "load_array_file",
    "parse_array_header",
]
