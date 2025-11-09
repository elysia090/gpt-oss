"""Unit tests for the lightweight helpers shared across the Sera runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_SERA_COMMON_PATH = Path(__file__).resolve().parents[2] / "src" / "gpt_oss" / "inference" / "sera_common.py"
_SERA_COMMON_SPEC = importlib.util.spec_from_file_location("_sera_common", _SERA_COMMON_PATH)
if _SERA_COMMON_SPEC is None or _SERA_COMMON_SPEC.loader is None:  # pragma: no cover - defensive
    raise RuntimeError("Unable to load Sera common module for tests")
sera_common = importlib.util.module_from_spec(_SERA_COMMON_SPEC)
sys.modules[_SERA_COMMON_SPEC.name] = sera_common
_SERA_COMMON_SPEC.loader.exec_module(sera_common)


@pytest.mark.parametrize(
    "dtype_code, expected",
    [
        (2, "f32"),
        (99, "code99"),
    ],
)
def test_parse_array_header_recovers_shape_and_dtype(dtype_code: int, expected: str) -> None:
    values = (
        sera_common.ARRAY_MAGIC,
        dtype_code,
        2,
        4,
        3,
        0,
        0,
        0,
        12,
        0x1234,
        0x5678,
        0,
        0,
    )

    header = sera_common.parse_array_header(values)

    assert header.dtype == expected
    assert header.rank == 2
    assert header.shape == (4, 3)
    assert header.byte_len == 12


def test_parse_array_header_rejects_invalid_magic() -> None:
    values = (
        0x12345678,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    with pytest.raises(sera_common.SeraArrayError):
        sera_common.parse_array_header(values)


@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"bytes", b"bytes"),
        ("text", b"text"),
        (bytearray(b"data"), b"data"),
        (memoryview(b"span"), b"span"),
        ([0, 255, 1], b"\x00\xff\x01"),
    ],
)
def test_ensure_bytes_handles_common_inputs(payload, expected: bytes) -> None:
    assert sera_common.ensure_bytes(payload) == expected


def test_encode_decode_snapshot_blob_roundtrip() -> None:
    payload = {"payload": b"data", "nested": [b"inner", {"tuple": (b"two",)}]}

    encoded = sera_common.encode_snapshot_blob(payload)
    decoded = sera_common.decode_snapshot_blob(encoded)

    assert decoded["payload"] == b"data"
    assert decoded["nested"][0] == b"inner"
    # Tuples are converted to lists to make the structure JSON-compatible.
    assert decoded["nested"][1]["tuple"] == [b"two"]


def test_encode_snapshot_blob_serialises_path_and_tuple() -> None:
    value = ("path", Path("artifact.bin"))

    encoded = sera_common.encode_snapshot_blob(value)

    assert encoded == ["path", "artifact.bin"]
    assert sera_common.decode_snapshot_blob(encoded) == encoded


def _write_array_file(path: Path, payload: bytes) -> None:
    header = sera_common.ARRAY_HEADER_STRUCT.pack(
        sera_common.ARRAY_MAGIC,
        1,
        1,
        len(payload),
        0,
        0,
        0,
        0,
        len(payload),
        0,
        0,
        0,
        0,
    )
    path.write_bytes(header + payload)


def test_load_array_file_validates_payload(tmp_path: Path) -> None:
    array_path = tmp_path / "array.bin"
    _write_array_file(array_path, b"payload")

    header, payload = sera_common.load_array_file(array_path)

    assert header.byte_len == len(payload) == len(b"payload")


def test_load_array_file_rejects_truncated_payload(tmp_path: Path) -> None:
    array_path = tmp_path / "array.bin"
    _write_array_file(array_path, b"payload")
    # Remove the trailing byte to break the length contract.
    data = array_path.read_bytes()[:-1]
    array_path.write_bytes(data)

    with pytest.raises(sera_common.SeraArrayError):
        sera_common.load_array_file(array_path)
