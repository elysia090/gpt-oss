import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
safetensors_numpy = pytest.importorskip("safetensors.numpy")
save_file = safetensors_numpy.save_file

try:
    from gpt_oss.tools import sera_transfer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if "pip install numpy" in str(exc):
        pytest.skip("Real numpy is required for Sera transfer tests", allow_module_level=True)
    raise


def _load_sera_runtime():
    root = Path(__file__).resolve().parents[2]
    sera_path = root / "src" / "gpt_oss" / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_transfer_loader", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_sera_runtime = _load_sera_runtime()
Sera = _sera_runtime.Sera
_validate_transfer_arrays = _sera_runtime._validate_transfer_arrays
_TRANSFER_ARRAY_STRUCT = _sera_runtime._TRANSFER_ARRAY_STRUCT
_TRANSFER_ARRAY_MAGIC = _sera_runtime._TRANSFER_ARRAY_MAGIC
_crc32c = _sera_runtime._crc32c


def _create_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _create_vector(length: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def _create_checkpoint(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)

    config = {
        "d_model": 4,
        "n_heads": 2,
        "head_dim": 2,
        "vocab_size": 8,
        "tau": 0.5,
    }
    layers = []
    rng = random.Random(42)
    tensors: dict[str, list] = {}
    for idx in range(2):
        prefix = f"layers.{idx}"
        layer = {
            "name": f"layer{idx}",
            "W_K": f"{prefix}.attn.k.weight",
            "W_O": f"{prefix}.attn.o.weight",
            "FFN_W1": f"{prefix}.ffn.w1.weight",
            "FFN_W2": f"{prefix}.ffn.w2.weight",
            "FFN_B1": f"{prefix}.ffn.w1.bias",
            "FFN_B2": f"{prefix}.ffn.w2.bias",
        }
        layers.append(layer)
        tensors[layer["W_K"]] = _create_matrix(4, 4, rng)
        tensors[layer["W_O"]] = _create_matrix(4, 4, rng)
        tensors[layer["FFN_W1"]] = _create_matrix(8, 4, rng)
        tensors[layer["FFN_W2"]] = _create_matrix(4, 8, rng)
        tensors[layer["FFN_B1"]] = _create_vector(8, rng)
        tensors[layer["FFN_B2"]] = _create_vector(4, rng)

    tensors["tok_embeddings.weight"] = _create_matrix(config["vocab_size"], config["d_model"], rng)
    config["layers"] = layers
    (source / "config.json").write_text(json.dumps(config))
    array_tensors = {name: np.array(value, dtype=np.float32) for name, value in tensors.items()}
    save_file(array_tensors, source / "model.safetensors")
    return source


def test_transfer_loader_restores_model(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"
    summary = sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    assert summary.manifest_path.exists()

    model, metadata = Sera.transfer(output)
    assert isinstance(model, Sera)
    assert metadata["manifest_path"] == summary.manifest_path
    assert metadata["state_path"].exists()
    assert metadata["state_path"].suffix == ".json"
    arrays = metadata["arrays"]
    assert "tokenizer_fst" in arrays
    assert arrays["tokenizer_fst"]["byte_len"] >= 0


def test_transfer_loader_rejects_pickle_without_opt_in(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    # Remove safe formats so only the pickle snapshot remains.
    json_path = output / "sera_state.json"
    if json_path.exists():
        json_path.unlink()
    msgpack_path = output / "sera_state.msgpack"
    if msgpack_path.exists():
        msgpack_path.unlink()

    with pytest.raises(RuntimeError, match="allow_pickle=True"):
        Sera.transfer(output)


def test_transfer_loader_allows_pickle_with_opt_in(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    json_path = output / "sera_state.json"
    if json_path.exists():
        json_path.unlink()
    msgpack_path = output / "sera_state.msgpack"
    if msgpack_path.exists():
        msgpack_path.unlink()

    model, metadata = Sera.transfer(output, allow_pickle=True)
    assert isinstance(model, Sera)
    assert metadata["state_path"].suffix == ".pkl"


def test_transfer_loader_fallback_without_runtime_snapshot(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    json_path = output / "sera_state.json"
    blob = json.loads(json_path.read_text())
    blob.pop("sera_snapshot", None)
    json_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")

    model, _ = Sera.transfer(output)
    assert isinstance(model, Sera)

    config_blob = blob["model_config"]
    metadata_blob = blob["metadata"]
    assert model.config.attention.dim == config_blob["d_model"]
    assert model.config.attention.features == metadata_blob["attention"]["features"]
    assert model.config.attention.value_dim == metadata_blob["overlays"]["cols"]
    assert model.config.tokenizer.max_piece_length == metadata_blob["tokenizer"]["max_piece_length"]
    assert model.config.linear.capacity >= len(metadata_blob["linear"]["keys"])

    step_result = model.step()
    assert "y_out" in step_result


def _make_transfer_array_header(
    byte_len: int,
    crc32c: int,
    sha_low64: int,
    *,
    flags: int = 0x1,
    reserved: int = 0,
) -> bytes:
    dims = (byte_len, 0, 0, 0, 0)
    return _TRANSFER_ARRAY_STRUCT.pack(
        _TRANSFER_ARRAY_MAGIC,
        6,  # u8
        1,
        *dims,
        byte_len,
        crc32c,
        sha_low64,
        flags,
        reserved,
    )


def _generate_payload(size: int) -> bytes:
    return bytes((idx % 251 for idx in range(size)))


def test_validate_transfer_arrays_streams_large_payload(tmp_path: Path) -> None:
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    size = 2 * 1024 * 1024 + 17
    payload = _generate_payload(size)
    digest = hashlib.sha256(payload).digest()
    header = _make_transfer_array_header(
        size,
        _crc32c(payload),
        int.from_bytes(digest[-8:], "little"),
    )
    array_path = arrays_dir / "large.bin"
    array_path.write_bytes(header + payload)
    arrays = _validate_transfer_arrays(
        arrays_dir,
        {"large": {"sha256": digest.hex()}},
    )
    assert arrays["large"]["byte_len"] == size
    assert arrays["large"]["flags"] == 0x1


@pytest.mark.parametrize(
    "modifier, message",
    [
        ("truncate", "Array payload length mismatch"),
        ("crc", "CRC32C mismatch"),
        ("sha_low", "sha256_low64 mismatch"),
        ("sha_meta", "failed SHA-256 validation"),
    ],
)
def test_validate_transfer_arrays_streaming_detects_corruption(
    tmp_path: Path, modifier: str, message: str
) -> None:
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    size = 2 * 1024 * 1024 + 17
    payload = _generate_payload(size)
    digest = hashlib.sha256(payload).digest()
    crc_value = _crc32c(payload)
    sha_low64 = int.from_bytes(digest[-8:], "little")
    array_path = arrays_dir / f"large_{modifier}.bin"

    if modifier == "truncate":
        header = _make_transfer_array_header(size, crc_value, sha_low64)
        array_path.write_bytes(header + payload[:-1])
        record = {"sha256": digest.hex()}
    elif modifier == "crc":
        bad_crc = (crc_value + 1) & 0xFFFFFFFF
        header = _make_transfer_array_header(size, bad_crc, sha_low64)
        array_path.write_bytes(header + payload)
        record = {"sha256": digest.hex()}
    elif modifier == "sha_low":
        bad_sha_low = (sha_low64 + 1) & 0xFFFFFFFFFFFFFFFF
        header = _make_transfer_array_header(size, crc_value, bad_sha_low)
        array_path.write_bytes(header + payload)
        record = {"sha256": digest.hex()}
    else:
        header = _make_transfer_array_header(size, crc_value, sha_low64)
        array_path.write_bytes(header + payload)
        record = {"sha256": "0" * 64}

    with pytest.raises(ValueError, match=message):
        _validate_transfer_arrays(arrays_dir, {f"large_{modifier}": record})


@pytest.mark.parametrize(
    "flags,reserved,message",
    [
        (0x0, 0, "row-major layout"),
        (0x8, 0, "unsupported flag bits"),
        (0x1, 1, "reserved header field must be zero"),
    ],
)
def test_validate_transfer_arrays_rejects_invalid_header_flags(
    tmp_path: Path, flags: int, reserved: int, message: str
) -> None:
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    payload = _generate_payload(64)
    digest = hashlib.sha256(payload).digest()
    header = _make_transfer_array_header(
        len(payload),
        _crc32c(payload),
        int.from_bytes(digest[-8:], "little"),
        flags=flags,
        reserved=reserved,
    )
    array_path = arrays_dir / "bad_flags.bin"
    array_path.write_bytes(header + payload)

    with pytest.raises(ValueError, match=message):
        _validate_transfer_arrays(arrays_dir, {"bad_flags": {"sha256": digest.hex()}})
