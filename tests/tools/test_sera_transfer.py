from __future__ import annotations

import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetensors.numpy import save_file

from gpt_oss.tools import sera_transfer


def _create_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _create_vector(length: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def _create_checkpoint(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()

    config = {
        "d_model": 4,
        "n_heads": 2,
        "head_dim": 2,
        "vocab_size": 8,
        "tau": 0.5,
        "rope_theta": 10000.0,
        "layers": [
            {
                "name": "layer0",
                "W_K": "layers.0.attn.k.weight",
                "W_O": "layers.0.attn.o.weight",
                "FFN_W1": "layers.0.ffn.w1.weight",
                "FFN_W2": "layers.0.ffn.w2.weight",
                "FFN_B1": "layers.0.ffn.w1.bias",
                "FFN_B2": "layers.0.ffn.w2.bias",
            }
        ],
    }
    (source / "config.json").write_text(json.dumps(config))

    rng = random.Random(0)
    tensors = {
        "layers.0.attn.k.weight": _create_matrix(4, 4, rng),
        "layers.0.attn.o.weight": _create_matrix(4, 4, rng),
        "layers.0.ffn.w1.weight": _create_matrix(8, 4, rng),
        "layers.0.ffn.w2.weight": _create_matrix(4, 8, rng),
        "layers.0.ffn.w1.bias": _create_vector(8, rng),
        "layers.0.ffn.w2.bias": _create_vector(4, rng),
    }
    save_file(tensors, source / "model.safetensors")
    return source


def _read_header(path: Path) -> sera_transfer.ArrayHeader:
    raw = path.read_bytes()[:sera_transfer.ArrayHeader.HEADER_STRUCT.size]
    values = sera_transfer.ArrayHeader.HEADER_STRUCT.unpack(raw)
    return sera_transfer.ArrayHeader(
        magic=values[0],
        dtype_code=values[1],
        rank=values[2],
        dims=values[3:8],
        byte_len=values[8],
        crc32c=values[9],
        sha256_low64=values[10],
        flags=values[11],
        reserved=values[12],
    )


def _payload(path: Path) -> bytes:
    data = path.read_bytes()
    return data[sera_transfer.ArrayHeader.HEADER_STRUCT.size :]


def test_cli_round_trip(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"

    cmd = [
        sys.executable,
        "-m",
        "gpt_oss.tools.sera_transfer",
        "--source",
        str(source),
        "--output",
        str(output),
        "--r",
        "4",
        "--rv",
        "2",
        "--topL",
        "2",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    subprocess.run(cmd, check=True, env=env)

    prf = output / "arrays" / "prf_W.bin"
    header = _read_header(prf)
    assert header.magic == sera_transfer.MAGIC_SERA_ARRAY
    assert header.dtype_code == 2
    payload = _payload(prf)
    assert sera_transfer.crc32c(payload) == header.crc32c
    assert sera_transfer.sha256_low64(payload) == header.sha256_low64

    manifest = (output / "sera_manifest.bin").read_bytes()
    assert manifest[:4] == struct.pack("<I", 0x5345524D)


def test_deterministic_conversion(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    sera_transfer.convert(source, out_a, r=4, r_v=2, top_l=2)
    sera_transfer.convert(source, out_b, r=4, r_v=2, top_l=2)

    files_a = sorted((out_a / "arrays").glob("*.bin"))
    files_b = sorted((out_b / "arrays").glob("*.bin"))
    assert [f.name for f in files_a] == [f.name for f in files_b]

    for file_a, file_b in zip(files_a, files_b):
        assert file_a.read_bytes() == file_b.read_bytes()
