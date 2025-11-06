from __future__ import annotations

import hashlib
import json
import os
import pickle
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
    }
    layers = []
    rng = random.Random(0)
    tensors = {}
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

    snapshot = pickle.loads((output / "sera_state.pkl").read_bytes())
    assert "sera_snapshot" in snapshot
    snapshot_cfg = snapshot["sera_snapshot"]["config"]
    assert snapshot_cfg["tokenizer"]["max_piece_length"] == 4
    assert snapshot["metadata"]["attention"]["features"] == 4


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

    snap_a = pickle.loads((out_a / "sera_state.pkl").read_bytes())
    snap_b = pickle.loads((out_b / "sera_state.pkl").read_bytes())
    assert snap_a["sera_snapshot"]["config"] == snap_b["sera_snapshot"]["config"]
    assert snap_a["metadata"] == snap_b["metadata"]


def test_written_arrays_match_reference(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    cfg = sera_transfer.ModelConfig.from_dict(json.loads((source / "config.json").read_text()))
    tensors = sera_transfer.load_tensors(source / "model.safetensors")
    arrays_dir = output / "arrays"

    tokenizer_data, tokenizer_meta = sera_transfer.tokenizer_arrays(cfg, tensors)
    fst_payload = _payload(arrays_dir / "tokenizer_fst.bin")
    expected_fst = sera_transfer._pack_values(tokenizer_data["tokenizer_fst"], "B")
    assert fst_payload == expected_fst
    assert len(tokenizer_meta["pieces"]) == cfg.vocab_size
    for length in range(1, tokenizer_meta["max_piece_length"] + 1):
        table_payload = _payload(arrays_dir / f"T_{length}.bin")
        expected_table = sera_transfer._pack_values(tokenizer_data[f"T_{length}"], "B")
        assert table_payload == expected_table
        mph_info = tokenizer_meta["mph"][length]
        assert mph_info["table_size"] >= len(mph_info["key_hashes"])

    linear_data, linear_meta = sera_transfer.collapse_ffn(cfg, tensors, top_l=2)
    assert set(linear_meta["keys"]) == {
        (layer_idx << 32) | feature
        for layer_idx in range(len(cfg.layers))
        for feature in range(cfg.d_model)
    }
    assert len(linear_meta["weights"]) == cfg.d_model * len(cfg.layers)
    assert linear_meta["residuals"] > 0
    assert _payload(arrays_dir / "linear_keys.bin") == sera_transfer._pack_values(
        linear_data["linear_keys"], "B"
    )
    assert _payload(arrays_dir / "linear_weights.bin") == sera_transfer._pack_values(
        linear_data["linear_weights"], "f"
    )
    assert _payload(arrays_dir / "cuckoo_delta.bin") == sera_transfer._pack_values(
        linear_data["cuckoo_delta"], "B"
    )

    memory_data, memory_meta = sera_transfer.memory_coefficients(cfg)
    assert memory_meta["layers"] == len(cfg.layers)
    assert _payload(arrays_dir / "memory_coeff.bin") == sera_transfer._pack_values(
        memory_data["memory_coeff"], "d"
    )

    bridge_data, bridge_meta = sera_transfer.bridge_records(cfg, tensors, cfg.vocab_size)
    assert bridge_meta["legs"] == 2
    assert len(bridge_meta["in_scales"]) == cfg.vocab_size
    assert _payload(arrays_dir / "bridge_qDin.bin") == sera_transfer._pack_values(
        bridge_data["bridge_qDin"], "h"
    )
    assert _payload(arrays_dir / "bridge_hubs.bin") == sera_transfer._pack_values(
        bridge_data["bridge_hubs"], "B"
    )

    snapshot = pickle.loads((output / "sera_state.pkl").read_bytes())
    artefacts = snapshot["artefacts"]
    assert "tokenizer_fst" in artefacts
    for name, record in artefacts.items():
        payload = _payload(arrays_dir / f"{name}.bin")
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert snapshot["metadata"]["tokenizer"]["max_piece_length"] == 4
    assert snapshot["sera_snapshot"]["config"]["attention"]["features"] == 4


def test_array_checksums_regression(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path)
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    arrays_dir = output / "arrays"
    checksums = {
        path.name: hashlib.sha256(_payload(path)).hexdigest()
        for path in sorted(arrays_dir.glob("*.bin"))
    }

    expected = {
        "R_init.bin": "c1e397da306abc45e5fbe4f03330dab17cceb14a5c46561e3f51f7fc4a1f90e8",
        "T_1.bin": "df5225054cae663ff1c726d803091622ffc765309c9e9b9e573c0bbd08105a37",
        "T_2.bin": "067fece1127c0e37ff0bf1ff566b727f33dd3f5d5ba15431916b61adfebc2e5b",
        "T_3.bin": "581fca69ff42c2b69b83ba1f128513e611ee8238448e0eb920a820e74b78de1a",
        "T_4.bin": "1147d97ec1eae29ec741abdaabfb36c6db64485f9a5438e51a4656f98c6eb3ec",
        "bridge_hubs.bin": "688d4e7d3b1c3204c0e3ac9c06ab96a0a7631d76c1737fc24250bc479e6f20e1",
        "bridge_qDin.bin": "ec0707f8cb38f54984497a0b51793e36473107ba8d294f96ac96164f5df4d1bd",
        "bridge_qDout.bin": "e2cb6dd3cda429569e818f29e4b102d9b5006f30e646d42460440922f1b55e32",
        "cuckoo_delta.bin": "a80189f45efc8626983c04226824ce57d66d8951d0711b27eb68125250a73d01",
        "delaybuf_init.bin": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
        "linear_bias.bin": "1d9e0bc18f8893dee78d668f460c9385b3dd6e469c55c79bce634fd16a74e8ce",
        "linear_keys.bin": "1518b6e567af558a19594c82a37d7fedbeb5dfee23a48713a8733b8a997556a6",
        "linear_mphf.bin": "f956fb7578c0cbdd582777104c27a351021d4f1d4c7bc065727b68f54c7962c0",
        "linear_weights.bin": "4ab874407f08958d85d2c0d0d6dec78366177538e4d0cb1662da81e98ed5a7f3",
        "memory_coeff.bin": "74e0abc5d53491b7328071ae92df330f5caaeec0d9cd183a2c583b2ea8e6b9e3",
        "overlays_DeltaW.bin": "d154e8363c23c577f20ce3d845b8f3e635ffeac9f9ee1c9fed46191b5d8eab62",
        "overlays_H.bin": "cc6c5558ac91886a9a65e73946a9037f3690a6af9889b297546d870d3998c058",
        "overlays_U.bin": "877425be63ca60e31d1e668d384bcde42fe19c2ec2793856143f83b6e7e87aff",
        "peer_scores.bin": "36d9d1d2694dd68a164ffaea6e4891a68bef887228851b1e7341a109a77eb638",
        "prf_W.bin": "fc6134f0ec03aff0ca21bfc7e0a5a2c70afa8c73ab290f7b694fd9f008fe293b",
        "s_init.bin": "4d848697b5664465351e1ba48be3e74ad691e88b1260ccb3e0b73db7118f55c4",
        "tokenizer_fst.bin": "45da3c4be5bdb7c344bb6e2ede224d877de3e51d50e39788e98a437dc7109d2e",
        "whitening_mu.bin": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb",
        "whitening_sig2.bin": "3c6add9ce200ba6e99cb568fd4213c456fc2e4b79d36d82eefad6705493b265a",
    }

    assert checksums == expected
