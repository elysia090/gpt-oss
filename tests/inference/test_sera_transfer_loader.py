import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from gpt_oss._stubs.safetensors.numpy import save_file

from gpt_oss.tools import sera_transfer


def _load_sera_runtime():
    root = Path(__file__).resolve().parents[2]
    sera_path = root / "src" / "gpt_oss" / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_transfer_loader", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sera


Sera = _load_sera_runtime()


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
