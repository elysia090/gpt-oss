from __future__ import annotations

import importlib
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import struct
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

safetensors_numpy = pytest.importorskip("safetensors.numpy")
save_file = safetensors_numpy.save_file

SPEC_DTYPE_CODES = {
    "f64": 1,
    "f32": 2,
    "i32": 3,
    "i16": 4,
    "i8": 5,
    "u8": 6,
    "q8_8": 7,
    "q4_12": 8,
    "bf16": 9,
}

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
try:
    from gpt_oss.tools import sera_transfer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if "pip install numpy" in str(exc):
        pytest.skip("Real numpy is required for Sera transfer tests", allow_module_level=True)
    raise

assert "src" in Path(sera_transfer.__file__).parts


def _create_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _create_vector(length: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def _as_numpy_tensors(tensors: dict[str, list]) -> dict[str, np.ndarray]:
    return {name: np.array(value, dtype=np.float32) for name, value in tensors.items()}


def _memory_worker(checkpoint: str, queue) -> None:
    import importlib
    import resource
    from pathlib import Path

    worker_module = importlib.import_module("gpt_oss.tools.sera_transfer")
    path = Path(checkpoint)
    index, _ = worker_module._read_safetensors_index(path)
    max_layer_bytes = max(
        entry["data_offsets"][1] - entry["data_offsets"][0]
        for entry in index.values()
    )

    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tensors = worker_module.load_tensors(path)
    del tensors
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    queue.put({
        "peak": max(0, peak_rss - start_rss),
        "max_layer": max_layer_bytes,
    })


def _bf16_payload(values: np.ndarray) -> bytes:
    data = np.asarray(values, dtype=np.float32)
    uint32 = data.view(np.uint32)
    bf16 = (uint32 >> 16).astype(np.uint16)
    return bf16.tobytes()


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header: dict[str, dict[str, object]] = {}
    offset = 0
    payloads: list[bytes] = []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payloads.append(data)
        offset += len(data)

    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        for chunk in payloads:
            handle.write(chunk)


def _sample_checkpoint_contents() -> tuple[dict, dict[str, np.ndarray]]:
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
    return config, _as_numpy_tensors(tensors)


def _create_checkpoint(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)

    config, tensors = _sample_checkpoint_contents()
    (source / "config.json").write_text(json.dumps(config))
    save_file(_as_numpy_tensors(tensors), source / "model.safetensors")
    return source


def _create_openai_checkpoint(root: Path) -> Path:
    source = root / "openai"
    source.mkdir(parents=True, exist_ok=True)

    config = {
        "d_model": 4,
        "n_heads": 2,
        "head_dim": 2,
        "vocab_size": 8,
        "tau": 0.5,
        "rope_theta": 10000.0,
    }
    (source / "config.json").write_text(json.dumps(config))

    rng = random.Random(0)
    tensors: dict[str, list] = {}
    for idx in range(2):
        prefix = f"model.layers.{idx}"
        attn = f"{prefix}.attention"
        mlp = f"{prefix}.mlp"
        tensors[f"{attn}.q_proj.weight"] = _create_matrix(4, 4, rng)
        tensors[f"{attn}.k_proj.weight"] = _create_matrix(4, 4, rng)
        tensors[f"{attn}.v_proj.weight"] = _create_matrix(4, 4, rng)
        tensors[f"{attn}.o_proj.weight"] = _create_matrix(4, 4, rng)
        tensors[f"{mlp}.gate_proj.weight"] = _create_matrix(8, 4, rng)
        tensors[f"{mlp}.down_proj.weight"] = _create_matrix(4, 8, rng)
        tensors[f"{mlp}.gate_proj.bias"] = _create_vector(8, rng)
        tensors[f"{mlp}.down_proj.bias"] = _create_vector(4, rng)

    tensors["tok_embeddings.weight"] = _create_matrix(config["vocab_size"], config["d_model"], rng)
    save_file(_as_numpy_tensors(tensors), source / "model.safetensors")
    return source


def _tensor_shape(tensor) -> list[int]:
    if isinstance(tensor, list) and tensor:
        return [len(tensor)] + _tensor_shape(tensor[0])
    if isinstance(tensor, list):
        return [0]
    return []


def _flatten_tensor(tensor) -> list[float]:
    if isinstance(tensor, list):
        result: list[float] = []
        for item in tensor:
            result.extend(_flatten_tensor(item))
        return result
    return [float(tensor)]


def _mxfp4_bytes(rows: int, bytes_per_row: int) -> list[list[int]]:
    data: list[list[int]] = []
    for r in range(rows):
        row: list[int] = []
        for c in range(bytes_per_row):
            lo = (r + c) % 16
            hi = (r + c + 1) % 16
            row.append(lo | (hi << 4))
        data.append(row)
    return data


@pytest.fixture
def gpt_oss_mxfp4_layout() -> tuple[dict[str, object], dict[str, list]]:
    config = {
        "d_model": 4,
        "n_heads": 2,
        "head_dim": 2,
        "vocab_size": 8,
        "tau": 0.5,
    }

    attn_qkv = [[float(r * 4 + c) for c in range(4)] for r in range(12)]
    attn_out = [[float((r + 1) * (c + 1)) for c in range(4)] for r in range(4)]
    mlp1_blocks = _mxfp4_bytes(8, 2)
    mlp2_blocks = _mxfp4_bytes(4, 4)
    mlp1_scales = [127 + ((idx % 3) - 1) for idx in range(8)]
    mlp2_scales = [127 + ((idx % 2) - 1) for idx in range(4)]
    mlp1_bias = [float(idx) for idx in range(8)]
    mlp2_bias = [float(idx) for idx in range(4)]

    rng = random.Random(123)
    gating_w1 = _create_matrix(8, 4, rng)
    gating_w2 = _create_matrix(4, 8, rng)
    gating_b1 = _create_vector(8, rng)
    gating_b2 = _create_vector(4, rng)

    tensors: dict[str, list] = {
        "block.0.attn.qkv.weight": attn_qkv,
        "block.0.attn.out.weight": attn_out,
        "block.0.mlp.mlp1_weight.blocks": mlp1_blocks,
        "block.0.mlp.mlp1_weight.scales": mlp1_scales,
        "block.0.mlp.mlp2_weight.blocks": mlp2_blocks,
        "block.0.mlp.mlp2_weight.scales": mlp2_scales,
        "block.0.mlp.mlp1_bias": mlp1_bias,
        "block.0.mlp.mlp2_bias": mlp2_bias,
        "block.0.mlp.gate_proj.weight": gating_w1,
        "block.0.mlp.down_proj.weight": gating_w2,
        "block.0.mlp.gate_proj.bias": gating_b1,
        "block.0.mlp.down_proj.bias": gating_b2,
    }

    return config, tensors


def test_load_tensors_requires_safetensors(tmp_path, monkeypatch):
    module = importlib.reload(sera_transfer)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")

    monkeypatch.setattr(
        module,
        "_read_safetensors_index",
        lambda path: ({"foo": {"dtype": "F32", "shape": (1,), "data_offsets": (0, 4)}}, 0),
    )

    def missing_safe_open(*args, **kwargs):  # noqa: ARG001
        raise ModuleNotFoundError(module._SAFETENSORS_MISSING_MSG)

    monkeypatch.setattr(module, "safe_open", missing_safe_open)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        module.load_tensors(checkpoint)

    assert "pip install safetensors" in str(excinfo.value)


def test_load_tensors_passes_numpy_framework(tmp_path, monkeypatch):
    module = importlib.reload(sera_transfer)

    calls: list[str] = []

    class _FakeSafeFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return ["foo"]

        def get_tensor(self, key):
            assert key == "foo"
            return [[1.0, 2.0]]

    def recording_safe_open(path, framework="numpy", **kwargs):  # noqa: ARG001
        calls.append(framework)
        return _FakeSafeFile()

    monkeypatch.setattr(
        module,
        "_read_safetensors_index",
        lambda path: ({"foo": {"dtype": "F32", "shape": (1,), "data_offsets": (0, 4)}}, 0),
    )
    monkeypatch.setattr(module, "safe_open", recording_safe_open)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")

    tensors = module.load_tensors(checkpoint)

    assert tensors["foo"] == [[1.0, 2.0]]
    assert calls == ["numpy"]


def test_load_tensors_keyword_only_safe_open_backward_compat(tmp_path, monkeypatch):
    module = importlib.reload(sera_transfer)

    assert module.safe_open is not None

    class _FakeTensor:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def __getitem__(self, index):
            return self._data[index]

        @property
        def shape(self):
            dims: list[int] = []
            current = self._data
            while isinstance(current, list):
                dims.append(len(current))
                if not current:
                    break
                current = current[0]
            return tuple(dims)

    class _FakeSafeFile:
        def __init__(self, mapping):
            self._mapping = mapping

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._mapping.keys())

        def get_tensor(self, key):
            return _FakeTensor(self._mapping[key])

    calls: list[tuple[str, str]] = []
    payload = {"foo": [[1.0, 2.0]]}

    def keyword_only_safe_open(path, *args, framework, device=None, **kwargs):  # noqa: ARG001
        if args:
            raise TypeError("safe_open() takes 1 positional argument but 2 were given")
        calls.append(("keyword", framework))
        return _FakeSafeFile(payload)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")

    monkeypatch.setattr(
        module,
        "_read_safetensors_index",
        lambda path: ({"foo": {"dtype": "F32", "shape": (1,), "data_offsets": (0, 4)}}, 0),
    )
    monkeypatch.setattr(module, "safe_open", keyword_only_safe_open)

    tensors = module.load_tensors(checkpoint)

    tensor = tensors["foo"]
    assert isinstance(tensor, _FakeTensor)
    assert tensor.tolist() == payload["foo"]
    assert calls == [("keyword", "numpy")]


def test_load_tensors_preserves_bf16_payload(tmp_path):
    module = importlib.reload(sera_transfer)

    path = tmp_path / "model.safetensors"
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    _write_safetensors(
        path,
        {"layer.weight": ("BF16", values.shape, _bf16_payload(values))},
    )

    tensors = module.load_tensors(path)

    bf_tensor = tensors["layer.weight"]
    bf_cls = getattr(module, "_Bfloat16Tensor")
    assert isinstance(bf_tensor, bf_cls)
    assert bf_tensor.shape == (4, 8)
    assert bf_tensor.nbytes == values.size * 2

    decoded = bf_tensor.to_float32()
    np.testing.assert_allclose(decoded, values)


def test_load_tensors_memory_budget_for_20b(tmp_path):
    module = importlib.reload(sera_transfer)

    path = tmp_path / "gpt-oss-20b.safetensors"
    qkv = np.zeros((5120, 2880), dtype=np.float32)
    o_proj = np.zeros((4096, 1024), dtype=np.float32)
    embed = np.zeros((32000, 128), dtype=np.float32)
    _write_safetensors(
        path,
        {
            "model.layers.0.attention.qkv.weight": ("BF16", qkv.shape, _bf16_payload(qkv)),
            "model.layers.0.attention.o_proj.weight": ("F32", o_proj.shape, o_proj.tobytes()),
            "model.embed_tokens.weight": ("F32", embed.shape, embed.tobytes()),
        },
    )

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_memory_worker, args=(str(path), queue))
    proc.start()
    proc.join()
    assert proc.exitcode == 0
    result = queue.get()
    queue.close()
    queue.join_thread()

    assert result["peak"] <= result["max_layer"] + 512 * 1024 * 1024


def test_model_config_accepts_hf_config_fields() -> None:
    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 8,
        "rope_theta": 10000.0,
    }

    rng = random.Random(0)
    tensors = {
        "model.layers.0.attention.q_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.k_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.v_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(8, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 8, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(8, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }
    tensors = _as_numpy_tensors(tensors)

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert cfg.d_model == 4
    assert cfg.n_heads == 2
    assert cfg.head_dim == 2
    assert len(cfg.layers) == 1
    layer = cfg.layers[0]
    assert layer.w_q.endswith("attention.q_proj.weight")
    assert layer.w_k.endswith("attention.k_proj.weight")
    assert layer.w_v.endswith("attention.v_proj.weight")
    assert layer.w_o.endswith("attention.o_proj.weight")
    assert layer.w1.endswith("mlp.gate_proj.weight")
    assert layer.w2.endswith("mlp.down_proj.weight")
    assert layer.b1.endswith("mlp.gate_proj.bias")
    assert layer.b2.endswith("mlp.down_proj.bias")


def test_materialize_qkv_layers_reports_inference_failure() -> None:
    layer = sera_transfer.LayerConfig(
        name="layer0",
        w_q="layer0.attn.qkv__sera_qkv__W_Q",
        w_k="layer0.attn.qkv__sera_qkv__W_K",
        w_v="layer0.attn.qkv__sera_qkv__W_V",
        w_o="layer0.attn.o.weight",
        w1="layer0.ffn.w1.weight",
        w2="layer0.ffn.w2.weight",
        b1="layer0.ffn.w1.bias",
        b2="layer0.ffn.w2.bias",
    )
    tensors: dict[str, np.ndarray] = {
        "layer0.attn.qkv": np.zeros((4,), dtype=np.float32),
    }

    with pytest.raises(ValueError) as excinfo:
        sera_transfer.ModelConfig._materialize_qkv_layers(
            [layer],
            tensors,
            d_model=4,
            n_heads=2,
            head_dim=2,
            num_key_value_heads=None,
        )

    message = str(excinfo.value)
    assert "layer0.attn.qkv" in message
    assert "(4,)" in message
    assert "n_heads=2" in message
    assert "num_key_value_heads=unspecified" in message


def test_model_config_infers_layers_from_generic_suffixes() -> None:
    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }

    rng = random.Random(1)
    tensors = {
        "layer00.q_proj.weight": _create_matrix(4, 4, rng),
        "layer00.k_proj.weight": _create_matrix(4, 4, rng),
        "layer00.v_proj.weight": _create_matrix(4, 4, rng),
        "layer00.o_proj.weight": _create_matrix(4, 4, rng),
        "layer00.up_proj.weight": _create_matrix(8, 4, rng),
        "layer00.down_proj.weight": _create_matrix(4, 8, rng),
        "layer00.up_proj.bias": _create_vector(8, rng),
        "layer00.down_proj.bias": _create_vector(4, rng),
    }
    tensors = _as_numpy_tensors(tensors)
    tensors = _as_numpy_tensors(tensors)

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert len(cfg.layers) == 1
    layer = cfg.layers[0]
    assert layer.w_q.endswith("q_proj.weight")
    assert layer.w_k.endswith("k_proj.weight")
    assert layer.w_v.endswith("v_proj.weight")
    assert layer.w_o.endswith("o_proj.weight")
    assert layer.w1.endswith("up_proj.weight")
    assert layer.w2.endswith("down_proj.weight")
    assert layer.b1.endswith("up_proj.bias")
    assert layer.b2.endswith("down_proj.bias")


def test_model_config_splits_fused_qkv_weights() -> None:
    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }

    rng = random.Random(2)
    qkv_weight = np.array(_create_matrix(12, 4, rng), dtype=np.float32)
    tensors = {
        "model.layers.0.attention.qkv.weight": qkv_weight,
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(8, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 8, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(8, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }
    tensors = _as_numpy_tensors(tensors)

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert len(cfg.layers) == 1
    layer = cfg.layers[0]
    assert layer.w_q.startswith("model.layers.0.attention.qkv.weight")
    assert layer.w_k.startswith("model.layers.0.attention.qkv.weight")
    assert layer.w_v.startswith("model.layers.0.attention.qkv.weight")
    assert layer.w_q != layer.w_k != layer.w_v

    w_q = tensors[layer.w_q]
    w_k = tensors[layer.w_k]
    w_v = tensors[layer.w_v]
    assert np.allclose(w_q, qkv_weight[:4])
    assert np.allclose(w_k, qkv_weight[4:8])
    assert np.allclose(w_v, qkv_weight[8:])


def test_model_config_infers_head_dim_from_qkv_tensor() -> None:
    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
    }

    rng = random.Random(3)
    qkv_weight = np.array(_create_matrix(16, 4, rng), dtype=np.float32)
    tensors = {
        "model.layers.0.attention.qkv.weight": qkv_weight,
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(8, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 8, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(8, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }
    tensors = _as_numpy_tensors(tensors)

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert cfg.head_dim == 4
    layer = cfg.layers[0]
    assert np.allclose(tensors[layer.w_q], qkv_weight[:8])
    assert np.allclose(tensors[layer.w_k], qkv_weight[8:12])
    assert np.allclose(tensors[layer.w_v], qkv_weight[12:])


def test_model_config_infers_kv_heads_from_gpt_oss_20b_shape() -> None:
    class FakeMatrix:
        def __init__(self, rows: int, cols: int) -> None:
            self.shape = (rows, cols)
            self._rows = rows
            self._row_template = array("f", [0.0] * cols)

        def __len__(self) -> int:
            return self._rows

        def __iter__(self):
            for _ in range(self._rows):
                yield self._row_template[:]

        def __getitem__(self, item):
            if isinstance(item, slice):
                start, stop, step = item.indices(self._rows)
                return [self._row_template[:] for _ in range(start, stop, step)]
            index = item if item >= 0 else self._rows + item
            if index < 0 or index >= self._rows:
                raise IndexError(index)
            return self._row_template[:]

    config = {
        "hidden_size": 2880,
        "num_attention_heads": 64,
        "head_dim": 64,
    }

    qkv_weight = FakeMatrix(5120, 2880)
    tensors = {
        "model.layers.0.attention.qkv.weight": qkv_weight,
        "model.layers.0.attention.o_proj.weight": FakeMatrix(4096, 1),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(4, 4, random.Random(4)),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 4, random.Random(5)),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(4, random.Random(6)),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, random.Random(7)),
    }

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 64

    layer = cfg.layers[0]
    assert layer.w_q.startswith("model.layers.0.attention.qkv.weight")
    assert layer.w_k.startswith("model.layers.0.attention.qkv.weight")
    assert layer.w_v.startswith("model.layers.0.attention.qkv.weight")

    q_rows = tensors[layer.w_q]
    k_rows = tensors[layer.w_k]
    v_rows = tensors[layer.w_v]

    assert len(q_rows) == 4096
    assert len(k_rows) == 512
    assert len(v_rows) == 512

    assert len(q_rows[0]) == 2880
    assert len(k_rows[0]) == 2880
    assert len(v_rows[0]) == 2880

def test_model_config_records_optional_fields() -> None:
    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "num_experts": 32,
        "experts_per_token": 4,
        "intermediate_size": 16,
        "swiglu_limit": 7.0,
        "sliding_window": 128,
        "initial_context_length": 4096,
        "rope_theta": 150000.0,
        "rope_scaling_factor": 32.0,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "tau": 0.75,
    }

    rng = random.Random(3)
    tensors = {
        "model.layers.0.attention.q_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.k_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.v_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(16, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 16, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(16, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert cfg.num_hidden_layers == 1
    assert cfg.num_experts == 32
    assert cfg.experts_per_token == 4
    assert cfg.intermediate_size == 16
    assert cfg.swiglu_limit == pytest.approx(7.0)
    assert cfg.sliding_window == 128
    assert cfg.initial_context_length == 4096
    assert cfg.rope_scaling_factor == pytest.approx(32.0)
    assert cfg.rope_ntk_alpha == pytest.approx(1.0)
    assert cfg.rope_ntk_beta == pytest.approx(32.0)

    serialised = sera_transfer._config_to_dict(cfg)
    assert serialised["num_experts"] == 32
    assert serialised["experts_per_token"] == 4
    assert serialised["rope_scaling_factor"] == pytest.approx(32.0)
    assert serialised["rope_ntk_beta"] == pytest.approx(32.0)


def test_model_config_missing_dim_fields_raises_helpful_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        sera_transfer.ModelConfig.from_dict({"foo": "bar"})

    message = str(excinfo.value)
    assert "d_model" in message
    assert "docs/operations/sera-transfer.md" in message


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
    header = _read_header(path)
    data = path.read_bytes()
    header_size = sera_transfer.ArrayHeader.HEADER_STRUCT.size
    alignment = 0
    if header.flags & sera_transfer.ARRAY_FLAG_ALIGNED_64B:
        alignment = (-header_size) % 64
    start = header_size + alignment
    end = start + header.byte_len
    return data[start:end]


@pytest.mark.parametrize(
    "dtype,data,expected_rank,expected_dims",
    [
        ("f64", 1.0, 0, (1, 1, 1, 1, 1)),
        ("f32", [1.0, -1.0], 1, (2, 1, 1, 1, 1)),
        ("i32", [[1, 2], [3, 4]], 2, (2, 2, 1, 1, 1)),
        ("i16", [1, -2, 3], 1, (3, 1, 1, 1, 1)),
        ("i8", [-4, 5, -6, 7], 1, (4, 1, 1, 1, 1)),
        ("u8", [0, 1, 2, 3, 4], 1, (5, 1, 1, 1, 1)),
        ("q8_8", [[256, -128], [512, 64]], 2, (2, 2, 1, 1, 1)),
        ("q4_12", [0x1234, 0x0001, 0x00FF], 1, (3, 1, 1, 1, 1)),
        ("bf16", [0.25, -1.5, 0.0, 7.0], 1, (4, 1, 1, 1, 1)),
    ],
)
def test_write_array_header_matches_spec(tmp_path: Path, dtype, data, expected_rank, expected_dims):
    target = tmp_path / f"{dtype}.bin"
    payload = sera_transfer.write_array(target, data, dtype)
    raw = target.read_bytes()
    header_size = sera_transfer.ArrayHeader.HEADER_STRUCT.size
    header_bytes = raw[:header_size]
    header_fields = sera_transfer.ArrayHeader.HEADER_STRUCT.unpack(header_bytes)

    assert header_fields[0] == sera_transfer.MAGIC_SERA_ARRAY
    assert header_fields[1] == SPEC_DTYPE_CODES[dtype]
    assert header_fields[2] == expected_rank
    assert header_fields[3:8] == expected_dims
    assert header_fields[8] == len(payload)
    assert header_fields[11] == (
        sera_transfer.ARRAY_FLAG_ROW_MAJOR
        | sera_transfer.ARRAY_FLAG_ALIGNED_64B
        | sera_transfer.ARRAY_FLAG_FTZ
    )

    alignment = (-header_size) % 64
    payload_offset = header_size + alignment
    assert payload_offset % 64 == 0
    if alignment:
        assert raw[header_size:payload_offset] == b"\x00" * alignment

    assert raw[payload_offset : payload_offset + len(payload)] == payload
    assert raw[payload_offset + len(payload) :] == b""

def test_model_config_infers_layers_from_openai_layout(tmp_path: Path) -> None:
    source = _create_openai_checkpoint(tmp_path / "openai_infer")
    config_data = json.loads((source / "config.json").read_text())
    tensors = sera_transfer.load_tensors(source / "model.safetensors")

    cfg = sera_transfer.ModelConfig.from_dict(config_data, tensors=tensors)
    assert len(cfg.layers) == 2

    first = cfg.layers[0]
    assert first.w_q.endswith("attention.q_proj.weight")
    assert first.w_k.endswith("attention.k_proj.weight")
    assert first.w_v.endswith("attention.v_proj.weight")
    assert first.w_o.endswith("attention.o_proj.weight")
    assert first.w1.endswith("mlp.gate_proj.weight")
    assert first.w2.endswith("mlp.down_proj.weight")
    assert first.b1.endswith("mlp.gate_proj.bias")
    assert first.b2.endswith("mlp.down_proj.bias")


def test_model_config_handles_gpt_oss_mxfp4_layout(gpt_oss_mxfp4_layout) -> None:
    config, tensors = gpt_oss_mxfp4_layout

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert len(cfg.layers) == 1
    layer = cfg.layers[0]
    assert layer.w_q.startswith("block.0.attn.qkv.weight")
    assert layer.w_k.startswith("block.0.attn.qkv.weight")
    assert layer.w_v.startswith("block.0.attn.qkv.weight")
    assert layer.w_o.endswith("attn.out.weight")
    assert layer.w1 == "block.0.mlp.mlp1_weight"
    assert layer.w2 == "block.0.mlp.mlp2_weight"
    assert layer.b1 == "block.0.mlp.mlp1_bias"
    assert layer.b2 == "block.0.mlp.mlp2_bias"
    assert layer.w1 != "block.0.mlp.gate_proj.weight"
    assert layer.w2 != "block.0.mlp.down_proj.weight"
    assert layer.b1 != "block.0.mlp.gate_proj.bias"
    assert layer.b2 != "block.0.mlp.down_proj.bias"

    decoded_shape_w1 = _tensor_shape(tensors[layer.w1])
    decoded_shape_w2 = _tensor_shape(tensors[layer.w2])
    assert decoded_shape_w1 == [8, 4]
    assert decoded_shape_w2 == [4, 8]

    expected_w1 = sera_transfer._decode_mxfp4_pair(
        tensors["block.0.mlp.mlp1_weight.blocks"],
        tensors["block.0.mlp.mlp1_weight.scales"],
    )
    expected_w2 = sera_transfer._decode_mxfp4_pair(
        tensors["block.0.mlp.mlp2_weight.blocks"],
        tensors["block.0.mlp.mlp2_weight.scales"],
    )
    assert _flatten_tensor(tensors[layer.w1]) == pytest.approx(
        _flatten_tensor(expected_w1)
    )
    assert _flatten_tensor(tensors[layer.w2]) == pytest.approx(
        _flatten_tensor(expected_w2)
    )

    w_q = tensors[layer.w_q]
    w_k = tensors[layer.w_k]
    w_v = tensors[layer.w_v]
    attn_qkv = tensors["block.0.attn.qkv.weight"]
    assert w_q == attn_qkv[:4]
    assert w_k == attn_qkv[4:8]
    assert w_v == attn_qkv[8:]

    prf = sera_transfer.compute_prf(cfg, tensors, r=2)
    assert "prf_W" in prf and prf["prf_W"]

    overlays = sera_transfer.compute_overlays(cfg, tensors, r=1, r_v=1)
    assert "overlays_H" in overlays

    linear_data, linear_meta = sera_transfer.collapse_ffn(cfg, tensors, top_l=2)
    assert linear_data["linear_bias"]
    assert linear_meta["bias"]

    bridge_data, bridge_meta = sera_transfer.bridge_records(cfg, tensors, config["vocab_size"])
    assert bridge_data["bridge_hubs"]
    assert len(bridge_meta["layer_seeds"]) == len(cfg.layers)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_create_checkpoint, id="explicit-layout"),
        pytest.param(_create_openai_checkpoint, id="openai-layout"),
    ],
)
def test_cli_round_trip(tmp_path: Path, factory) -> None:
    source = factory(tmp_path / factory.__name__)
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

    snapshot = pickle.loads((output / "sera_state.pkl").read_bytes())
    assert "sera_snapshot" in snapshot
    snapshot_cfg = snapshot["sera_snapshot"]["config"]
    assert snapshot_cfg["tokenizer"]["max_piece_length"] == 4
    assert snapshot["metadata"]["attention"]["features"] == 4

    manifest_bytes = (output / "sera_manifest.bin").read_bytes()
    assert manifest_bytes[:4] == struct.pack("<I", 0x5345524D)
    manifest = sera_transfer.decode_manifest(manifest_bytes)
    if factory.__name__ == "_create_checkpoint":
        fixture_path = ROOT / "tests" / "data" / "sera_manifest_fixture.json"
        fixture = json.loads(fixture_path.read_text())
        assert manifest.to_dict() == fixture
    else:
        assert manifest.header.magic == 0x5345524D
        assert manifest.tokenizer.p_gen == snapshot["model_config"].get("vocab_size", 0)


def test_convert_preserves_optional_config_metadata(tmp_path: Path) -> None:
    source = tmp_path / "optional_config"
    source.mkdir()

    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "num_experts": 32,
        "experts_per_token": 4,
        "intermediate_size": 16,
        "swiglu_limit": 7.0,
        "sliding_window": 128,
        "initial_context_length": 4096,
        "rope_theta": 150000.0,
        "rope_scaling_factor": 32.0,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "tau": 0.5,
    }

    rng = random.Random(6)
    tensors = {
        "model.layers.0.attention.q_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.k_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.v_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(16, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 16, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(16, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }

    (source / "config.json").write_text(json.dumps(config))
    save_file(_as_numpy_tensors(tensors), source / "model.safetensors")

    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    snapshot = pickle.loads((output / "sera_state.pkl").read_bytes())
    model_cfg = snapshot["model_config"]
    assert model_cfg["num_hidden_layers"] == 1
    assert model_cfg["num_experts"] == 32
    assert model_cfg["rope_scaling_factor"] == pytest.approx(32.0)

    attention_meta = snapshot["metadata"]["attention"]
    assert attention_meta["rope_theta"] == pytest.approx(150000.0)
    assert attention_meta["rope_scaling_factor"] == pytest.approx(32.0)
    assert attention_meta["sliding_window"] == 128
    assert attention_meta["num_key_value_heads"] == 2

    linear_meta = snapshot["metadata"]["linear"]
    assert linear_meta["num_experts"] == 32
    assert linear_meta["experts_per_token"] == 4
    assert linear_meta["swiglu_limit"] == pytest.approx(7.0)
def test_cli_verbose_emits_search_hints(tmp_path: Path) -> None:
    source = _create_openai_checkpoint(tmp_path / "verbose_logging")
    output = tmp_path / "output"

    cmd = [
        sys.executable,
        "-m",
        "gpt_oss.tools.sera_transfer",
        "--source",
        str(source),
        "--output",
        str(output),
        "--verbose",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    completed = subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    log_output = completed.stdout
    resolved_source = str(source.resolve())
    config_path = str((source / "config.json").resolve())
    weights_path = str((source / "model.safetensors").resolve())

    assert "Resolved source path" in log_output
    assert resolved_source in log_output
    assert "Registered search root" in log_output
    assert config_path in log_output
    assert weights_path in log_output
    assert "Probing" in log_output
    assert "Model config keys" in log_output
    assert "d_model" in log_output
    assert "vocab_size" in log_output
    assert "Generating tokenizer arrays" in log_output
    assert "Generating PRF arrays" in log_output
    assert "Computing attention overlays" in log_output
    assert "Collapsing FFN weights" in log_output
    assert "Computing memory coefficients" in log_output
    assert "Constructing bridge records" in log_output


def test_cli_produces_conversion_artifacts(tmp_path: Path) -> None:
    """End-to-end CLI conversion should emit manifest, arrays, and snapshot."""

    source = _create_openai_checkpoint(tmp_path / "cli_conversion")
    output = tmp_path / "sera_out"

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

    completed = subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stdout = completed.stdout
    assert "Sera Transfer Conversion Summary" in stdout

    manifest = output / "sera_manifest.bin"
    arrays_dir = output / "arrays"
    snapshot = output / "sera_state.pkl"

    assert manifest.is_file()
    assert manifest.stat().st_size > 0
    assert arrays_dir.is_dir()
    assert any(arrays_dir.glob("*.bin"))
    assert snapshot.is_file()
    assert snapshot.stat().st_size > 0


def test_format_model_config_keys_preserves_remaining_order() -> None:
    config = {
        "zeta": 0,
        "architectures": "GptOssForCausalLM",
        "epsilon": 1,
        "model_type": "gpt_oss",
        "alpha": 2,
    }

    formatted = sera_transfer._format_model_config_keys(config)

    assert formatted == "architectures, model_type, zeta, epsilon, alpha"


def test_cli_reports_missing_safetensors(tmp_path: Path, monkeypatch, capsys) -> None:
    module = importlib.reload(sera_transfer)

    def missing_safe_open(*args, **kwargs):  # noqa: ARG001
        raise ModuleNotFoundError(module._SAFETENSORS_MISSING_MSG)

    monkeypatch.setattr(module, "safe_open", missing_safe_open)

    source = tmp_path / "binary_source"
    source.mkdir()
    config = {
        "d_model": 4,
        "n_heads": 2,
        "head_dim": 2,
        "vocab_size": 8,
        "tau": 0.5,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"BINARY")

    output = tmp_path / "output"

    with pytest.raises(SystemExit) as excinfo:
        module.main([
            "--source",
            str(source),
            "--output",
            str(output),
        ])

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "sera_transfer:" in stderr
    assert "pip install safetensors" in stderr
    assert "No artefacts were written" in stderr
    assert not output.exists()


def test_cli_reports_value_error(monkeypatch, capsys, tmp_path: Path) -> None:
    module = importlib.reload(sera_transfer)

    def failing_convert(*args, **kwargs):  # noqa: ARG001
        raise ValueError("example failure message")

    monkeypatch.setattr(module, "convert", failing_convert)

    source = tmp_path / "source"
    output = tmp_path / "output"

    with pytest.raises(SystemExit) as excinfo:
        module.main([
            "--source",
            str(source),
            "--output",
            str(output),
        ])

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "sera_transfer: example failure message" in stderr


def test_convert_handles_path_resolution_failures(tmp_path: Path, monkeypatch) -> None:
    source = _create_openai_checkpoint(tmp_path / "resolve_error" / "checkpoint")
    output = tmp_path / "resolve_error" / "output"

    original_resolve = Path.resolve
    failure_targets = {source, output}

    def fake_resolve(self):
        if any(self == target for target in failure_targets):
            raise OSError("resolve failed")
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    summary = sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    assert summary.output == output.absolute()
    assert summary.output.exists()
    assert summary.manifest_path.exists()


def test_convert_handles_stat_failure(tmp_path: Path, monkeypatch) -> None:
    base = _create_checkpoint(tmp_path / "stat_failure_base")

    source = tmp_path / "stat_failure"
    source.mkdir()
    shutil.copytree(base, source / "original")

    output = tmp_path / "output"

    original_exists = Path.exists
    call_count = {"value": 0}

    def flaky_exists(self: Path) -> bool:
        if self == output:
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise OSError("mock stat failure")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", flaky_exists)

    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    assert call_count["value"] >= 1
    assert (output / "sera_manifest.bin").exists()


def test_convert_continues_when_snapshot_generation_fails(tmp_path: Path, monkeypatch) -> None:
    module = importlib.reload(sera_transfer)

    class DummyConfig:
        pass

    class BrokenSera:
        def __init__(self, config):  # noqa: D401,ARG002 - signature matches runtime expectation
            self.config = config

        def snapshot(self):  # noqa: D401 - simple stub method
            raise RuntimeError("snap failure")

    def broken_loader():
        return BrokenSera, DummyConfig

    monkeypatch.setattr(module, "_load_sera_runtime", broken_loader)

    source = _create_checkpoint(tmp_path / "snapshot_failure")
    output = tmp_path / "output"

    summary = module.convert(source, output, r=4, r_v=2, top_l=2)

    assert summary.output == output
    snapshot_path = output / "sera_state.pkl"
    json_path = output / "sera_state.json"
    assert snapshot_path.exists()
    assert json_path.exists()

    snapshot_blob = pickle.loads(snapshot_path.read_bytes())
    runtime_snapshot = snapshot_blob["sera_snapshot"]
    assert runtime_snapshot["status"] == "error"
    assert runtime_snapshot["error"]["type"] == "RuntimeError"
    assert snapshot_blob["metadata"]["runtime_snapshot_error"]["message"] == "snap failure"


def test_convert_continues_when_runtime_loader_missing(tmp_path: Path, monkeypatch) -> None:
    module = importlib.reload(sera_transfer)

    def missing_loader():
        raise ModuleNotFoundError("runtime unavailable")

    monkeypatch.setattr(module, "_load_sera_runtime", missing_loader)

    source = _create_checkpoint(tmp_path / "runtime_missing")
    output = tmp_path / "output"

    summary = module.convert(source, output, r=4, r_v=2, top_l=2)

    assert summary.output == output

    snapshot_path = output / "sera_state.pkl"
    assert snapshot_path.exists()

    snapshot_blob = pickle.loads(snapshot_path.read_bytes())
    runtime_snapshot = snapshot_blob["sera_snapshot"]
    assert runtime_snapshot["status"] == "error"
    assert runtime_snapshot["error"]["type"] == "ModuleNotFoundError"
    assert runtime_snapshot["error"]["message"] == "runtime unavailable"

    metadata_error = snapshot_blob["metadata"]["runtime_snapshot_error"]
    assert metadata_error["type"] == "ModuleNotFoundError"
    assert metadata_error["message"] == "runtime unavailable"


def test_deterministic_conversion(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path / "deterministic")
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


def test_convert_emits_json_snapshot(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path / "json_snapshot")
    output = tmp_path / "output"

    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    json_path = output / "sera_state.json"
    assert json_path.exists()

    json_blob = json.loads(json_path.read_text())
    vocab = json_blob["sera_snapshot"]["config"]["tokenizer"]["vocabulary"]
    assert any(
        isinstance(key, str) and key.startswith(sera_transfer.JSON_BYTES_PREFIX)
        for key in vocab
    )

    from gpt_oss.cli import sera_chat

    json_state = sera_chat._load_state(json_path)
    pickle_state = pickle.loads((output / "sera_state.pkl").read_bytes())
    assert (
        json_state["sera_snapshot"]["config"]["tokenizer"]["vocabulary"]
        == pickle_state["sera_snapshot"]["config"]["tokenizer"]["vocabulary"]
    )


def test_parse_args_expands_user(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    args = sera_transfer._parse_args(
        [
            "--source",
            "~/checkpoint",
            "--output",
            "~/output",
            "--summary-output",
            "~/summary.json",
            "--original-subdir",
            "~/nested/original",
        ]
    )

    assert args.source == Path(home / "checkpoint")
    assert args.output == Path(home / "output")
    assert args.summary_output == Path(home / "summary.json")
    assert args.original_subdir == Path(home / "nested" / "original")


def test_written_arrays_match_reference(tmp_path: Path) -> None:
    source = _create_checkpoint(tmp_path / "reference")
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    config_data = json.loads((source / "config.json").read_text())
    tensors = sera_transfer.load_tensors(source / "model.safetensors")
    cfg = sera_transfer.ModelConfig.from_dict(config_data, tensors=tensors)
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
    source = _create_checkpoint(tmp_path / "checksums")
    output = tmp_path / "output"
    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    arrays_dir = output / "arrays"
    checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(arrays_dir.glob("*.bin"))
    }

    expected = {
        "R_init.bin": "7d7f9cf695a015d4dd5e862e102e6fc39ab76b69b5b9318434a5a6f196b0c595",
        "T_1.bin": "575385afc5de59cd4cff920b8f5ca9c5f57d2f1a469d1ffadcc9ecfc689d1209",
        "T_2.bin": "0313a62554d8b532bce4261a39baafc98526e26bf473265b58aa3b12f3e1c3e9",
        "T_3.bin": "e87942391bf2fe9d6fa5d81cd3d1f2b023d123bb9fb5a4adefc5595ad00c9b9b",
        "T_4.bin": "797e35cc1d64523fa13907ce9b0e0dc06c913cd07b07eb28052cd30dd6c0919f",
        "bridge_hubs.bin": "b1765af176adc937bbdcaded86d2e8bc756358dba74a03412e16240b2f8eff30",
        "bridge_qDin.bin": "1edef4d83792250982a9ee583b85ccffbe40cc13c4ecf080e2cbfa1a09606709",
        "bridge_qDout.bin": "2090c735da1cbcbe3fc1a72437219ce325eecdb978f7f1e9c60785a92b374197",
        "cuckoo_delta.bin": "576acdb10a9d0d9f5dcb9970eb7b6105d930cd3317afcf2a07c62330586a3145",
        "delaybuf_init.bin": "2d65a9ed7f442cc54cb983094b6feb9a99635210701c038b6d316e451a2b3654",
        "linear_bias.bin": "9c4dcbf57c1e5a6890a9c882e3ba12a90ece9fbf99a86818652b9969170d37ba",
        "linear_keys.bin": "15913f521cf0e0ca841b4617aecf55b7dece60ff8bb610968f8fd13b081a6d27",
        "linear_mphf.bin": "990652fb2ceaa7532b2b0e40b7ac1a145262b6272e119c04664ba15b53ff3283",
        "linear_weights.bin": "b9e896e87767b2defb037a085d5bbd9776b7ee429fe3f7aa3882370ba5923961",
        "memory_coeff.bin": "724d23280561eb1e7909a86fefdc778a6d8ba0e209734bd7f26edffe48a76182",
        "overlays_DeltaW.bin": "6cd9368a0c3333594fe8a356d010538cd4c577b94c5f974183831b42dccbd1c6",
        "overlays_H.bin": "646005ada365721c1a4c7c6c4c75d2f8153ffa24f937e58be8224b79dc2c411b",
        "overlays_U.bin": "9ba6d03ff54587bdd0304c6b9b128d4acb73afd449c305e87f3fc841edf551b1",
        "peer_scores.bin": "7044204096c6000dacb5c823a58031797bbea9515d08f7383f76064dbdf7d227",
        "prf_W.bin": "b94cc880877de076921f8bea3a3f3b9d46a7bf91349208107d515da9659cdbb5",
        "s_init.bin": "5a534c99468d8f1d53ed0b4058e1b376265361a42fc8fd57cbf7d28e11d0eab2",
        "tokenizer_fst.bin": "3f7f982b24185fe6da5aa84f43adec88cd4236447ed7f8f854399c4700455115",
        "whitening_mu.bin": "6de8277b57b6819bda93c1e23cdbc536d9e3806760ec71b78de79c8d7cb45982",
        "whitening_sig2.bin": "a49a905b5c9d158bf85efc300b0bb9c29908c4d95fc68dc582ffa054569d8447",
    }

    differences = {
        name: (checksums[name], expected.get(name))
        for name in sorted(checksums)
        if checksums[name] != expected.get(name)
    }
    assert differences == {}
