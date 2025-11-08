from __future__ import annotations

import importlib
import inspect
import hashlib
import json
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpt_oss._stubs.safetensors as safetensors_stub
from gpt_oss._stubs.safetensors.numpy import save_file

from gpt_oss.tools import sera_transfer


def _create_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _create_vector(length: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def _sample_checkpoint_contents() -> tuple[dict, dict[str, list]]:
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
    return config, tensors


def _create_checkpoint(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)

    config, tensors = _sample_checkpoint_contents()
    (source / "config.json").write_text(json.dumps(config))
    save_file(tensors, source / "model.safetensors")
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
    tensors = {}
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
    save_file(tensors, source / "model.safetensors")
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

    # Force the stub to be used so that binary payloads trigger the guidance.
    monkeypatch.setattr(module, "safe_open", safetensors_stub.safe_open)

    binary_checkpoint = tmp_path / "model.safetensors"
    binary_checkpoint.write_bytes(b"BIN\x00\x01")

    with pytest.raises(ModuleNotFoundError) as excinfo:
        module.load_tensors(binary_checkpoint)

    message = str(excinfo.value)
    assert "pip install safetensors" in message

    importlib.reload(sera_transfer)


def test_load_tensors_stub_passes_python_framework(tmp_path, monkeypatch):
    module = importlib.reload(sera_transfer)

    assert module.safe_open is not None
    assert module._looks_like_repo_stub_safe_open(module.safe_open)

    original_safe_open = safetensors_stub.safe_open
    calls: list[tuple[str, str | None]] = []

    def recording_safe_open(*args, **kwargs):
        if len(args) >= 2:
            calls.append(("positional", args[1]))
        else:
            calls.append(("keyword", kwargs.get("framework")))
        return original_safe_open(*args, **kwargs)

    recording_safe_open.__module__ = original_safe_open.__module__

    monkeypatch.setattr(module, "safe_open", recording_safe_open)

    payload = {
        "tensors": {
            "foo": {"shape": [1, 2], "data": [1.0, 2.0]},
        }
    }
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_text(json.dumps(payload))

    tensors = module.load_tensors(checkpoint)

    tensor = tensors["foo"]
    assert tensor == [[1.0, 2.0]]
    assert calls == [("positional", "python")]


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

    def keyword_only_safe_open(path, *args, framework, device=None):  # noqa: ARG001
        if args:
            raise TypeError("safe_open() takes 1 positional argument but 2 were given")
        calls.append(("keyword", framework))
        return _FakeSafeFile(payload)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")

    monkeypatch.setattr(module, "safe_open", keyword_only_safe_open)

    tensors = module.load_tensors(checkpoint)

    tensor = tensors["foo"]
    assert isinstance(tensor, _FakeTensor)
    assert tensor.tolist() == payload["foo"]
    assert calls == [("keyword", "pt")]


def test_load_tensors_prefers_real_safetensors_when_available(tmp_path, monkeypatch):
    module = importlib.reload(sera_transfer)

    assert module.safe_open is not None
    assert module._looks_like_repo_stub_safe_open(module.safe_open)

    for name in list(sys.modules):
        if name == "safetensors" or name.startswith("safetensors."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_impl_root = tmp_path / "real_impl"
    real_pkg = real_impl_root / "safetensors"
    real_pkg.mkdir(parents=True)
    (real_pkg / "__init__.py").write_text(
        """
import pickle
from pathlib import Path


class SafetensorError(RuntimeError):
    pass


class _FakeSafeFile:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        if not raw.startswith(b"BIN"):
            raise SafetensorError("Unexpected payload")
        loaded = pickle.loads(raw[3:])
        self._tensors = {key: _FakeTensor(value) for key, value in loaded.items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return self._tensors[key]


class _FakeTensor:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


def safe_open(path, framework="pt", **kwargs):
    if framework != "pt":
        raise SafetensorError(f"Unexpected framework: {framework!r}")
    return _FakeSafeFile(path)
"""
    )

    monkeypatch.syspath_prepend(str(real_impl_root))
    importlib.invalidate_caches()

    expected = {"foo": [[1.0, 2.0], [3.0, 4.0]]}
    payload = pickle.dumps(expected)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"BIN" + payload)

    tensors = module.load_tensors(checkpoint)
    assert tensors.keys() == expected.keys()
    for name, tensor in tensors.items():
        assert hasattr(tensor, "tolist")
        assert tensor.tolist() == expected[name]

    safe_open_fn = module._resolve_safe_open()
    assert not module._looks_like_repo_stub_safe_open(safe_open_fn)
    owning_module = inspect.getmodule(safe_open_fn)
    assert owning_module is not None
    assert Path(owning_module.__file__).resolve().parent == real_pkg.resolve()

    for name in list(sys.modules):
        if name == "safetensors" or name.startswith("safetensors."):
            sys.modules.pop(name, None)

    monkeypatch.undo()
    importlib.reload(sera_transfer)


def test_resolve_safe_open_prefers_real_wheel(tmp_path, monkeypatch):
    module = importlib.reload(sera_transfer)

    assert module.safe_open is not None
    assert module._looks_like_repo_stub_safe_open(module.safe_open)

    # Ensure the cached resolver currently points at the repository stub.
    initial = module._resolve_safe_open()
    assert module._looks_like_repo_stub_safe_open(initial)
    assert initial is module.safe_open

    real_root = tmp_path / "real_wheel"
    real_pkg = real_root / "safetensors"
    real_pkg.mkdir(parents=True)
    (real_pkg / "__init__.py").write_text(
        """
from pathlib import Path


class _FakeTensor:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


class _FakeSafeFile:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        if raw != b"real":
            raise SafetensorError("Unexpected payload")
        self._payload = {"foo": _FakeTensor([[1.0]])}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self._payload.keys())

    def get_tensor(self, key):
        return self._payload[key]


class SafetensorError(RuntimeError):
    pass


def safe_open(path, framework="pt", **kwargs):
    if framework != "pt":
        raise SafetensorError(f"Unexpected framework: {framework!r}")
    return _FakeSafeFile(path)
"""
    )

    monkeypatch.syspath_prepend(str(real_root))
    importlib.invalidate_caches()

    # Ensure the stub is currently loaded to simulate an environment upgrade.
    assert module._looks_like_repo_stub_safe_open(module.safe_open)

    # Create a dummy file that only the real implementation accepts.
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"real")

    try:
        resolved = module._resolve_safe_open()
        assert not module._looks_like_repo_stub_safe_open(resolved)
        owning_module = inspect.getmodule(resolved)
        assert owning_module is not None
        assert Path(owning_module.__file__).resolve().parent == real_pkg.resolve()

        with resolved(checkpoint) as handle:
            assert handle.keys() == ["foo"]
            tensor = handle.get_tensor("foo")
            assert tensor.tolist() == [[1.0]]
    finally:
        for name in list(sys.modules):
            if name == "safetensors" or name.startswith("safetensors."):
                sys.modules.pop(name, None)
        monkeypatch.undo()
        importlib.reload(sera_transfer)


@pytest.mark.parametrize("mode", ["stub", "wheel"])
def test_convert_handles_stub_and_real_safetensors(tmp_path, monkeypatch, mode):
    module = importlib.reload(sera_transfer)

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    config, tensors = _sample_checkpoint_contents()
    (source / "config.json").write_text(json.dumps(config))

    if mode == "stub":
        stub_safe_open = safetensors_stub.safe_open
        monkeypatch.setattr(module, "_resolve_safe_open", lambda: stub_safe_open)

        payload = {
            "tensors": {
                key: {"shape": _tensor_shape(value), "data": _flatten_tensor(value)}
                for key, value in tensors.items()
            }
        }
        (source / "model.safetensors").write_text(json.dumps(payload))
    else:
        fake_impl = tmp_path / "fake_safetensors.py"
        fake_impl.write_text(
            """
import pickle
from pathlib import Path


class SafetensorError(RuntimeError):
    pass


class _FakeTensor:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload

    def __len__(self):
        return len(self._payload)

    def __iter__(self):
        return iter(self._payload)

    def __getitem__(self, index):
        return self._payload[index]

    @property
    def shape(self):
        dims = []
        current = self._payload
        while isinstance(current, list):
            dims.append(len(current))
            if not current:
                break
            current = current[0]
        return tuple(dims)


class _FakeSafeFile:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        if not raw.startswith(b"BIN"):
            raise SafetensorError("Unexpected payload")
        payload = pickle.loads(raw[3:])
        self._payload = {key: _FakeTensor(value) for key, value in payload.items()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def keys(self):
        return list(self._payload.keys())

    def get_tensor(self, key):
        return self._payload[key]


def safe_open(path, framework="pt", **kwargs):
    if framework != "pt":
        raise SafetensorError(f"Unexpected framework: {framework!r}")
    return _FakeSafeFile(path)
"""
        )
        spec = importlib.util.spec_from_file_location("fake_safetensors", fake_impl)
        assert spec is not None and spec.loader is not None
        fake_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fake_module
        spec.loader.exec_module(fake_module)
        fake_safe_open = fake_module.safe_open
        monkeypatch.setattr(module, "_resolve_safe_open", lambda: fake_safe_open)

        (source / "model.safetensors").write_bytes(b"BIN" + pickle.dumps(tensors))

    safe_open_fn = module._resolve_safe_open()
    assert module._looks_like_repo_stub_safe_open(safe_open_fn) is (mode == "stub")

    tensors_loaded = module.load_tensors(source / "model.safetensors")
    assert tensors_loaded.keys() == tensors.keys()

    summary = module.convert(source, output, r=4, r_v=2, top_l=2)
    assert summary.manifest_path.exists()
    assert summary.output == output
    assert summary.total_bytes > 0

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
    qkv_weight = _create_matrix(12, 4, rng)
    tensors = {
        "model.layers.0.attention.qkv.weight": qkv_weight,
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(8, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 8, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(8, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }

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
    assert w_q == qkv_weight[:4]
    assert w_k == qkv_weight[4:8]
    assert w_v == qkv_weight[8:]


def test_model_config_infers_head_dim_from_qkv_tensor() -> None:
    config = {
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
    }

    rng = random.Random(3)
    qkv_weight = _create_matrix(16, 4, rng)
    tensors = {
        "model.layers.0.attention.qkv.weight": qkv_weight,
        "model.layers.0.attention.o_proj.weight": _create_matrix(4, 4, rng),
        "model.layers.0.mlp.gate_proj.weight": _create_matrix(8, 4, rng),
        "model.layers.0.mlp.down_proj.weight": _create_matrix(4, 8, rng),
        "model.layers.0.mlp.gate_proj.bias": _create_vector(8, rng),
        "model.layers.0.mlp.down_proj.bias": _create_vector(4, rng),
    }

    cfg = sera_transfer.ModelConfig.from_dict(config, tensors=tensors)

    assert cfg.head_dim == 4
    layer = cfg.layers[0]
    assert tensors[layer.w_q] == qkv_weight[:8]
    assert tensors[layer.w_k] == qkv_weight[8:12]
    assert tensors[layer.w_v] == qkv_weight[12:]


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
    assert "docs/howto-sera-transfer.md" in message


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

    manifest = (output / "sera_manifest.bin").read_bytes()
    assert manifest[:4] == struct.pack("<I", 0x5345524D)

    snapshot = pickle.loads((output / "sera_state.pkl").read_bytes())
    assert "sera_snapshot" in snapshot
    snapshot_cfg = snapshot["sera_snapshot"]["config"]
    assert snapshot_cfg["tokenizer"]["max_piece_length"] == 4
    assert snapshot["metadata"]["attention"]["features"] == 4


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
    save_file(tensors, source / "model.safetensors")

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
    monkeypatch.setattr(module, "_resolve_safe_open", lambda: safetensors_stub.safe_open)

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
    assert not output.exists()


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
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise OSError("mock stat failure")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", flaky_exists)

    sera_transfer.convert(source, output, r=4, r_v=2, top_l=2)

    assert call_count["value"] >= 1
    assert (output / "sera_manifest.bin").exists()


def test_convert_supports_stub_and_wheel_safe_open(tmp_path: Path, monkeypatch) -> None:
    module = importlib.reload(sera_transfer)

    stub_config, stub_tensors = _sample_checkpoint_contents()
    stub_source = tmp_path / "stub_source"
    stub_source.mkdir()
    (stub_source / "config.json").write_text(json.dumps(stub_config))
    stub_payload = {
        "tensors": {
            name: {"shape": _tensor_shape(data), "data": _flatten_tensor(data)}
            for name, data in stub_tensors.items()
        }
    }
    (stub_source / "model.safetensors").write_text(json.dumps(stub_payload))

    stub_frameworks: list[str] = []
    stub_safe_open = safetensors_stub.safe_open

    def recording_stub_safe_open(path, framework="python", **kwargs):
        stub_frameworks.append(framework)
        return stub_safe_open(path, framework=framework, **kwargs)

    recording_stub_safe_open.__module__ = getattr(
        stub_safe_open, "__module__", "gpt_oss._stubs.safetensors"
    )

    monkeypatch.setattr(module, "_resolve_safe_open", lambda: recording_stub_safe_open)

    stub_output = tmp_path / "stub_output"
    module.convert(stub_source, stub_output, r=4, r_v=2, top_l=2)
    assert stub_frameworks == ["python"]
    assert (stub_output / "sera_manifest.bin").exists()

    wheel_config, wheel_tensors = _sample_checkpoint_contents()
    wheel_source = tmp_path / "wheel_source"
    wheel_source.mkdir()
    (wheel_source / "config.json").write_text(json.dumps(wheel_config))
    wheel_source.joinpath("model.safetensors").write_bytes(b"BIN" + pickle.dumps(wheel_tensors))

    wheel_frameworks: list[str] = []

    class _WheelTensor:
        def __init__(self, payload):
            self._payload = payload

        def tolist(self):
            return self._payload

        def __len__(self):
            return len(self._payload)

        def __iter__(self):
            return iter(self._payload)

        def __getitem__(self, index):
            return self._payload[index]

        @property
        def shape(self):
            dims: list[int] = []
            current = self._payload
            while isinstance(current, list):
                dims.append(len(current))
                if not current:
                    break
                current = current[0]
            return tuple(dims)

    class _WheelSafeFile:
        def __init__(self, path, *, framework="numpy"):
            wheel_frameworks.append(framework)
            if framework != "numpy":
                raise RuntimeError(f"Unexpected framework: {framework!r}")
            raw = Path(path).read_bytes()
            if not raw.startswith(b"BIN"):
                raise ValueError("Unexpected payload")
            loaded = pickle.loads(raw[3:])
            self._tensors = {
                key: _WheelTensor(value)
                for key, value in loaded.items()
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return list(self._tensors.keys())

        def get_tensor(self, key):
            return self._tensors[key]

    def fake_safe_open(path, framework="numpy", **kwargs):
        return _WheelSafeFile(path, framework=framework)

    monkeypatch.setattr(module, "_resolve_safe_open", lambda: fake_safe_open)

    wheel_output = tmp_path / "wheel_output"
    module.convert(wheel_source, wheel_output, r=4, r_v=2, top_l=2)
    assert wheel_frameworks == ["numpy"]
    assert (wheel_output / "sera_manifest.bin").exists()

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
        path.name: hashlib.sha256(_payload(path)).hexdigest()
        for path in sorted(arrays_dir.glob("*.bin"))
    }

    expected = {
        "R_init.bin": "c1e397da306abc45e5fbe4f03330dab17cceb14a5c46561e3f51f7fc4a1f90e8",
        "T_1.bin": "df5225054cae663ff1c726d803091622ffc765309c9e9b9e573c0bbd08105a37",
        "T_2.bin": "067fece1127c0e37ff0bf1ff566b727f33dd3f5d5ba15431916b61adfebc2e5b",
        "T_3.bin": "581fca69ff42c2b69b83ba1f128513e611ee8238448e0eb920a820e74b78de1a",
        "T_4.bin": "1147d97ec1eae29ec741abdaabfb36c6db64485f9a5438e51a4656f98c6eb3ec",
        "bridge_hubs.bin": "278257c861ac4de67792e075d1fc4b5d35cf9c2dace8a552c2b806968bbeb70a",
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
        "peer_scores.bin": "a2336c64b60dbc69d3353f8db55a2c07f280ae1859056482cbcfb6b209726820",
        "prf_W.bin": "fc6134f0ec03aff0ca21bfc7e0a5a2c70afa8c73ab290f7b694fd9f008fe293b",
        "s_init.bin": "4d848697b5664465351e1ba48be3e74ad691e88b1260ccb3e0b73db7118f55c4",
        "tokenizer_fst.bin": "45da3c4be5bdb7c344bb6e2ede224d877de3e51d50e39788e98a437dc7109d2e",
        "whitening_mu.bin": "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb",
        "whitening_sig2.bin": "3c6add9ce200ba6e99cb568fd4213c456fc2e4b79d36d82eefad6705493b265a",
    }

    assert checksums == expected
