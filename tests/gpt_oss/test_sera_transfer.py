"""Tests for the Sera transfer conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_oss.tools import sera_transfer


class DummyConfig:
    def __init__(self):
        self.value = 1


class DummySera:
    def __init__(self, config):
        self.config = config

    def snapshot(self):
        return {"config": {}}


@pytest.fixture(autouse=True)
def patch_conversion_dependencies(monkeypatch):
    """Stub out heavyweight conversion helpers so tests remain lightweight."""

    dummy_layer = SimpleNamespace(
        w_k="W_K",
        w_o="W_O",
        w1="FFN_W1",
        w2="FFN_W2",
        b1="FFN_B1",
        b2="FFN_B2",
    )
    dummy_cfg = SimpleNamespace(d_model=16, vocab_size=8, layers=[dummy_layer])

    def fake_from_dict(data):  # noqa: ARG001 - signature kept for compatibility
        return dummy_cfg

    monkeypatch.setattr(sera_transfer.ModelConfig, "from_dict", staticmethod(fake_from_dict))

    monkeypatch.setattr(sera_transfer, "tokenizer_arrays", lambda vocab_size: {"tok": [0]})
    monkeypatch.setattr(sera_transfer, "compute_prf", lambda *args, **kwargs: {"prf": [0.0]})
    monkeypatch.setattr(
        sera_transfer,
        "compute_overlays",
        lambda *args, **kwargs: {"overlay": [0.0]},
    )
    monkeypatch.setattr(
        sera_transfer,
        "collapse_ffn",
        lambda *args, **kwargs: {
            "linear_mphf": [0],
            "linear_keys": [0],
            "linear_weights": [0.0],
            "linear_bias": [0.0],
            "cuckoo_delta": [0],
        },
    )
    monkeypatch.setattr(
        sera_transfer,
        "memory_coefficients",
        lambda *args, **kwargs: {"memory_coeff": [0.0], "delaybuf_init": [0.0]},
    )
    monkeypatch.setattr(
        sera_transfer,
        "bridge_records",
        lambda *args, **kwargs: {
            "bridge_hubs": [0],
            "bridge_qDin": [0],
            "bridge_qDout": [0],
            "peer_scores": [0],
        },
    )
    monkeypatch.setattr(sera_transfer, "write_array", lambda *args, **kwargs: b"")
    monkeypatch.setattr(sera_transfer, "write_manifest", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        sera_transfer,
        "_load_sera_runtime",
        lambda: (DummySera, DummyConfig),
    )


def test_convert_finds_files_in_original(tmp_path: Path, monkeypatch):
    """`convert` should probe `original/` when locating model artefacts."""

    source = tmp_path / "source"
    original = source / "original"
    original.mkdir(parents=True)

    (original / "config.json").write_text(json.dumps({}))
    (original / "model.safetensors").write_bytes(b"")

    captured_model_path: dict[str, Path] = {}

    def fake_load_tensors(path: Path):
        captured_model_path["path"] = path
        return {}

    monkeypatch.setattr(sera_transfer, "load_tensors", fake_load_tensors)

    output = tmp_path / "output"
    output.mkdir()

    # Should not raise FileNotFoundError even though source lacks the files directly.
    sera_transfer.convert(source, output)

    assert captured_model_path["path"] == original / "model.safetensors"
