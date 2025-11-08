"""Tests for the Sera transfer conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from gpt_oss.tools import sera_transfer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if "pip install numpy" in str(exc):
        pytest.skip("Real numpy is required for Sera transfer tests", allow_module_level=True)
    raise


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
        w_q="W_Q",
        w_k="W_K",
        w_v="W_V",
        w_o="W_O",
        w1="FFN_W1",
        w2="FFN_W2",
        b1="FFN_B1",
        b2="FFN_B2",
    )
    dummy_cfg = SimpleNamespace(
        d_model=16,
        vocab_size=8,
        layers=[dummy_layer],
        rope_theta=None,
        num_key_value_heads=None,
        sliding_window=None,
        initial_context_length=None,
        rope_scaling_factor=None,
        rope_ntk_alpha=None,
        rope_ntk_beta=None,
        tau=1.0,
        num_experts=None,
        experts_per_token=None,
        intermediate_size=None,
        swiglu_limit=None,
    )

    def fake_from_dict(data, tensors=None):  # noqa: ARG001 - signature kept for compatibility
        return dummy_cfg

    monkeypatch.setattr(sera_transfer.ModelConfig, "from_dict", staticmethod(fake_from_dict))

    monkeypatch.setattr(
        sera_transfer,
        "tokenizer_arrays",
        lambda *args, **kwargs: ({"tok": [0]}, {"info": "tokenizer"}),
    )
    monkeypatch.setattr(sera_transfer, "compute_prf", lambda *args, **kwargs: {"prf": [0.0]})
    monkeypatch.setattr(
        sera_transfer,
        "compute_overlays",
        lambda *args, **kwargs: {"overlay": [0.0]},
    )
    monkeypatch.setattr(
        sera_transfer,
        "collapse_ffn",
        lambda *args, **kwargs: (
            {
                "linear_mphf": [0],
                "linear_keys": [0],
                "linear_weights": [0.0],
                "linear_bias": [0.0],
                "cuckoo_delta": [0],
            },
            {"info": "linear"},
        ),
    )
    monkeypatch.setattr(
        sera_transfer,
        "memory_coefficients",
        lambda *args, **kwargs: (
            {"memory_coeff": [0.0], "delaybuf_init": [0.0]},
            {"info": "memory"},
        ),
    )
    monkeypatch.setattr(
        sera_transfer,
        "bridge_records",
        lambda *args, **kwargs: (
            {
                "bridge_hubs": [0],
                "bridge_qDin": [0],
                "bridge_qDout": [0],
                "peer_scores": [0],
            },
            {"info": "bridge"},
        ),
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

    (original / "config.json").write_text(json.dumps({"location": "original"}))
    (original / "model.safetensors").write_bytes(b"")

    captured_model_path: dict[str, Path] = {}
    captured_config: dict[str, object] = {}

    def fake_load_tensors(path: Path):
        captured_model_path["path"] = path
        return {}

    dummy_layer = SimpleNamespace(
        w_q="W_Q",
        w_k="W_K",
        w_v="W_V",
        w_o="W_O",
        w1="FFN_W1",
        w2="FFN_W2",
        b1="FFN_B1",
        b2="FFN_B2",
    )
    dummy_cfg = SimpleNamespace(
        d_model=16,
        vocab_size=8,
        layers=[dummy_layer],
        rope_theta=None,
        num_key_value_heads=None,
        sliding_window=None,
        initial_context_length=None,
        rope_scaling_factor=None,
        rope_ntk_alpha=None,
        rope_ntk_beta=None,
        tau=1.0,
        num_experts=None,
        experts_per_token=None,
        intermediate_size=None,
        swiglu_limit=None,
    )

    def fake_from_dict(data, tensors=None):  # noqa: ARG001 - compatibility with signature
        captured_config["data"] = data
        return dummy_cfg

    monkeypatch.setattr(
        sera_transfer.ModelConfig,
        "from_dict",
        staticmethod(fake_from_dict),
    )
    monkeypatch.setattr(sera_transfer, "load_tensors", fake_load_tensors)

    output = tmp_path / "output"
    output.mkdir()

    # Should not raise FileNotFoundError even though source lacks the files directly.
    sera_transfer.convert(source, output)

    assert captured_model_path["path"] == original / "model.safetensors"
    assert captured_config["data"]["location"] == "original"


def test_convert_honours_original_subdir(tmp_path: Path, monkeypatch):
    """A custom `original_subdir` should take precedence over heuristics."""

    source = tmp_path / "source"
    export = source / "hf_export"
    export.mkdir(parents=True)

    (export / "config.json").write_text(json.dumps({"location": "custom"}))
    (export / "model.safetensors").write_bytes(b"")

    captured_model_path: dict[str, Path] = {}
    captured_config: dict[str, object] = {}

    def fake_load_tensors(path: Path):
        captured_model_path["path"] = path
        return {}

    dummy_layer = SimpleNamespace(
        w_q="W_Q",
        w_k="W_K",
        w_v="W_V",
        w_o="W_O",
        w1="FFN_W1",
        w2="FFN_W2",
        b1="FFN_B1",
        b2="FFN_B2",
    )
    dummy_cfg = SimpleNamespace(
        d_model=16,
        vocab_size=8,
        layers=[dummy_layer],
        rope_theta=None,
        num_key_value_heads=None,
        sliding_window=None,
        initial_context_length=None,
        rope_scaling_factor=None,
        rope_ntk_alpha=None,
        rope_ntk_beta=None,
        tau=1.0,
        num_experts=None,
        experts_per_token=None,
        intermediate_size=None,
        swiglu_limit=None,
    )

    def fake_from_dict(data, tensors=None):  # noqa: ARG001 - compatibility with signature
        captured_config["data"] = data
        return dummy_cfg

    monkeypatch.setattr(
        sera_transfer.ModelConfig,
        "from_dict",
        staticmethod(fake_from_dict),
    )
    monkeypatch.setattr(sera_transfer, "load_tensors", fake_load_tensors)

    output = tmp_path / "output"
    output.mkdir()

    sera_transfer.convert(source, output, original_subdir="hf_export")

    assert captured_model_path["path"] == export / "model.safetensors"
    assert captured_config["data"]["location"] == "custom"


def test_convert_returns_summary(tmp_path: Path, monkeypatch):
    """`convert` should return a rich summary describing generated artefacts."""

    source = tmp_path / "source"
    original = source / "original"
    original.mkdir(parents=True)

    (original / "config.json").write_text(json.dumps({"location": "original"}))
    (original / "model.safetensors").write_bytes(b"")

    monkeypatch.setattr(sera_transfer, "load_tensors", lambda path: {})

    output = tmp_path / "output"
    output.mkdir()

    summary = sera_transfer.convert(source, output, verbose=False)

    assert isinstance(summary, sera_transfer.ConversionSummary)
    assert summary.array_count > 0
    assert summary.total_bytes >= 0
    assert summary.manifest_path == output / "sera_manifest.bin"
    assert any(info.kind == "json" for info in summary.snapshots)
    assert summary.log_path == output / "conversion.log"
    assert summary.log_path.exists()

    formatted = sera_transfer.format_summary(summary)
    assert "Sera Transfer Conversion Summary" in formatted
    assert str(output) in formatted
    assert "Log file:" in formatted


def test_convert_writes_error_log(tmp_path: Path):
    """Conversion failures should still leave a log describing the issue."""

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        sera_transfer.convert(source, output, verbose=False)

    log_path = output / "conversion.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "Unable to locate 'config.json'" in log_text
