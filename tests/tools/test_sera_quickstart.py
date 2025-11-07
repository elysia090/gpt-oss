from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
import types
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.modules.pop("safetensors", None)
sys.modules.pop("safetensors.numpy", None)

_safetensors_spec = importlib.util.spec_from_file_location(
    "safetensors", ROOT / "safetensors" / "__init__.py"
)
assert _safetensors_spec is not None and _safetensors_spec.loader is not None
_safetensors_module = importlib.util.module_from_spec(_safetensors_spec)
sys.modules["safetensors"] = _safetensors_module
_safetensors_spec.loader.exec_module(_safetensors_module)

import gpt_oss.tools.sera_quickstart as quickstart
import pytest

_safetensors_numpy_spec = importlib.util.spec_from_file_location(
    "safetensors.numpy", ROOT / "safetensors" / "numpy.py"
)
assert _safetensors_numpy_spec is not None and _safetensors_numpy_spec.loader is not None
_safetensors_numpy = importlib.util.module_from_spec(_safetensors_numpy_spec)
sys.modules.setdefault("safetensors.numpy", _safetensors_numpy)
_safetensors_numpy_spec.loader.exec_module(_safetensors_numpy)
save_file = _safetensors_numpy.save_file
quickstart.sera_transfer.safe_open = _safetensors_module.safe_open  # type: ignore[attr-defined]


def _create_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _create_vector(length: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def _create_checkpoint(tmp_path: Path) -> Path:
    source = tmp_path / "checkpoint"
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


@pytest.mark.parametrize("launch_chat", [False, True])
def test_quickstart_pipeline(tmp_path: Path, monkeypatch, launch_chat: bool) -> None:
    checkpoint_dir = _create_checkpoint(tmp_path)
    download_dir = tmp_path / "download"
    output_dir = tmp_path / "sera"

    calls: list[tuple[str, str | None]] = []
    chat_calls: list[tuple[Path, tuple[str, ...]]] = []

    def _fake_snapshot_download(
        repo_id: str,
        revision: str | None,
        local_dir: str,
        local_dir_use_symlinks: bool,
    ) -> str:
        calls.append((repo_id, revision))
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for item in checkpoint_dir.iterdir():
            target = destination / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return str(destination)

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.snapshot_download = _fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_stub)

    if launch_chat:
        def _fake_launch_chat_cli(manifest: Path, extra: Sequence[str]) -> int:
            chat_calls.append((manifest, tuple(extra)))
            return 0

        monkeypatch.setattr(quickstart, "_launch_chat_cli", _fake_launch_chat_cli)

    argv = [
        "--repo-id",
        "openai/gpt-oss-20b",
        "--download-dir",
        str(download_dir),
        "--output-dir",
        str(output_dir),
        "--r",
        "4",
        "--rv",
        "2",
        "--topL",
        "2",
        "--force-clean",
    ]
    if launch_chat:
        argv.append("--chat")

    exit_code = quickstart.main(argv)
    assert exit_code == 0
    assert calls == [("openai/gpt-oss-20b", None)]
    assert (output_dir / "sera_manifest.bin").exists()
    assert any(output_dir.glob("sera_state.*"))
    if launch_chat:
        assert chat_calls == [(output_dir.resolve(), tuple())]


def test_quickstart_missing_checkpoint(tmp_path: Path, capsys) -> None:
    argv = [
        "--checkpoint-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(tmp_path / "sera"),
    ]

    exit_code = quickstart.main(argv)
    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "sera_quickstart:" in stderr


def test_stub_safe_open_binary_blob(tmp_path: Path) -> None:
    binary_path = tmp_path / "model.safetensors"
    binary_path.write_bytes(b"\x00\x01binary")

    with pytest.raises(ModuleNotFoundError, match="pip install safetensors"):
        quickstart.sera_transfer.load_tensors(binary_path)
