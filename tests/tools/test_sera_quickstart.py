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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from safetensors.numpy import save_file


def _load_quickstart_module():
    module_path = ROOT / "tools" / "sera_quickstart.py"
    spec = importlib.util.spec_from_file_location("sera_quickstart", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    quickstart = _load_quickstart_module()

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
    quickstart = _load_quickstart_module()

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
