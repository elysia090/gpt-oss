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

import gpt_oss._stubs.safetensors as safetensors_stub
from gpt_oss._stubs.safetensors.numpy import save_file
import gpt_oss.tools.sera_quickstart as quickstart
import pytest

quickstart.sera_transfer.safe_open = safetensors_stub.safe_open  # type: ignore[attr-defined]


def _create_matrix(rows: int, cols: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def _create_vector(length: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def _create_checkpoint(tmp_path: Path, *, nested: bool = False) -> Path:
    source = tmp_path / "checkpoint"
    source.mkdir()
    original = source / "original"
    original.mkdir()

    payload_root = original / "model" if nested else original
    payload_root.mkdir(parents=True, exist_ok=True)

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
    (payload_root / "config.json").write_text(json.dumps(config))

    tokenizer_stub = json.dumps({"tokenizer": "stub"})
    for filename in quickstart.TOKENIZER_FILENAMES:
        (original / filename).write_text(tokenizer_stub)

    rng = random.Random(0)
    tensors = {
        "layers.0.attn.k.weight": _create_matrix(4, 4, rng),
        "layers.0.attn.o.weight": _create_matrix(4, 4, rng),
        "layers.0.ffn.w1.weight": _create_matrix(8, 4, rng),
        "layers.0.ffn.w2.weight": _create_matrix(4, 8, rng),
        "layers.0.ffn.w1.bias": _create_vector(8, rng),
        "layers.0.ffn.w2.bias": _create_vector(4, rng),
    }
    save_file(tensors, payload_root / "model.safetensors")
    return source


def test_normalise_path_handles_resolve_oserror(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "hf-cache"
    target.mkdir()

    original_resolve = type(target).resolve

    def fake_resolve(self):  # type: ignore[no-untyped-def]
        if self == target:
            raise OSError("simulated reparse failure")
        return original_resolve(self)

    monkeypatch.setattr(type(target), "resolve", fake_resolve, raising=False)

    resolved = quickstart._normalise_path(target)
    assert resolved == target.absolute()


@pytest.mark.parametrize("nested_layout", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("launch_chat", [False, True])
def test_quickstart_pipeline(
    tmp_path: Path,
    monkeypatch,
    launch_chat: bool,
    nested_layout: bool,
) -> None:
    checkpoint_dir = _create_checkpoint(tmp_path, nested=nested_layout)
    if nested_layout:
        assert not (checkpoint_dir / "config.json").exists()
        assert not (checkpoint_dir / "model.safetensors").exists()
        nested_root = checkpoint_dir / "original" / "model"
        assert nested_root.is_dir()
        assert (nested_root / "config.json").exists()
        assert (nested_root / "model.safetensors").exists()
    else:
        assert (checkpoint_dir / "original" / "config.json").exists()
        assert (checkpoint_dir / "original" / "model.safetensors").exists()
    download_dir = tmp_path / "download"
    output_dir = tmp_path / "sera"

    calls: list[tuple[str, str | None, tuple[str, ...] | None]] = []
    chat_calls: list[tuple[Path, tuple[str, ...]]] = []

    def _fake_snapshot_download(
        repo_id: str,
        revision: str | None,
        *,
        allow_patterns: Sequence[str] | None = None,
        local_dir: str | None = None,
        local_dir_use_symlinks: bool | None = None,
    ) -> str:
        calls.append((repo_id, revision, tuple(allow_patterns) if allow_patterns is not None else None))
        assert allow_patterns == quickstart.CHECKPOINT_ALLOW_PATTERNS
        assert local_dir is not None
        assert local_dir_use_symlinks is None
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
    assert calls == [("openai/gpt-oss-20b", None, quickstart.CHECKPOINT_ALLOW_PATTERNS)]
    assert (output_dir / "sera_manifest.bin").exists()
    assert any(output_dir.glob("sera_state.*"))
    tokenizer_root = download_dir / "original"
    for filename in quickstart.TOKENIZER_FILENAMES:
        assert (tokenizer_root / filename).exists()
    if launch_chat:
        assert chat_calls == [(output_dir.resolve(), tuple())]


def test_quickstart_materialize_download(tmp_path: Path, monkeypatch) -> None:
    checkpoint_dir = _create_checkpoint(tmp_path)
    download_dir = tmp_path / "download"
    output_dir = tmp_path / "sera"

    calls: list[tuple[str, str | None, bool | None, tuple[str, ...] | None]] = []

    def _fake_snapshot_download(
        repo_id: str,
        revision: str | None,
        *,
        allow_patterns: Sequence[str] | None = None,
        local_dir: str | None = None,
        local_dir_use_symlinks: bool | None = None,
    ) -> str:
        calls.append((
            repo_id,
            revision,
            local_dir_use_symlinks,
            tuple(allow_patterns) if allow_patterns is not None else None,
        ))
        assert allow_patterns == quickstart.CHECKPOINT_ALLOW_PATTERNS
        assert local_dir is not None
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

    exit_code = quickstart.main(
        [
            "--repo-id",
            "openai/gpt-oss-20b",
            "--download-dir",
            str(download_dir),
            "--materialize-download",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("openai/gpt-oss-20b", None, False, quickstart.CHECKPOINT_ALLOW_PATTERNS)
    ]


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


def test_quickstart_uses_cache_when_download_dir_unspecified(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = _create_checkpoint(tmp_path)
    output_dir = tmp_path / "sera"
    cache_calls: list[tuple[str, str | None, dict[str, object]]] = []

    def _fake_snapshot_download(
        repo_id: str,
        revision: str | None,
        **kwargs: object,
    ) -> str:
        cache_calls.append((repo_id, revision, kwargs))
        assert kwargs.get("allow_patterns") == quickstart.CHECKPOINT_ALLOW_PATTERNS
        assert "local_dir" not in kwargs
        assert "local_dir_use_symlinks" not in kwargs
        return str(checkpoint_dir)

    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.snapshot_download = _fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_stub)
    monkeypatch.chdir(tmp_path)

    exit_code = quickstart.main([
        "--output-dir",
        str(output_dir),
        "--force-clean",
    ])

    assert exit_code == 0
    assert cache_calls == [
        (
            "openai/gpt-oss-20b",
            None,
            {"allow_patterns": quickstart.CHECKPOINT_ALLOW_PATTERNS},
        )
    ]
    assert not (tmp_path / "gpt-oss-20b").exists()
    assert (output_dir / "sera_manifest.bin").exists()
