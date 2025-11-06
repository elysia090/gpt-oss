from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


def _load_sera():
    sera_path = Path(__file__).resolve().parents[2] / "src" / "gpt_oss" / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_runtime", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to import Sera runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sera, module.SeraConfig


def _config_to_dict(config):
    if dataclasses.is_dataclass(config):
        result = {}
        for field in dataclasses.fields(config):
            name = field.name
            if name.startswith("_") or not field.init:
                continue
            result[name] = _config_to_dict(getattr(config, name))
        return result
    if isinstance(config, dict):
        return {key: _config_to_dict(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_config_to_dict(value) for value in config]
    return config


@pytest.fixture()
def sera_manifest(tmp_path: Path) -> Path:
    Sera, SeraConfig = _load_sera()
    model = Sera(SeraConfig())
    snapshot = model.snapshot()
    snapshot["config"] = _config_to_dict(model.config)
    state_path = tmp_path / "sera_state.pkl"
    with state_path.open("wb") as fh:
        pickle.dump(snapshot, fh)
    manifest_path = tmp_path / "sera_manifest.bin"
    manifest_path.write_bytes(b"SERM")
    (tmp_path / "arrays").mkdir()
    return tmp_path


def test_single_turn_prompt(sera_manifest: Path) -> None:
    script = [
        sys.executable,
        "-m",
        "gpt_oss.cli.sera_chat",
        "--manifest",
        str(sera_manifest),
        "--prompt",
        "hello",
        "--response-token",
        "65",
        "--tool",
        "python",
    ]
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    proc = subprocess.run(script, capture_output=True, text=True, env=env, check=True)
    assert "Loaded Sera manifest" in proc.stdout
    assert "Enabled tools: python" in proc.stdout
    assert "Sera" in proc.stdout
    assert "A" in proc.stdout
