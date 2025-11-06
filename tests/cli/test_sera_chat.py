from __future__ import annotations

import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import dataclasses
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


def _strip_private_fields(blob):
    if isinstance(blob, dict):
        return {
            key: _strip_private_fields(value)
            for key, value in blob.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(blob, list):
        return [_strip_private_fields(value) for value in blob]
    return blob


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
def sera_snapshot(tmp_path: Path) -> Path:
    Sera, SeraConfig = _load_sera()
    model = Sera(SeraConfig())
    snapshot = model.snapshot()
    snapshot["config"] = _config_to_dict(model.config)
    state_path = tmp_path / "sera_state.pkl"
    with state_path.open("wb") as fh:
        pickle.dump(snapshot, fh)
    return tmp_path


def test_single_turn_prompt(sera_snapshot: Path) -> None:
    script = [
        sys.executable,
        "-m",
        "gpt_oss.cli.sera_chat",
        "--artifacts",
        str(sera_snapshot),
        "--prompt",
        "hello",
        "--response-token",
        "65",
    ]
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    proc = subprocess.run(script, capture_output=True, text=True, env=env, check=True)
    assert "Sera" in proc.stdout
    assert "A" in proc.stdout
