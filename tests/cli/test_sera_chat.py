from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import pickle
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from gpt_oss.cli import sera_chat


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
    payload = b"\x01\x02\x03\x04"
    header = struct.pack(
        "<I H H 5Q Q Q Q I I",
        0x53455241,
        6,
        1,
        len(payload),
        1,
        1,
        1,
        1,
        len(payload),
        0,
        0,
        1,
        0,
    )
    state_path = tmp_path / "sera_state.pkl"
    arrays_dir = tmp_path / "arrays"
    arrays_dir.mkdir()
    array_path = arrays_dir / "toy.bin"
    array_path.write_bytes(header + payload)
    snapshot["artefacts"] = {
        "toy": {"sha256": hashlib.sha256(payload).hexdigest()},
    }
    with state_path.open("wb") as fh:
        pickle.dump(snapshot, fh)
    manifest_path = tmp_path / "sera_manifest.bin"
    manifest_path.write_bytes(b"SERM")
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
        "--tool",
        "python",
        "--stats-refresh",
        "0",
    ]
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    proc = subprocess.run(script, capture_output=True, text=True, env=env, check=True)
    assert "Loaded Sera manifest" in proc.stdout
    assert "Enabled tools: python" in proc.stdout
    assert "Loaded 1 arrays" in proc.stdout
    assert "Sera: hello" in proc.stdout
    assert "Logit y_out:" in proc.stdout
    assert "tokens/sec=" in proc.stdout


def test_manifest_env_fallback(sera_manifest: Path) -> None:
    script = [
        sys.executable,
        "-m",
        "gpt_oss.cli.sera_chat",
        "--state-file",
        "sera_state.pkl",
        "--prompt",
        "ping",
        "--stats-refresh",
        "0",
    ]
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        "GPT_OSS_SERA_MANIFEST": str(sera_manifest),
    }
    proc = subprocess.run(script, capture_output=True, text=True, env=env, check=True)
    assert "Enabled tools: none" in proc.stdout
    assert "Sera: ping" in proc.stdout
    assert "tokens/sec=" in proc.stdout


def test_dashboard_formatter_snapshot() -> None:
    diag = {
        "tokens_emitted": 42,
        "bridge_hits": 7,
        "bridge_misses": 3,
        "bridge_guard_rate": 0.7,
        "trust_decision": 1,
        "trust_consistent": False,
        "trust_llr": 0.125,
        "store_load_p99": 0.93,
        "stash_occ_p99": 0.12,
        "kick_len_p99": 2.5,
        "capacity_load": 0.4,
        "capacity_slack": 0.6,
        "capacity_margin": 0.1,
        "capacity_frozen": True,
        "attention_updates": 11,
        "attention_clip_rate": 0.33,
        "attention_den_min": 0.02,
        "lambda_star": 0.9,
        "tree_simulations": 5,
        "trust_m": 4,
        "trust_gamma": 0.07,
        "trust_beta_min": 0.1,
        "trust_beta_cap": 0.9,
        "cfr_mode": "ACTIVE",
        "cfr_beta": 0.5,
        "cfr_guard": True,
        "cfr_health_ok": False,
        "cfr_y_cfr": 1.2,
    }
    output = sera_chat._format_dashboard(
        diag,
        generation=3,
        turn_tokens=6,
        verbose=True,
    )
    expected = (
        "[diag g=3] turn_tokens=6 tokens/sec=n/a total_emitted=42 bridge=7/3 (0.70) "
        "p99(store/stash/kick)=0.93/0.12/2.50 trust=1 (consistent=False) llr=0.12 "
        "capacity(load/slack/margin)=0.40/0.60/0.10 frozen=True\n"
        "  attention: updates=11 clip=0.33 min_den=0.02 lambda*=0.90 tree_sims=5\n"
        "  trust: m=4 gamma=0.07 beta=[0.10, 0.90]\n"
        "  cfr: mode=ACTIVE beta=0.50 guard=True health=False y_cfr=1.20"
    )
    assert output == expected


def test_dashboard_logs_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dash = sera_chat.DiagnosticsDashboard(
        refresh_interval=0.0,
        verbose=False,
        log_path=tmp_path / "diag.jsonl",
        _clock=lambda: 0.0,
    )
    diag = {"tokens_emitted": 1, "bridge_hits": 0, "bridge_misses": 0}
    dash.update(diagnostics=diag, generation=1, turn_tokens=1)
    dash.close()
    captured = capsys.readouterr()
    assert "[diag g=1]" in captured.out
    log_path = tmp_path / "diag.jsonl"
    payloads = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert payloads and payloads[0]["generation"] == 1
    assert payloads[0]["turn_tokens"] == 1
