import io
import json
from pathlib import Path

import pytest

from gpt_oss.cli.sera_chat import (
    DiagnosticsDashboard,
    _execute_turn,
    _format_dashboard,
    _format_metrics_line,
)
from gpt_oss.cli.sera_tui import SeraTUI


class FauxTokenizer:
    def encode(self, value: bytes) -> list[int]:
        return [c for c in value]

    def decode(self, tokens: list[int]) -> str:
        return bytes(int(token) for token in tokens).decode("utf-8")


class FauxSera:
    def __init__(self) -> None:
        self.tokenizer = FauxTokenizer()
        self.generation = 2
        self._diagnostics = {
            "trust_decision": 1,
            "trust_llr": 0.5,
            "latency_ms": 10.0,
            "tokens_per_sec": 120.0,
            "tokens_emitted": 8,
        }

    def step(self, *, bytes_data: bytes) -> dict[str, object]:
        text = bytes_data.decode("utf-8")
        tokens = [ord(char) for char in text]
        response = "diag-ok"
        generated = [ord(char) for char in response]
        return {"tokens": tokens, "generated": generated, "y_out": 0.25}

    def diagnostics_record(self) -> dict[str, object]:
        return dict(self._diagnostics)


@pytest.fixture()
def faux_model() -> FauxSera:
    return FauxSera()


def test_tui_updates_and_commands(tmp_path: Path, faux_model: FauxSera) -> None:
    manifest_dir = tmp_path
    manifest_state_path = manifest_dir / "sera_state.pkl"
    manifest_state_path.write_text("placeholder", encoding="utf-8")

    dashboard = DiagnosticsDashboard(
        refresh_interval=0.0,
        verbose=False,
        log_path=None,
        stream=io.StringIO(),
    )

    tui = SeraTUI(
        model=faux_model,
        dashboard=dashboard,
        execute_turn=lambda prompt: _execute_turn(faux_model, prompt),
        diagnostics_formatter=lambda diagnostics, generation, turn_tokens: _format_dashboard(
            diagnostics,
            generation=generation,
            turn_tokens=turn_tokens,
            verbose=False,
        ),
        metrics_formatter=_format_metrics_line,
        manifest_dir=manifest_dir,
        manifest_state_path=manifest_state_path,
        manifest_state={"config": {"layers": 1}},
        manifest_arrays={"demo": {"dtype": "f32", "shape": (1, 2)}},
        initial_tools=["browser"],
        optional_tools=("browser", "python"),
        load_messages=["Loaded Sera manifest"],
        metrics_mode="plain",
        transcript_logger=None,
    )

    app = tui._ensure_application()
    assert app is not None
    assert tui._operations_area is not None
    assert "browser" in tui._operations_area.text

    turn = tui.handle_prompt("hello")
    assert turn.prompt == "hello"
    assert any(line.startswith("Sera:") for line in tui._chat_transcript)
    assert "[metrics]" in tui._chat_transcript[-1]
    assert tui._diagnostics_area is not None
    assert "latency_ms" in tui._diagnostics_area.text

    tui._toggle_tool("python")
    assert "python" in tui._operations_area.text
    assert "ON" in tui._operations_area.text

    tui.view_manifest_metadata()
    assert any(line.startswith("[manifest]") for line in tui._chat_transcript)

    export_path = tui.export_diagnostics()
    assert export_path is not None
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["diagnostics"]["trust_decision"] == 1
    assert sorted(payload["tools"]) == ["browser", "python"]

    transcript_path = tui.export_transcript()
    assert transcript_path is not None
    transcript_data = transcript_path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("You:") for line in transcript_data)
