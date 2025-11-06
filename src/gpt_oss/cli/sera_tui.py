"""Interactive terminal UI for the Sera chat helper."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - executed at import time
    from prompt_toolkit.application import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, VSplit, Layout
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.widgets import Frame, TextArea
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without prompt_toolkit
    class Application:  # type: ignore[override]
        def __init__(self, layout, full_screen=False, key_bindings=None):
            self.layout = layout
            self.full_screen = full_screen
            self.key_bindings = key_bindings

        def run(self) -> None:
            raise RuntimeError(
                "prompt_toolkit is required to run the Sera TUI. Install the optional dependency to enable it."
            )

        def invalidate(self) -> None:
            pass

    class Buffer:  # type: ignore[override]
        def __init__(self, accept_handler=None):
            self.accept_handler = accept_handler
            self.text = ""
            self.cursor_position = 0

        def reset(self) -> None:
            self.text = ""
            self.cursor_position = 0

    class TextArea:  # type: ignore[override]
        def __init__(self, text="", buffer: Optional[Buffer] = None, **_: object) -> None:
            self.buffer = buffer or Buffer()
            self.text = text
            self.buffer.text = text

    class Frame:  # type: ignore[override]
        def __init__(self, body, title: Optional[str] = None, height: Optional[object] = None):
            self.body = body
            self.title = title
            self.height = height

    class Dimension:  # type: ignore[override]
        def __init__(self, min: Optional[int] = None, preferred: Optional[int] = None):
            self.min = min
            self.preferred = preferred

    class HSplit(list):  # type: ignore[override]
        def __init__(self, children):
            super().__init__(children)

    class VSplit(list):  # type: ignore[override]
        def __init__(self, children, padding: int | None = None):
            super().__init__(children)
            self.padding = padding

    class KeyBindings:  # type: ignore[override]
        def __init__(self) -> None:
            self._bindings: Dict[str, List[Callable[..., None]]] = {}

        def add(self, key: str):  # type: ignore[override]
            def decorator(func: Callable[..., None]) -> Callable[..., None]:
                self._bindings.setdefault(key, []).append(func)
                return func

            return decorator

    class Layout:  # type: ignore[override]
        def __init__(self, container, focused_element=None):
            self.container = container
            self.focused_element = focused_element

from .sera_chat import TurnRecord

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from .sera_chat import DiagnosticsDashboard


@dataclass
class _ManifestSummary:
    directory: Path
    state_path: Path
    arrays: Mapping[str, Mapping[str, object]]
    state: Mapping[str, object] | Sequence[object] | None

    def render(self, max_arrays: int = 6) -> List[str]:
        lines = [f"Manifest: {self.directory}", f"Snapshot: {self.state_path.name}"]
        if self.arrays:
            lines.append(f"Arrays: {len(self.arrays)} total")
            for name, info in list(sorted(self.arrays.items()))[:max_arrays]:
                dtype = info.get("dtype", "?")
                shape = info.get("shape", "?")
                lines.append(f"  - {name}: {dtype}@{shape}")
            if len(self.arrays) > max_arrays:
                lines.append("  - …")
        if isinstance(self.state, Mapping):
            keys = list(sorted(str(key) for key in self.state.keys()))
            if keys:
                preview = ", ".join(keys[:6])
                if len(keys) > 6:
                    preview += ", …"
                lines.append(f"Runtime keys: {preview}")
        return lines


class SeraTUI:
    """Prompt-toolkit powered TUI for ``gpt-oss-sera-chat``."""

    _OPERATIONS_PANEL_WIDTH = 32

    def __init__(
        self,
        *,
        model,
        dashboard: "DiagnosticsDashboard",
        execute_turn: Callable[[str], TurnRecord],
        diagnostics_formatter: Callable[[Mapping[str, object], int, int], str],
        metrics_formatter: Callable[[Mapping[str, object]], str],
        manifest_dir: Path,
        manifest_state_path: Path,
        manifest_state,
        manifest_arrays: Mapping[str, Mapping[str, object]] | None,
        initial_tools: Iterable[str],
        optional_tools: Sequence[str],
        load_messages: Sequence[str],
        metrics_mode: str,
    ) -> None:
        self.model = model
        self.dashboard = dashboard
        self.execute_turn = execute_turn
        self.diagnostics_formatter = diagnostics_formatter
        self.metrics_formatter = metrics_formatter
        self.manifest_summary = _ManifestSummary(
            directory=manifest_dir,
            state_path=manifest_state_path,
            arrays=manifest_arrays or {},
            state=manifest_state
            if isinstance(manifest_state, Mapping)
            or (
                isinstance(manifest_state, Sequence)
                and not isinstance(manifest_state, (bytes, bytearray, str))
            )
            else None,
        )
        self.metrics_mode = metrics_mode
        self._enabled_tools = set(initial_tools)
        self._optional_tools = list(optional_tools)
        self._load_messages = list(load_messages)
        self._app: Optional[Application] = None
        self._chat_area: Optional[TextArea] = None
        self._diagnostics_area: Optional[TextArea] = None
        self._operations_area: Optional[TextArea] = None
        self._input_area: Optional[TextArea] = None
        self._chat_transcript: List[str] = [f"[system] {msg}" for msg in self._load_messages]
        self._last_diagnostics: Optional[Mapping[str, object]] = None
        self._last_metrics: Optional[Mapping[str, object]] = None
        self._last_export_path: Optional[Path] = None
        self._tool_key_map = self._build_tool_key_map(self._optional_tools)

    def _build_tool_key_map(self, tools: Sequence[str]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for index, name in enumerate(tools, start=2):
            mapping[f"f{index}"] = name
        return mapping

    # ------------------------------------------------------------------ UI glue
    def _ensure_application(self) -> Application:
        if self._app is not None:
            return self._app

        chat_area = TextArea(text="\n".join(self._chat_transcript), scrollbar=True, wrap_lines=True, read_only=True)
        diagnostics_area = TextArea(text="Awaiting diagnostics…", height=Dimension(min=3), read_only=True)
        operations_area = TextArea(read_only=True, width=Dimension(min=24, preferred=32))
        input_buffer = Buffer(accept_handler=self._handle_submit)
        input_area = TextArea(
            height=1,
            prompt="You: ",
            multiline=False,
            wrap_lines=False,
            buffer=input_buffer,
        )

        body = HSplit(
            [
                VSplit(
                    [
                        Frame(chat_area, title="Conversation"),
                        Frame(operations_area, title="Operations"),
                    ],
                    padding=1,
                ),
                Frame(diagnostics_area, title="Diagnostics", height=Dimension(preferred=6)),
                input_area,
            ]
        )

        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def _(event) -> None:  # pragma: no cover - interactive exit
            event.app.exit()

        for key, tool in self._tool_key_map.items():
            @kb.add(key)
            def _(event, tool_name=tool) -> None:  # type: ignore[misc]
                self._toggle_tool(tool_name)

        @kb.add("f9")
        def _(event) -> None:
            self.view_manifest_metadata()

        @kb.add("f10")
        def _(event) -> None:
            self.export_diagnostics()

        @kb.add("escape")
        def _(event) -> None:  # pragma: no cover - interactive exit
            event.app.exit()

        self._chat_area = chat_area
        self._diagnostics_area = diagnostics_area
        self._operations_area = operations_area
        self._input_area = input_area
        self._refresh_operations()

        self._app = Application(layout=Layout(body, focused_element=input_area), full_screen=True, key_bindings=kb)
        return self._app

    def run(self) -> None:
        app = self._ensure_application()
        app.run()

    # ----------------------------------------------------------------- Handlers
    def _handle_submit(self, buff: Buffer) -> bool:
        text = buff.text.strip()
        buff.reset()
        if not text:
            return False
        self.handle_prompt(text)
        return False

    def handle_prompt(self, prompt: str) -> TurnRecord:
        self._append_chat_line(f"You: {prompt}")
        turn = self.execute_turn(prompt)
        rendered_response = "".join(turn.response_fragments) or turn.response_text
        self._append_chat_line(f"Sera: {rendered_response}")
        self.dashboard.update(turn.diagnostics, generation=turn.generation, turn_tokens=turn.turn_tokens)
        self._last_diagnostics = turn.diagnostics
        self._last_metrics = turn.metrics_payload
        self._update_diagnostics_display(turn)
        if self.metrics_mode == "plain":
            self._append_chat_line(self.metrics_formatter(turn.metrics_payload))
        elif self.metrics_mode == "json":
            json_payload = json.dumps(turn.metrics_payload, sort_keys=True)
            self._append_chat_line(f"[metrics] {json_payload}")
        return turn

    # ----------------------------------------------------------------- Actions
    def _append_chat_line(self, line: str) -> None:
        self._chat_transcript.append(line)
        if self._chat_area is not None:
            new_text = "\n".join(self._chat_transcript)
            self._chat_area.text = new_text
            if hasattr(self._chat_area, "buffer"):
                self._chat_area.buffer.text = new_text
                self._chat_area.buffer.cursor_position = len(new_text)
            self._invalidate()

    def _update_diagnostics_display(self, turn: TurnRecord) -> None:
        if self._diagnostics_area is None:
            return
        diagnostics_text = self.diagnostics_formatter(turn.diagnostics, turn.generation, turn.turn_tokens)
        lines = [diagnostics_text]
        if self.metrics_mode == "plain":
            lines.append(self.metrics_formatter(turn.metrics_payload))
        elif self.metrics_mode == "json":
            lines.append("[metrics] " + json.dumps(turn.metrics_payload, sort_keys=True))
        new_text = "\n".join(lines)
        self._diagnostics_area.text = new_text
        if hasattr(self._diagnostics_area, "buffer"):
            self._diagnostics_area.buffer.text = new_text
        self._invalidate()

    def _toggle_tool(self, tool: str) -> None:
        if tool in self._enabled_tools:
            self._enabled_tools.remove(tool)
            status = "disabled"
        else:
            self._enabled_tools.add(tool)
            status = "enabled"
        self._append_chat_line(f"[system] Tool '{tool}' {status}.")
        self._refresh_operations()

    def view_manifest_metadata(self) -> None:
        for line in self.manifest_summary.render():
            self._append_chat_line(f"[manifest] {line}")

    def export_diagnostics(self) -> Optional[Path]:
        if not self._last_diagnostics:
            self._append_chat_line("[system] No diagnostics to export yet.")
            return None
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        export_path = self.manifest_summary.directory / f"sera_diagnostics_{timestamp}.json"
        payload = {
            "diagnostics": self._last_diagnostics,
            "metrics": self._last_metrics or {},
            "tools": sorted(self._enabled_tools),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        export_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        self._last_export_path = export_path
        self._append_chat_line(f"[system] Diagnostics exported to {export_path}")
        self._refresh_operations()
        return export_path

    # ---------------------------------------------------------------- Utilities
    def _refresh_operations(self) -> None:
        if self._operations_area is None:
            return
        lines: List[str] = []
        lines.extend(self._format_tools_section())
        lines.append("")
        lines.append("Hotkeys:")
        lines.extend(self._format_hotkey_lines())
        metadata_lines = self._format_metadata_lines()
        if metadata_lines:
            lines.append("")
            lines.extend(metadata_lines)
        new_text = "\n".join(lines)
        self._operations_area.text = new_text
        if hasattr(self._operations_area, "buffer"):
            self._operations_area.buffer.text = new_text
            self._operations_area.buffer.cursor_position = 0
        self._invalidate()

    def _format_tools_section(self) -> List[str]:
        if not self._enabled_tools:
            return ["Tools: none"]
        lines = ["Tools:"]
        for tool in sorted(self._enabled_tools):
            lines.append(f"  • {tool}")
        return lines

    def _format_hotkey_lines(self) -> List[str]:
        entries: List[tuple[str, str, str]] = []
        for key, tool in sorted(self._tool_key_map.items()):
            status = "ON" if tool in self._enabled_tools else "OFF"
            entries.append((key.upper(), f"Toggle {tool}", f"({status})"))
        entries.extend(
            [
                ("F9", "View manifest metadata", ""),
                ("F10", "Export diagnostics", ""),
                ("Ctrl+C", "Exit", ""),
            ]
        )
        key_width = max((len(entry[0]) for entry in entries), default=0)
        lines: List[str] = []
        for key, description, suffix in entries:
            padded_key = key.ljust(key_width)
            line = f"  [{padded_key}] {description}"
            if suffix:
                line = f"{line} {suffix}"
            lines.append(line)
        return lines

    def _format_metadata_lines(self) -> List[str]:
        lines: List[str] = []
        if self.metrics_mode != "off":
            lines.append(f"Metrics mode: {self.metrics_mode}")
        lines.extend(
            self._wrap_label_value("Manifest dir:", str(self.manifest_summary.directory))
        )
        if self._last_export_path is not None:
            lines.extend(self._wrap_label_value("Last export:", str(self._last_export_path)))
        return lines

    def _wrap_label_value(self, label: str, value: str) -> List[str]:
        width = max(8, self._OPERATIONS_PANEL_WIDTH - len(label) - 1)
        wrapped = textwrap.wrap(
            value,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            return [label.rstrip()]
        first, *rest = wrapped
        lines = [f"{label} {first}"]
        indent = " " * (len(label) + 1)
        for chunk in rest:
            lines.append(f"{indent}{chunk}")
        return lines

    def _invalidate(self) -> None:
        if self._app is not None:
            self._app.invalidate()


__all__ = ["SeraTUI"]
