"""Terminal chat helper for Sera manifest snapshots.

This module provides a small CLI that restores a :class:`~gpt_oss.inference.sera.Sera`
instance from the lightweight manifest bundle produced by
``gpt_oss.tools.sera_transfer``.  The bundle contains the manifest itself and a
runtime snapshot (``sera_state.pkl`` or ``sera_state.json``) that can be handed
to :meth:`gpt_oss.inference.sera.Sera.restore`.  Once restored we expose a
minimal REPL similar to :mod:`gpt_oss.interfaces.cli.chat` that simply echoes
the prompt tokens and emits a placeholder response token so callers can verify
that the runtime is wired correctly.

Only a subset of the tool toggles from the main chat interface are supported.
They act as a UX hint and are printed so that downstream wrappers can make a
decision on which integrations to enable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
import hashlib
import io
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Dict,
    List,
    TextIO,
)

if TYPE_CHECKING:  # pragma: no cover - import only used for type checkers
    from gpt_oss.inference.sera import Sera


def _load_sera_common_module():
    sera_common_path = Path(__file__).resolve().parents[1] / "inference" / "sera_common.py"
    spec = importlib.util.spec_from_file_location("_sera_common", sera_common_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_sera_common = _load_sera_common_module()
DEFAULT_STATE_FILENAMES = _sera_common.DEFAULT_STATE_FILENAMES
JSON_BYTES_PREFIX = _sera_common.JSON_BYTES_PREFIX
SeraArrayError = _sera_common.SeraArrayError
decode_snapshot_blob = _sera_common.decode_snapshot_blob
load_array_file = _sera_common.load_array_file

OPTIONAL_TOOLS: Sequence[str] = ("browser", "python")


def _format_float(value: Optional[object]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return str(value)


def _format_int(value: Optional[object]) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return "0"


def _coerce_number(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return None


def _format_metrics_line(metrics: Mapping[str, object]) -> str:
    latency = _format_float(metrics.get("latency_ms"))
    tokens_per_sec = _format_float(metrics.get("tokens_per_sec"))
    trust_decision = metrics.get("trust_decision", 0)
    trust_llr = _format_float(metrics.get("trust_llr"))
    turn_tokens = _format_int(metrics.get("turn_tokens"))
    return (
        f"[metrics] latency_ms={latency} tokens/sec={tokens_per_sec} "
        f"trust={trust_decision} llr={trust_llr} turn_tokens={turn_tokens}"
    )


def _format_dashboard(
    diagnostics: Mapping[str, object],
    *,
    generation: int,
    turn_tokens: int,
    verbose: bool,
) -> str:
    """Render a compact diagnostic dashboard for the CLI."""

    bridge_hits = int(diagnostics.get("bridge_hits", 0) or 0)
    bridge_misses = int(diagnostics.get("bridge_misses", 0) or 0)
    bridge_guard = diagnostics.get("bridge_guard_rate", 0.0) or 0.0
    trust_decision = diagnostics.get("trust_decision", 0)
    trust_consistent = diagnostics.get("trust_consistent", True)
    trust_llr = diagnostics.get("trust_llr", 0.0)
    tokens_emitted = diagnostics.get("tokens_emitted", 0)
    latency_ms = diagnostics.get("latency_ms")
    store_p99 = diagnostics.get("store_load_p99")
    stash_p99 = diagnostics.get("stash_occ_p99")
    kick_p99 = diagnostics.get("kick_len_p99")
    capacity_load = diagnostics.get("capacity_load")
    capacity_slack = diagnostics.get("capacity_slack")
    capacity_margin = diagnostics.get("capacity_margin")
    capacity_frozen = diagnostics.get("capacity_frozen", False)

    tokens_per_sec = diagnostics.get("tokens_per_sec")

    summary = (
        f"[diag g={generation}] turn_tokens={turn_tokens} "
        f"tokens/sec={_format_float(tokens_per_sec)} "
        f"latency_ms={_format_float(latency_ms)} "
        f"total_emitted={_format_int(tokens_emitted)} "
        f"bridge={bridge_hits}/{bridge_misses} ({_format_float(bridge_guard)}) "
        f"p99(store/stash/kick)={_format_float(store_p99)}/"
        f"{_format_float(stash_p99)}/{_format_float(kick_p99)} "
        f"trust={trust_decision} (consistent={trust_consistent}) llr={_format_float(trust_llr)} "
        f"capacity(load/slack/margin)={_format_float(capacity_load)}/"
        f"{_format_float(capacity_slack)}/{_format_float(capacity_margin)} frozen={capacity_frozen}"
    )

    if not verbose:
        return summary

    attention_updates = diagnostics.get("attention_updates", 0)
    attention_clip = diagnostics.get("attention_clip_rate", 0.0)
    attention_den = diagnostics.get("attention_den_min")
    lambda_star = diagnostics.get("lambda_star")
    tree_simulations = diagnostics.get("tree_simulations", 0)
    trust_m = diagnostics.get("trust_m", 0)
    trust_gamma = diagnostics.get("trust_gamma", 0.0)
    trust_beta_min = diagnostics.get("trust_beta_min", 0.0)
    trust_beta_cap = diagnostics.get("trust_beta_cap", 0.0)
    corrector_mode = diagnostics.get("corrector_mode", "OFF")
    corrector_beta = diagnostics.get("corrector_beta", 0.0)
    corrector_guard = diagnostics.get("corrector_guard", False)
    corrector_health = diagnostics.get("corrector_health_ok", True)
    corrector_y = diagnostics.get("corrector_y_corrector", 0.0)

    details = [summary]
    details.append(
        "  attention: updates="
        + _format_int(attention_updates)
        + f" clip={_format_float(attention_clip)} min_den={_format_float(attention_den)}"
        + f" lambda*={_format_float(lambda_star)} tree_sims={_format_int(tree_simulations)}"
    )
    details.append(
        "  trust: m="
        + _format_int(trust_m)
        + f" gamma={_format_float(trust_gamma)} beta=[{_format_float(trust_beta_min)},"
        + f" {_format_float(trust_beta_cap)}]"
    )
    details.append(
        "  corrector: mode="
        + str(corrector_mode)
        + f" beta={_format_float(corrector_beta)} guard={corrector_guard}"
        + f" health={corrector_health}"
        + f" y={_format_float(corrector_y)}"
    )
    return "\n".join(details)


@dataclass
class DiagnosticsDashboard:
    refresh_interval: float
    verbose: bool
    log_path: Optional[Path] = None
    stream: Optional[TextIO] = None
    _clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.refresh_interval = max(0.0, float(self.refresh_interval))
        self._last_render: float = 0.0
        self._log_file = None
        self._last_line_len = 0
        self._stream = self.stream if self.stream is not None else sys.stdout
        if self.log_path is not None:
            log_path = self.log_path.expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self.refresh_interval > 0.0 and self._last_line_len and self._stream is not None:
            self._stream.write("\n")
            self._stream.flush()
            self._last_line_len = 0
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _should_render(self) -> bool:
        now = self._clock()
        if self._last_render == 0.0 or now - self._last_render >= self.refresh_interval:
            self._last_render = now
            return True
        return False

    def update(
        self,
        diagnostics: Mapping[str, object],
        *,
        generation: int,
        turn_tokens: int,
    ) -> None:
        if self._log_file is not None:
            payload = dict(diagnostics)
            payload.update(
                {
                    "generation": generation,
                    "turn_tokens": turn_tokens,
                    "ts": time.time(),
                }
            )
            json.dump(payload, self._log_file)
            self._log_file.write("\n")
            self._log_file.flush()

        if self._should_render():
            line = _format_dashboard(
                diagnostics,
                generation=generation,
                turn_tokens=turn_tokens,
                verbose=self.verbose,
            )
            if self._stream is None:
                return
            if self.refresh_interval <= 0.0:
                self._stream.write(line + "\n")
                self._stream.flush()
            else:
                padding = max(0, self._last_line_len - len(line))
                self._stream.write("\r" + line + (" " * padding))
                self._stream.flush()
                self._last_line_len = len(line)


@dataclass
class TurnRecord:
    prompt: str
    user_text: str
    user_tokens: int
    response_text: str
    response_fragments: List[str]
    response_tokens: List[int]
    diagnostics: Dict[str, object]
    metrics_payload: Dict[str, object]
    generation: int
    turn_tokens: int
    y_out: Optional[float]


@dataclass
class TranscriptLogger:
    """Persist chat transcripts for later inspection."""

    path: Path
    include_diagnostics: bool = True

    def __post_init__(self) -> None:
        self.path = self.path.expanduser()
        parent = self.path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def record_event(self, message: str, *, role: str = "system", **extra: object) -> None:
        payload: Dict[str, object] = {
            "type": "event",
            "role": role,
            "message": message,
        }
        if extra:
            payload.update(extra)
        self._write(payload)

    def record_turn(
        self,
        prompt: str,
        turn: TurnRecord,
        *,
        tools: Iterable[str] = (),
    ) -> None:
        payload: Dict[str, object] = {
            "type": "turn",
            "prompt": prompt,
            "prompt_tokens": turn.user_tokens,
            "response_text": turn.response_text,
            "response_fragments": list(turn.response_fragments),
            "response_tokens": list(turn.response_tokens),
            "metrics": dict(turn.metrics_payload),
            "generation": turn.generation,
            "turn_tokens": turn.turn_tokens,
            "y_out": turn.y_out,
            "tools": sorted(set(tools)),
        }
        if self.include_diagnostics:
            payload["diagnostics"] = dict(turn.diagnostics)
        self._write(payload)

    def _write(self, payload: Dict[str, object]) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")


def _load_sera_class():
    sera_path = Path(__file__).resolve().parents[1] / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_runtime", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sera, getattr(module, "SeraConfig")


def _strip_private_fields(blob):
    """Remove keys that start with an underscore from nested structures."""

    if isinstance(blob, dict):
        return {
            key: _strip_private_fields(value)
            for key, value in blob.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(blob, list):
        return [_strip_private_fields(value) for value in blob]
    return blob


def _decode_snapshot_types(blob):
    return decode_snapshot_blob(blob)


def _decode_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.hex()
    return str(value)


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_manifest_dir(path: Optional[str]) -> Path:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get("GPT_OSS_SERA_MANIFEST")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd())

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved / "sera_manifest.bin").exists():
            return resolved
    raise FileNotFoundError(
        "Unable to locate a Sera manifest directory. Provide --manifest or set "
        "GPT_OSS_SERA_MANIFEST to a directory containing sera_manifest.bin."
    )


def _locate_state_file(manifest_dir: Path, override: Optional[str]) -> Path:
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = manifest_dir / override_path
        override_path = override_path.resolve()
        if not override_path.exists():
            raise FileNotFoundError(f"State file {override_path} not found")
        return override_path

    candidates = [manifest_dir / name for name in DEFAULT_STATE_FILENAMES]
    state_path = _first_existing(*candidates)
    if state_path is None:
        raise FileNotFoundError(
            "No Sera runtime snapshot found alongside the manifest. Expected one "
            "of: " + ", ".join(DEFAULT_STATE_FILENAMES)
        )
    return state_path


def _load_state(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            return pickle.load(fh)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return _decode_snapshot_types(blob)
    if suffix in {".msgpack", ".mpk"}:
        if importlib.util.find_spec("msgpack") is None:  # pragma: no cover - deterministic import guard
            raise RuntimeError("Support for msgpack snapshots requires the 'msgpack' package")
        import msgpack  # type: ignore
        with path.open("rb") as fh:
            blob = msgpack.unpack(fh, raw=False)
        return _decode_snapshot_types(blob)
    raise RuntimeError(f"Unsupported snapshot format: {path}")


def _load_manifest_arrays(
    manifest_dir: Path, artefacts: Optional[Mapping[str, Mapping[str, object]]]
) -> Dict[str, Dict[str, object]]:
    if not artefacts or not isinstance(artefacts, Mapping):
        return {}
    arrays_dir = manifest_dir / "arrays"
    loaded: Dict[str, Dict[str, object]] = {}
    for name, record in sorted(artefacts.items()):
        array_path = arrays_dir / f"{name}.bin"
        if not array_path.exists():
            raise FileNotFoundError(f"Required array {array_path} is missing")
        try:
            header, payload = load_array_file(array_path)
        except SeraArrayError as exc:
            raise ValueError(str(exc)) from exc
        expected_sha = record.get("sha256") if isinstance(record, Mapping) else None
        if expected_sha:
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected_sha:
                raise ValueError(
                    f"Array {array_path} failed SHA-256 validation: "
                    f"expected {expected_sha}, got {digest}"
                )
        loaded[name] = {
            "dtype": header.dtype,
            "shape": header.shape,
            "byte_len": header.byte_len,
        }
    return loaded


def _execute_turn(
    model: "Sera",
    prompt: str,
) -> TurnRecord:
    start = time.perf_counter()
    result = model.step(bytes_data=prompt.encode("utf-8"))
    elapsed = max(time.perf_counter() - start, 1e-6)

    tokens = result.get("tokens", [])
    try:
        user_text = model.tokenizer.decode(tokens)
    except Exception:  # pragma: no cover - defensive against tokenizer errors
        user_text = b""
    user_text = _decode_text(user_text)

    generated_ids = result.get("generated") or result.get("response_tokens")
    if not generated_ids:
        generated_ids = tokens

    print(f"User ({len(tokens)} tokens): {user_text}")
    diagnostics = dict(model.diagnostics_record())
    turn_tokens = len(generated_ids)
    diagnostics["tokens_per_sec"] = turn_tokens / elapsed if turn_tokens else 0.0
    diagnostics["latency_ms"] = elapsed * 1000.0

    trust_decision = diagnostics.get("trust_decision", 0)
    trust_llr = diagnostics.get("trust_llr")
    latency_text = _format_float(diagnostics.get("latency_ms"))
    tokens_per_sec_text = _format_float(diagnostics.get("tokens_per_sec"))
    response_text = (
        "trust="
        + str(trust_decision)
        + f" llr={_format_float(trust_llr)} latency_ms={latency_text}"
        + f" tokens/sec={tokens_per_sec_text}"
    )

    response_tokens = []
    try:
        response_tokens = model.tokenizer.encode(response_text.encode("utf-8"))
    except Exception:  # pragma: no cover - fallback if tokenizer encode fails
        response_tokens = [ord(ch) for ch in response_text]

    response_fragments: List[str] = []
    for token_id in response_tokens:
        try:
            fragment = model.tokenizer.decode([int(token_id)])
        except Exception:  # pragma: no cover - defensive against tokenizer errors
            fragment = bytes([int(token_id) & 0xFF])
        response_fragments.append(_decode_text(fragment))

    metrics_payload = {
        "latency_ms": _coerce_number(diagnostics.get("latency_ms")),
        "tokens_per_sec": _coerce_number(diagnostics.get("tokens_per_sec")),
        "trust_decision": int(trust_decision) if isinstance(trust_decision, (int, float)) else trust_decision,
        "trust_llr": _coerce_number(trust_llr),
        "turn_tokens": int(turn_tokens),
    }

    return TurnRecord(
        prompt=prompt,
        user_text=user_text,
        user_tokens=len(tokens),
        response_text=response_text,
        response_fragments=response_fragments,
        response_tokens=[int(token) for token in response_tokens],
        diagnostics=diagnostics,
        metrics_payload=metrics_payload,
        generation=getattr(model, "generation", 0),
        turn_tokens=turn_tokens,
        y_out=_coerce_number(result.get("y_out")),
    )


def _run_turn(
    model: "Sera",
    prompt: str,
    dashboard: DiagnosticsDashboard,
    *,
    metrics_mode: str = "off",
) -> TurnRecord:
    turn = _execute_turn(model, prompt)

    print(f"User ({turn.user_tokens} tokens): {turn.user_text}")
    print("Sera: ", end="", flush=True)
    for fragment in turn.response_fragments:
        sys.stdout.write(fragment)
        sys.stdout.flush()
    print()

    if turn.y_out is not None:
        print(f"Logit y_out: {_format_float(turn.y_out)}")

    dashboard.update(
        turn.diagnostics,
        generation=turn.generation,
        turn_tokens=turn.turn_tokens,
    )

    if metrics_mode == "plain":
        print(_format_metrics_line(turn.metrics_payload))
    elif metrics_mode == "json":
        json_payload = {k: v for k, v in turn.metrics_payload.items()}
        print("[metrics] " + json.dumps(json_payload, sort_keys=True))

    return turn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive chat using Sera manifest artefacts"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Directory containing sera_manifest.bin and runtime snapshot",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        help="Explicit path (relative to manifest dir) for the runtime snapshot",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional prompt to run in single-shot mode and exit",
    )
    parser.add_argument(
        "--tool",
        action="append",
        dest="tools",
        default=[],
        choices=sorted(OPTIONAL_TOOLS),
        help="Enable optional tools exposed in the interface",
    )
    parser.add_argument(
        "--stats-refresh",
        type=float,
        default=0.5,
        help="Minimum seconds between diagnostic dashboard updates",
    )
    parser.add_argument(
        "--verbose-stats",
        action="store_true",
        help="Include extended diagnostic details in the dashboard",
    )
    parser.add_argument(
        "--diagnostic-log",
        type=str,
        help="Write diagnostics to the given JSONL file for scripting",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        help="Record a JSONL transcript of the session to the given path",
    )
    parser.add_argument(
        "--metrics",
        choices=("off", "plain", "json"),
        default="off",
        help="Render per-turn metrics after streaming tokens",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive terminal UI (default mode)",
    )
    mode_group.add_argument(
        "--plain",
        action="store_true",
        help="Use the legacy plain-text interface",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    use_tui = True
    if getattr(args, "plain", False):
        use_tui = False
    elif getattr(args, "tui", False):
        use_tui = True

    manifest_dir = _resolve_manifest_dir(args.manifest)
    state_path = _locate_state_file(manifest_dir, args.state_file)
    state_blob = _strip_private_fields(_load_state(state_path))
    runtime_blob = state_blob.get("sera_snapshot")
    if isinstance(runtime_blob, Mapping):
        runtime_blob = _strip_private_fields(runtime_blob)
    else:
        runtime_blob = state_blob

    arrays = _load_manifest_arrays(manifest_dir, state_blob.get("artefacts"))
    if arrays:
        state_blob.setdefault("arrays", arrays)

    Sera, SeraConfig = _load_sera_class()
    if "config" in runtime_blob:
        model = Sera.restore(runtime_blob)
    else:
        if SeraConfig is None:  # pragma: no cover - defensive fallback
            raise RuntimeError("Sera runtime does not expose SeraConfig")
        config_blob = state_blob.get("model_config")
        if isinstance(config_blob, Mapping):
            config = SeraConfig(**config_blob)
        else:
            config = SeraConfig()
        model = Sera(config)

    load_messages = [f"Loaded Sera manifest from {manifest_dir}"]
    dashboard_stream = io.StringIO() if use_tui else None
    dashboard = DiagnosticsDashboard(
        refresh_interval=args.stats_refresh,
        verbose=args.verbose_stats,
        log_path=Path(args.diagnostic_log).expanduser() if args.diagnostic_log else None,
        stream=dashboard_stream,
    )

    transcript_logger = (
        TranscriptLogger(Path(args.transcript)) if args.transcript else None
    )

    enabled_tools = sorted(set(args.tools or []))
    if arrays:
        preview = ", ".join(
            f"{name}:{info['dtype']}@{info['shape']}"
            for name, info in list(arrays.items())[:3]
        )
        if len(arrays) > 3:
            preview += ", ..."
        load_messages.append(
            f"Loaded {len(arrays)} arrays from {manifest_dir / 'arrays'}"
            + (f" [{preview}]" if preview else "")
        )
    if enabled_tools:
        load_messages.append("Enabled tools: " + ", ".join(enabled_tools))
    else:
        load_messages.append("Enabled tools: none")

    if args.prompt is not None:
        use_tui = False

    try:
        if not use_tui:
            if transcript_logger:
                for message in load_messages:
                    transcript_logger.record_event(message, role="system")
            for line in load_messages:
                print(line)
            if args.prompt is not None:
                turn = _run_turn(
                    model,
                    args.prompt,
                    dashboard,
                    metrics_mode=args.metrics,
                )
                if transcript_logger:
                    transcript_logger.record_turn(
                        args.prompt,
                        turn,
                        tools=enabled_tools,
                    )
                return 0

            print("Type your message and press enter (Ctrl+C to exit).")
            while True:
                prompt = input("You: ")
                if not prompt:
                    continue
                turn = _run_turn(
                    model,
                    prompt,
                    dashboard,
                    metrics_mode=args.metrics,
                )
                if transcript_logger:
                    transcript_logger.record_turn(
                        prompt,
                        turn,
                        tools=enabled_tools,
                    )
            return 0

        from .sera_tui import SeraTUI

        tui = SeraTUI(
            model=model,
            dashboard=dashboard,
            execute_turn=lambda prompt: _execute_turn(model, prompt),
            diagnostics_formatter=lambda diagnostics, generation, turn_tokens: _format_dashboard(
                diagnostics,
                generation=generation,
                turn_tokens=turn_tokens,
                verbose=args.verbose_stats,
            ),
            metrics_formatter=_format_metrics_line,
            manifest_dir=manifest_dir,
            manifest_state_path=state_path,
            manifest_state=runtime_blob,
            manifest_arrays=arrays,
            initial_tools=enabled_tools,
            optional_tools=OPTIONAL_TOOLS,
            load_messages=load_messages,
            metrics_mode=args.metrics,
            transcript_logger=transcript_logger,
        )
        tui.run()
    except KeyboardInterrupt:
        print("\nExiting.")
        if transcript_logger:
            transcript_logger.record_event("Session interrupted", role="system")
    finally:
        dashboard.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
