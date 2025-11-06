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
import struct
import sys
import time
import hashlib
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
)

if TYPE_CHECKING:  # pragma: no cover - import only used for type checkers
    from gpt_oss.inference.sera import Sera

DEFAULT_STATE_FILENAMES: Sequence[str] = (
    "sera_state.pkl",
    "sera_state.json",
    "sera_state.msgpack",
)

OPTIONAL_TOOLS: Sequence[str] = ("browser", "python")


_ARRAY_HEADER_STRUCT = struct.Struct("<I H H 5Q Q Q Q I I")
_DTYPE_CODES: Dict[int, str] = {
    1: "f64",
    2: "f32",
    3: "i32",
    4: "i16",
    5: "i8",
    6: "u8",
    7: "q8_8",
    8: "q4_12",
    9: "bf16",
}


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
    cfr_mode = diagnostics.get("cfr_mode", "OFF")
    cfr_beta = diagnostics.get("cfr_beta", 0.0)
    cfr_guard = diagnostics.get("cfr_guard", False)
    cfr_health = diagnostics.get("cfr_health_ok", True)
    cfr_y_cfr = diagnostics.get("cfr_y_cfr", 0.0)

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
        "  cfr: mode="
        + str(cfr_mode)
        + f" beta={_format_float(cfr_beta)} guard={cfr_guard} health={cfr_health}"
        + f" y_cfr={_format_float(cfr_y_cfr)}"
    )
    return "\n".join(details)


@dataclass
class DiagnosticsDashboard:
    refresh_interval: float
    verbose: bool
    log_path: Optional[Path] = None
    _clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.refresh_interval = max(0.0, float(self.refresh_interval))
        self._last_render: float = 0.0
        self._log_file = None
        self._last_line_len = 0
        if self.log_path is not None:
            log_path = self.log_path.expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self.refresh_interval > 0.0 and self._last_line_len:
            print()
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
            if self.refresh_interval <= 0.0:
                print(line)
            else:
                padding = max(0, self._last_line_len - len(line))
                sys.stdout.write("\r" + line + (" " * padding))
                sys.stdout.flush()
                self._last_line_len = len(line)


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
            return json.load(fh)
    raise RuntimeError(f"Unsupported snapshot format: {path}")


def _parse_array_header(values: Sequence[int]) -> Dict[str, object]:
    magic = int(values[0])
    if magic != 0x53455241:
        raise ValueError("Invalid Sera array header magic")
    dtype_code = int(values[1])
    rank = int(values[2])
    dims = tuple(int(dim) for dim in values[3:8])
    shape = tuple(dim for dim in dims[:rank]) if rank > 0 else tuple()
    header = {
        "magic": magic,
        "dtype_code": dtype_code,
        "dtype": _DTYPE_CODES.get(dtype_code, f"code{dtype_code}"),
        "rank": rank,
        "dims": dims,
        "shape": shape,
        "byte_len": int(values[8]),
        "crc32c": int(values[9]),
        "sha256_low64": int(values[10]),
        "flags": int(values[11]),
        "reserved": int(values[12]),
    }
    return header


def _load_array_file(path: Path) -> tuple[Dict[str, object], bytes]:
    with path.open("rb") as fh:
        header_raw = fh.read(_ARRAY_HEADER_STRUCT.size)
        if len(header_raw) != _ARRAY_HEADER_STRUCT.size:
            raise ValueError(f"Array file {path} is truncated")
        values = _ARRAY_HEADER_STRUCT.unpack(header_raw)
        payload = fh.read()
    header = _parse_array_header(values)
    if len(payload) != header["byte_len"]:
        raise ValueError(
            f"Array payload length mismatch for {path}: "
            f"expected {header['byte_len']}, got {len(payload)}"
        )
    return header, payload


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
        header, payload = _load_array_file(array_path)
        expected_sha = record.get("sha256") if isinstance(record, Mapping) else None
        if expected_sha:
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected_sha:
                raise ValueError(
                    f"Array {array_path} failed SHA-256 validation: "
                    f"expected {expected_sha}, got {digest}"
                )
        loaded[name] = {
            "dtype": header["dtype"],
            "shape": header["shape"],
            "byte_len": header["byte_len"],
        }
    return loaded


def _run_turn(
    model: "Sera",
    prompt: str,
    dashboard: DiagnosticsDashboard,
) -> str:
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
    print("Sera: ", end="", flush=True)

    fragments: list[str] = []
    for token_id in generated_ids:
        try:
            fragment = model.tokenizer.decode([int(token_id)])
        except Exception:  # pragma: no cover - defensive against tokenizer errors
            fragment = b""
        text = _decode_text(fragment)
        fragments.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()
    print()

    response_text = "".join(fragments)

    if "y_out" in result:
        print(f"Logit y_out: {_format_float(result['y_out'])}")

    diagnostics = dict(model.diagnostics_record())
    turn_tokens = len(generated_ids)
    diagnostics["tokens_per_sec"] = turn_tokens / elapsed if turn_tokens else 0.0
    dashboard.update(
        diagnostics,
        generation=getattr(model, "generation", 0),
        turn_tokens=turn_tokens,
    )
    return response_text


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
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_dir = _resolve_manifest_dir(args.manifest)
    state_path = _locate_state_file(manifest_dir, args.state_file)
    state_blob = _strip_private_fields(_load_state(state_path))

    arrays = _load_manifest_arrays(manifest_dir, state_blob.get("artefacts"))
    if arrays:
        state_blob.setdefault("arrays", arrays)

    Sera, SeraConfig = _load_sera_class()
    if "config" in state_blob:
        model = Sera.restore(state_blob)
    else:
        if SeraConfig is None:  # pragma: no cover - defensive fallback
            raise RuntimeError("Sera runtime does not expose SeraConfig")
        config_blob = state_blob.get("model_config")
        if isinstance(config_blob, Mapping):
            config = SeraConfig(**config_blob)
        else:
            config = SeraConfig()
        model = Sera(config)

    dashboard = DiagnosticsDashboard(
        refresh_interval=args.stats_refresh,
        verbose=args.verbose_stats,
        log_path=Path(args.diagnostic_log).expanduser() if args.diagnostic_log else None,
    )

    enabled_tools = sorted(set(args.tools or []))
    print(f"Loaded Sera manifest from {manifest_dir}")
    if arrays:
        preview = ", ".join(
            f"{name}:{info['dtype']}@{info['shape']}"
            for name, info in list(arrays.items())[:3]
        )
        if len(arrays) > 3:
            preview += ", ..."
        print(
            f"Loaded {len(arrays)} arrays from {manifest_dir / 'arrays'}"
            + (f" [{preview}]" if preview else "")
        )
    if enabled_tools:
        print("Enabled tools: " + ", ".join(enabled_tools))
    else:
        print("Enabled tools: none")

    try:
        if args.prompt is not None:
            _run_turn(model, args.prompt, dashboard)
            return 0

        print("Type your message and press enter (Ctrl+C to exit).")
        while True:
            prompt = input("You: ")
            if not prompt:
                continue
            _run_turn(model, prompt, dashboard)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        dashboard.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
