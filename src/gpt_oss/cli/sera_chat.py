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
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - import only used for type checkers
    from gpt_oss.inference.sera import Sera

DEFAULT_STATE_FILENAMES: Sequence[str] = (
    "sera_state.pkl",
    "sera_state.json",
    "sera_state.msgpack",
)

OPTIONAL_TOOLS: Sequence[str] = ("browser", "python")


def _load_sera_class():
    sera_path = Path(__file__).resolve().parents[1] / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_runtime", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sera


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


def _run_turn(model: "Sera", prompt: str, response_token: int) -> str:
    result = model.step(bytes_data=prompt.encode("utf-8"))
    tokens = result.get("tokens", [])
    try:
        user_text = model.tokenizer.decode(tokens)
    except Exception:  # pragma: no cover - defensive against tokenizer errors
        user_text = b""
    user_text = _decode_text(user_text)

    response_bytes = model.tokenizer.decode([int(response_token)])
    response_text = _decode_text(response_bytes)

    print(f"User ({len(tokens)} tokens): {user_text}")
    print(f"Sera: {response_text}")
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
        "--response-token",
        type=int,
        default=65,
        help="Token id to use for the placeholder model response",
    )
    parser.add_argument(
        "--tool",
        action="append",
        dest="tools",
        default=[],
        choices=sorted(OPTIONAL_TOOLS),
        help="Enable optional tools exposed in the interface",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_dir = _resolve_manifest_dir(args.manifest)
    state_path = _locate_state_file(manifest_dir, args.state_file)
    state_blob = _strip_private_fields(_load_state(state_path))
    Sera = _load_sera_class()
    model = Sera.restore(state_blob)

    enabled_tools = sorted(set(args.tools or []))
    print(f"Loaded Sera manifest from {manifest_dir}")
    if enabled_tools:
        print("Enabled tools: " + ", ".join(enabled_tools))
    else:
        print("Enabled tools: none")

    if args.prompt is not None:
        _run_turn(model, args.prompt, args.response_token)
        return 0

    print("Type your message and press enter (Ctrl+C to exit).")
    try:
        while True:
            prompt = input("You: ")
            if not prompt:
                continue
            _run_turn(model, prompt, args.response_token)
    except KeyboardInterrupt:
        print("\nExiting.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
