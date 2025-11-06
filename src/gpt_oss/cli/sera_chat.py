"""Interactive CLI for chatting with a Sera manifest snapshot.

The CLI expects a directory containing a converted Sera manifest bundle as
produced by :mod:`gpt_oss.tools.sera_transfer`. The directory must at least
provide ``sera_manifest.bin`` and a serialised runtime snapshot (``.pkl`` or
``.json``). The snapshot is restored with
:meth:`gpt_oss.inference.sera.Sera.restore` and then used in a simple
read–eval–print loop that mirrors the UX of :mod:`gpt_oss.cli.chat_cli`.

For automated smoke tests a one-shot mode is provided via ``--prompt`` which
executes a single turn and exits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_STATE_FILENAMES: Sequence[str] = (
    "sera_state.pkl",
    "sera_state.json",
    "sera_state.msgpack",
)

OPTIONAL_TOOLS = {"browser", "python"}


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
    """Remove private keys (prefixed with ``_``) from nested structures."""

    if isinstance(blob, dict):
        return {
            key: _strip_private_fields(value)
            for key, value in blob.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(blob, list):
        return [_strip_private_fields(value) for value in blob]
    return blob


def _decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.hex()


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
        candidate = candidate.resolve()
        manifest_file = candidate / "sera_manifest.bin"
        if candidate.is_dir() and manifest_file.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate a Sera manifest directory. Provide --manifest or set "
        "GPT_OSS_SERA_MANIFEST to a directory containing sera_manifest.bin."
    )


def _locate_state(manifest_dir: Path, override: Optional[str]) -> Path:
    if override is not None:
        state_path = Path(override).expanduser()
        if not state_path.is_absolute():
            state_path = manifest_dir / state_path
        state_path = state_path.resolve()
        if not state_path.exists():
            raise FileNotFoundError(f"State file {state_path} not found")
        return state_path

    candidate_paths = [manifest_dir / name for name in DEFAULT_STATE_FILENAMES]
    state_path = _first_existing(*candidate_paths)
    if state_path is None:
        raise FileNotFoundError(
            "No Sera runtime snapshot found alongside the manifest. Expected one "
            "of: " + ", ".join(DEFAULT_STATE_FILENAMES)
        )
    return state_path


def _load_state(path: Path):
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            return pickle.load(fh)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise RuntimeError(f"Unsupported snapshot format: {path.suffix}")


def _run_turn(model, prompt: str, response_token: int) -> str:
    result = model.step(bytes_data=prompt.encode("utf-8"))
    tokens = result.get("tokens", [])
    try:
        normalised = model.tokenizer.decode(tokens)
    except Exception:
        normalised = b""
    if isinstance(normalised, (bytes, bytearray)):
        user_text = _decode_bytes(bytes(normalised))
    else:
        user_text = str(normalised)
    response_bytes = model.tokenizer.decode([int(response_token)])
    if isinstance(response_bytes, (bytes, bytearray)):
        response_text = _decode_bytes(bytes(response_bytes))
    else:
        response_text = str(response_bytes)

    print(f"User ({len(tokens)} tokens): {user_text}")
    print(f"Sera: {response_text}")
    return response_text


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat using Sera manifest artefacts")
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
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(list(argv) if argv is not None else None)

    manifest_dir = _resolve_manifest_dir(args.manifest)
    state_path = _locate_state(manifest_dir, args.state_file)
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
