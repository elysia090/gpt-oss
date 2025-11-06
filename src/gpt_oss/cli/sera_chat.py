"""Interactive CLI for chatting with a Sera runtime snapshot.

The CLI expects a directory containing a serialized :class:`~gpt_oss.inference.sera.Sera`
snapshot (typically produced by the Sera transfer tooling).  The snapshot is
loaded via :func:`gpt_oss.inference.sera.Sera.restore` and then used to run a
simple read-eval-print loop.

For automated smoke tests a one-shot mode is provided via ``--prompt`` which
executes a single turn and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional

import importlib.util
import sys


def _load_sera_runtime():
    sera_path = Path(__file__).resolve().parents[1] / "inference" / "sera.py"
    spec = importlib.util.spec_from_file_location("_sera_runtime", sera_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to locate Sera runtime module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sera


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


DEFAULT_STATE_FILENAMES: Iterable[str] = (
    "sera_state.pkl",
    "sera_state.json",
    "sera_state.msgpack",
    "sera_state.yaml",
)


def _first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _detect_artifacts(path: Optional[str]) -> Path:
    """Resolve the directory that contains the Sera snapshot."""

    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser().resolve())
    env_path = os.environ.get("GPT_OSS_SERA_ARTIFACTS")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())
    candidates.append(Path.cwd() / "gpt-oss-sera-20b")
    candidates.append(Path.cwd())

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Unable to locate the Sera artifact directory. Provide --artifacts or "
        "set GPT_OSS_SERA_ARTIFACTS."
    )


def _load_state(artifacts: Path, state_file: Optional[str]) -> dict:
    """Load the serialized Sera snapshot from disk."""

    if state_file is not None:
        state_path = Path(state_file).expanduser()
        if not state_path.is_absolute():
            state_path = (artifacts / state_path).resolve()
        if not state_path.exists():
            raise FileNotFoundError(f"State file {state_path} not found")
    else:
        candidate_paths = [artifacts / name for name in DEFAULT_STATE_FILENAMES]
        state_path = _first_existing(*candidate_paths)
        if state_path is None:
            raise FileNotFoundError(
                f"No Sera snapshot found in {artifacts}. Expected one of: "
                + ", ".join(name for name in DEFAULT_STATE_FILENAMES)
            )

    if state_path.suffix == ".json":
        try:
            with state_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to parse Sera snapshot {state_path}: {exc}") from exc
    elif state_path.suffix in {".pkl", ".pickle"}:
        with state_path.open("rb") as fh:
            return pickle.load(fh)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported snapshot format for {state_path}")


def _decode_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.hex()


def _run_turn(model, prompt: str, response_token: int) -> str:
    result = model.step(bytes_data=prompt.encode("utf-8"))
    tokens = result.get("tokens", [])
    try:
        normalised = model.tokenizer.decode(tokens)
    except Exception:  # pragma: no cover - tokenizer failures are surfaced
        normalised = b""
    response_bytes = model.tokenizer.decode([int(response_token)])
    response_text = _decode_bytes(response_bytes)

    print(f"User ({len(tokens)} tokens): {_decode_bytes(normalised)}")
    print(f"Sera: {response_text}")
    return response_text


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive chat using Sera runtime snapshots")
    parser.add_argument(
        "--artifacts",
        type=str,
        help="Directory containing the converted Sera artifact bundle",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        help="Explicit path (relative or absolute) to the serialized Sera snapshot",
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
        help="Token id to use for the default response when generating output",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    artifacts = _detect_artifacts(args.artifacts)
    state_blob = _load_state(artifacts, args.state_file)
    if isinstance(state_blob, dict) and "config" in state_blob:
        state_blob["config"] = _strip_private_fields(state_blob["config"])
    Sera = _load_sera_runtime()
    model = Sera.restore(state_blob)

    if args.prompt is not None:
        _run_turn(model, args.prompt, args.response_token)
        return 0

    print("Loaded Sera runtime. Type your message and press enter (Ctrl+C to exit).")
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
