from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

FIXTURE_ROOT = Path(__file__).resolve().parent / "data"
SAMPLE_TOKENIZER = FIXTURE_ROOT / "sample_tokenizer.json"


def install_sample_tokenizer(destination: Path, *, filenames: Iterable[str] | None = None) -> None:
    """Install the sample tokenizer into *destination*.

    ``filenames`` allows callers to mirror checkpoint layouts that expect
    multiple tokenizer artefacts. Missing files are ignored.
    """

    destination.mkdir(parents=True, exist_ok=True)
    payload = SAMPLE_TOKENIZER.read_text()
    tokenizer_path = destination / "tokenizer.json"
    tokenizer_path.write_text(payload)

    if filenames is None:
        return

    for name in filenames:
        target = destination / name
        if target.name == "tokenizer.json":
            continue
        try:
            target.write_text(payload)
        except OSError:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload)


def load_sample_tokenizer() -> dict:
    return json.loads(SAMPLE_TOKENIZER.read_text())
