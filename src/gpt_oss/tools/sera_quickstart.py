"""Quickstart helper for preparing Sera chat artefacts.

This module downloads the published GPT-OSS checkpoint from the Hugging Face
Hub, converts it into Sera Transfer artefacts, and optionally launches the
:mod:`gpt_oss.cli.sera_chat` interface. By default the helper reuses the
Hugging Face cache instead of materialising a fresh checkpoint copy on disk; a
local directory can be requested via ``--download-dir`` when needed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import sera_transfer

DEFAULT_REPO_ID = "openai/gpt-oss-20b"
DEFAULT_OUTPUT_DIR = Path("gpt-oss-sera-20b")
DEFAULT_R = 512
DEFAULT_R_V = 12
DEFAULT_TOP_L = 12

# Restrict downloads to the original checkpoint payload alongside tokenizer
# metadata required by :mod:`gpt_oss.tools.sera_transfer`. The conversion step
# reads the ``config.json`` and ``model.safetensors`` files from the repository
# root in addition to the ``original/`` subdirectory, so we must permit both
# locations. The Hugging Face repository also contains larger converted variants
# that are unnecessary for the quickstart workflow; scoping the patterns
# prevents unnecessary disk usage while keeping the tokenizer assets intact for
# downstream consumers.
TOKENIZER_FILENAMES: tuple[str, ...] = (
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
CHECKPOINT_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "original/**",
    *TOKENIZER_FILENAMES,
    "tokenizer/**",
)


class QuickstartError(RuntimeError):
    """Raised when the quickstart pipeline encounters a fatal error."""


def _download_checkpoint(
    repo_id: str,
    revision: Optional[str],
    download_dir: Optional[Path],
    *,
    materialize: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - defensive
        raise QuickstartError(
            "huggingface_hub is required to download checkpoints. Install it with"
            " `pip install huggingface-hub`."
        ) from exc

    kwargs = {"allow_patterns": CHECKPOINT_ALLOW_PATTERNS}
    if download_dir is not None:
        download_dir = download_dir.expanduser()
        download_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} to {download_dir}...")
        kwargs["local_dir"] = str(download_dir)
        if materialize:
            # Allow callers to request a fully materialised copy explicitly;
            # otherwise the helper reuses the Hugging Face cache via symlinks.
            kwargs["local_dir_use_symlinks"] = False
    else:
        print(f"Downloading {repo_id} using the Hugging Face cache...")

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        **kwargs,
    )
    return Path(snapshot_path)


def _convert_checkpoint(
    source_dir: Path,
    output_dir: Path,
    *,
    r: int,
    r_v: int,
    top_l: int,
) -> Path:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise QuickstartError(f"Checkpoint directory {source_dir} does not exist")

    output_dir = output_dir.expanduser().resolve()
    print(
        "Converting checkpoint at"
        f" {source_dir} -> {output_dir} (r={r}, r_v={r_v}, top_l={top_l})"
    )
    sera_transfer.convert(source_dir, output_dir, r=r, r_v=r_v, top_l=top_l)
    return output_dir


def _launch_chat_cli(artifacts_dir: Path, extra_args: Sequence[str]) -> int:
    from gpt_oss.cli import sera_chat

    argv = ["--manifest", str(artifacts_dir), *extra_args]
    return sera_chat.main(argv)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GPT-OSS, convert to Sera artefacts, and optionally chat",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository containing the checkpoint",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision to download",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Directory to expose the downloaded checkpoint in. By default the"
            " helper reuses the Hugging Face cache and populates the directory"
            " with symlinks; combine with --materialize-download to request a"
            " full copy on disk."
        ),
    )
    parser.add_argument(
        "--materialize-download",
        action="store_true",
        help=(
            "Force the helper to create a real copy of the checkpoint when"
            " --download-dir is provided. Without this flag the download"
            " directory contains symlinks back into the Hugging Face cache"
            " whenever supported."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Use an existing checkpoint directory instead of downloading",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write Sera artefacts into",
    )
    parser.add_argument("--r", type=int, default=DEFAULT_R, help="Rank compression parameter")
    parser.add_argument(
        "--rv",
        type=int,
        default=DEFAULT_R_V,
        help="Rotary value compression parameter",
    )
    parser.add_argument(
        "--topL",
        type=int,
        default=DEFAULT_TOP_L,
        help="Top-level compression parameter",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch the sera_chat CLI after conversion",
    )
    parser.add_argument(
        "--chat-arg",
        action="append",
        default=[],
        help="Additional argument passed through to sera_chat (can be repeated)",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help=(
            "Delete any existing download/output directories before running."
            " The download directory is only removed when --download-dir is set."
        ),
    )
    return parser.parse_args([] if argv is None else list(argv))


def _prepare_directories(args: argparse.Namespace) -> tuple[Optional[Path], Path]:
    download_dir = args.download_dir
    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    if args.force_clean:
        if (
            args.checkpoint_dir is None
            and download_dir is not None
            and download_dir.exists()
        ):
            shutil.rmtree(download_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)

    return download_dir, output_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = _parse_args(list(argv) if argv is not None else None)
        download_dir, output_dir = _prepare_directories(args)

        if args.checkpoint_dir is not None:
            checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        else:
            checkpoint_dir = _download_checkpoint(
                args.repo_id,
                args.revision,
                download_dir,
                materialize=args.materialize_download,
            )

        artifacts_dir = _convert_checkpoint(
            checkpoint_dir,
            output_dir,
            r=args.r,
            r_v=args.rv,
            top_l=args.topL,
        )

        print(f"Sera artefacts written to {artifacts_dir}")

        if args.chat:
            print("Launching sera_chat CLI...")
            return _launch_chat_cli(artifacts_dir, args.chat_arg)
        return 0
    except QuickstartError as exc:
        print(f"sera_quickstart: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
