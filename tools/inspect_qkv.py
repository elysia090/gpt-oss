"""Utility script to inspect QKV tensors in GPT-OSS models.

This helper verifies that the attention projection weights stored in a
``safetensors`` file match the configuration specified in ``config.json``.

Example
-------
python tools/inspect_qkv.py \
    --model gpt-oss-20b/original/model.safetensors \
    --config gpt-oss-20b/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model.safetensors file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the config.json file",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=0,
        help="Transformer block index to inspect (default: 0)",
    )
    parser.add_argument(
        "--prefix",
        default="block.{block}.attn.qkv",
        help=(
            "Template for the parameter prefix. Use {block} placeholder for the "
            "block index."
        ),
    )
    return parser


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tensor_name(prefix: str, suffix: str, block: int) -> str:
    return f"{prefix.format(block=block)}.{suffix}"


def print_expected_dimensions(cfg: dict) -> None:
    hidden_size = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg["head_dim"]

    q_dim = n_heads * head_dim
    k_dim = v_dim = n_kv * head_dim
    total = q_dim + k_dim + v_dim

    print("hidden_size =", hidden_size)
    print("num_attention_heads =", n_heads)
    print("num_key_value_heads =", n_kv)
    print("head_dim =", head_dim)
    print("expected [q_dim, k_dim, v_dim] =", [q_dim, k_dim, v_dim])
    print("expected total qkv dim =", total)


def inspect_tensors(
    *,
    tensor_file: Path,
    keys: Iterable[str],
) -> None:
    with safe_open(str(tensor_file), framework="pt", device="cpu") as f:
        available_keys = set(f.keys())
        for key in keys:
            print(f"has {key:>32}:", key in available_keys)
            if key in available_keys:
                tensor = f.get_tensor(key)
                print(f"{key} shape: {tuple(tensor.shape)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = args.config.resolve()
    model_path = args.model.resolve()

    print("config_path:", config_path)
    cfg = read_config(config_path)
    print_expected_dimensions(cfg)

    print("model_path:", model_path)
    prefix = args.prefix
    block = args.block
    name_w = tensor_name(prefix, "weight", block)
    name_b = tensor_name(prefix, "bias", block)

    inspect_tensors(tensor_file=model_path, keys=[name_w, name_b])


if __name__ == "__main__":
    main()
