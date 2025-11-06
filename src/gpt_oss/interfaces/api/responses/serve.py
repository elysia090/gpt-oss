# torchrun --nproc-per-node=4 serve.py

import argparse

import platform

import uvicorn
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from .api_server import create_api_server


def _resolve_backend(name: str):
    match name:
        case "triton":
            from .inference.triton import setup_model

            return setup_model
        case "stub":
            from .inference.stub import setup_model

            return setup_model
        case "metal":
            from .inference.metal import setup_model

            return setup_model
        case "ollama":
            from .inference.ollama import setup_model

            return setup_model
        case "vllm":
            from .inference.vllm import setup_model

            return setup_model
        case "transformers":
            from .inference.transformers import setup_model

            return setup_model
        case _:
            raise ValueError(f"Invalid inference backend: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Responses API server")
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
        default="~/model",
        required=False,
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--inference-backend",
        metavar="BACKEND",
        type=str,
        help="Inference backend to use",
        # default to metal on macOS, triton on other platforms
        default="metal" if platform.system() == "Darwin" else "triton",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_model = _resolve_backend(args.inference_backend)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    infer_next_token = setup_model(args.checkpoint)
    uvicorn.run(create_api_server(infer_next_token, encoding), port=args.port)


if __name__ == "__main__":
    main()
