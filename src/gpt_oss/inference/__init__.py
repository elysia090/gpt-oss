"""Inference backends available for gpt-oss."""

from .torch.model import TokenGenerator as TorchTokenGenerator
from .triton.model import TokenGenerator as TritonTokenGenerator
from .vllm.token_generator import TokenGenerator as VLLMTokenGenerator

from .sera import GenerationPin, GenerationPointer, ManifestRecord, Sera, SeraConfig

__all__ = [
    "TorchTokenGenerator",
    "TritonTokenGenerator",
    "VLLMTokenGenerator",
    "GenerationPin",
    "GenerationPointer",
    "ManifestRecord",
    "Sera",
    "SeraConfig",
]
