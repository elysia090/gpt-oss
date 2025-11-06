"""Command line interfaces for the gpt-oss project."""

from .chat import main as chat_main
from .generate import main as generate_main

__all__ = ["chat_main", "generate_main"]
