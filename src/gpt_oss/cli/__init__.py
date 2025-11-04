"""Command line interfaces for the gpt-oss project."""

from .chat_cli import main as chat_main
from .generate_cli import main as generate_main

__all__ = ["chat_main", "generate_main"]
