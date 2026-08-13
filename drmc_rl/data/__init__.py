"""Versioned external datasets consumed by drmc-rl."""

from .human_corpus import HumanCorpus, decode_input_rle

__all__ = ["HumanCorpus", "decode_input_rle"]
