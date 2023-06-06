"""Typing stubs for the ``sentencepieces`` package.

The Python version of this package is poorly documented, so I'm trying to do
what I can using the examples
[here](https://github.com/google/sentencepiece/tree/master/python).
"""
from typing import BinaryIO, Iterator

class SentencePieceProcessor:
    def __init__(self, model_file: str, num_threads: int = -1) -> None: ...

class SentencePieceTrainer:
    @staticmethod
    def train(
        sentence_iterator: Iterator[str], model_writer: BinaryIO, vocab_size: int = 8000
    ) -> None: ...
