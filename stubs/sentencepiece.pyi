"""Typing stubs for the ``sentencepieces`` package.

The Python version of this package is poorly documented, so I'm trying to do
what I can using the examples
[here](https://github.com/google/sentencepiece/tree/master/python).
"""
from typing import BinaryIO, Iterator, overload

class SentencePieceProcessor:
    def __init__(self, model_file: str, num_threads: int = -1) -> None: ...
    def id_to_piece(self, id: int) -> str: ...
    @overload
    def encode(
        self,
        input: str,
        out_type: type[int] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[int]: ...
    @overload
    def encode(
        self,
        input: list[str],
        out_type: type[int] | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[list[int]]: ...
    @overload
    def encode(
        self,
        input: str,
        out_type: type[str],
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[str]: ...
    @overload
    def encode(
        self,
        input: list[str],
        out_type: type[str],
        add_bos: bool = False,
        add_eos: bool = False,
        revers: bool = False,
        num_threads: int = -1,
    ) -> list[list[int]]: ...
    def get_score(self, id: int) -> float: ...
    def get_piece_size(self) -> int: ...
    def vocab_size(self) -> int: ...

class SentencePieceTrainer:
    @staticmethod
    def train(
        sentence_iterator: Iterator[str],
        model_writer: BinaryIO,
        vocab_size: int = 8000,
        character_coverage: float = 0.9995,
        pad_id: int = -1,
    ) -> None: ...
