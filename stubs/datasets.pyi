"""Typing stubs for the datasets module.

This stub only applies for my project and doesn't attempt to be a general
typing stub for the datasets package.
"""
import pathlib
from typing import Any, Callable, Iterable, Literal, ParamSpec, TypedDict, overload

from torch.utils.data import Dataset
from typing_extensions import Self

class TranslationItem(TypedDict):
    en: str
    fr: str

class DatasetDict(TypedDict):
    train: Dataset[TranslationItem]
    validation: Dataset[TranslationItem]
    test: Dataset[TranslationItem]

def load_from_disk(dataset_path: str) -> DatasetDict: ...

P = ParamSpec("P")

class _DatasetsCommon:
    def flatten(self) -> Self: ...
    def iter(self, batch_size: int) -> Iterable[Any]: ...
    def rename_column(
        self, original_column_name: str, new_column_name: str
    ) -> Self: ...
    def map(
        self,
        function: Callable[..., Any],
        batched: bool = False,
        num_proc: int | None = None,
    ) -> Self: ...
    def filter(
        self,
        function: Callable[..., Any],
        batched: bool = False,
        num_proc: int | None = None,
    ) -> Self: ...
    def remove_columns(self, columns: str | list[str]) -> Self: ...
    def save_to_disk(self, dataset_path: str | pathlib.Path) -> None: ...

class WMT14Dict(_DatasetsCommon):
    def __getitem__(self, key: Literal["train", "validation", "test"]) -> WMT14: ...
    def __setitem__(
        self, key: Literal["train", "validation", "test"], value: WMT14
    ) -> None: ...
    def __len__(self) -> Literal[3]: ...

class WMT14(_DatasetsCommon):
    def __getitem__(self, key: int) -> TranslationItem: ...
    def __len__(self) -> int: ...

@overload
def load_dataset(
    path: Literal["wmt14"],
    name: Literal["fr-en"],
    split: str,
    revision: str | None = None,
) -> WMT14: ...
@overload
def load_dataset(
    path: Literal["wmt14"],
    name: Literal["fr-en"],
    split: None = None,
    revision: str | None = None,
) -> WMT14Dict: ...
