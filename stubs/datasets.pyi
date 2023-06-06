"""Typing stubs for the dataset package.

I am choosing to keep the definitions here minimal, just so my code passes
type checks. This is in no way an attempt to fully statically type the
``datasets`` package.
"""
from typing import Generic, Iterable, Literal, TypeVar, overload

from typing_extensions import TypedDict

T1 = TypeVar("T1")

# NOTE: here are a bunch of typed dicts, correcponding to the possible
# different return types of __getitem__ in a datasets.Dataset. These
# definitions cannot be inlined, see https://github.com/python/mypy/issues/9884

class EnFrDict(TypedDict, Generic[T1]):
    fr: T1
    en: T1

class WMT14Item(TypedDict, Generic[T1]):
    translation: T1

FlatWMT14Item = TypedDict("FlatWMT14Item", {"translation.fr": T1, "translation.en": T1})

class WMT14Dataset:
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> WMT14Item[EnFrDict[str]]: ...
    @overload
    def __getitem__(self, key: slice) -> WMT14Item[list[EnFrDict[str]]]: ...
    def __iter__(self) -> Iterable[WMT14Item[EnFrDict[str]]]: ...
    def flatten(
        self, new_fingerprint: str | None = None, max_depth: int = 16
    ) -> FlatWMT14Dataset: ...
    def iter(self, batch_size: int) -> Iterable[WMT14Item[list[EnFrDict[str]]]]: ...

class FlatWMT14Dataset:
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: int) -> FlatWMT14Item[str]: ...
    @overload
    def __getitem__(self, key: slice) -> FlatWMT14Item[list[str]]: ...
    def __iter__(self) -> Iterable[FlatWMT14Item[str]]: ...
    def iter(self, batch_size: int) -> Iterable[FlatWMT14Item[list[str]]]: ...

def load_dataset(
    path: Literal["wmt14"],
    name: Literal["fr-en"],
    split: str,
    cache_dir: str | None = None,
) -> WMT14Dataset: ...
