"""Typing stubs for the dataset package.

I am choosing to keep the definitions here minimal, just so my code passes
type checks. This is in no way an attempt to fully statically type the
``datasets`` package.
"""
from typing import Generic, Iterable, Literal, Mapping, TypeAlias, TypeVar

KT: TypeAlias = int
VT = TypeVar("VT")

class Dataset(Mapping[KT, VT], Generic[VT]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: KT) -> VT: ...
    # NOTE: on ignore[override], normal mapping allow for iteration over keys,
    # but a Dataset iterates over values
    def __iter__(self) -> Iterable[VT]: ...  # type: ignore[override]

class IterableDataset: ...

def load_dataset(path: Literal["wmt14"]) -> Dataset: ...
