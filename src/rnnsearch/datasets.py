"""LightningDataModule for WMT14."""
from __future__ import annotations

import dataclasses
import pathlib
import shlex
import subprocess
import typing as tp

import pytorch_lightning as pl
import sentencepiece
import torch
import torch.nn.utils.rnn as rnnutils
from datasets import load_dataset

if tp.TYPE_CHECKING:
    import torch.utils.data
    from datasets import (  # WMT14Dataset,; WMT14DatasetDict,
        FlatWMT14Dataset,
        FlatWMT14DatasetDict,
        FlatWMT14Item,
    )


@dataclasses.dataclass
class TrainItem:
    """A simple translation train item.

    The ``target`` attribute is used for teacher forcing.
    """

    fr: rnnutils.PackedSequence
    en: rnnutils.PackedSequence
    target: rnnutils.PackedSequence


def _is_path(maybe_path: pathlib.Path | None) -> tp.TypeGuard[pathlib.Path]:
    return maybe_path is not None


def _narrow_cast(
    x: FlatWMT14Dataset,
) -> torch.utils.data.Dataset[TrainItem]:
    return x  # type: ignore[return-value]


class WMT14(pl.LightningDataModule):
    """LightningDataModule implementing Dataloaders for WMT14.

    Todo:
        Maybe use datasets' native data loading capabilities? i.e,
        Dataset.shuffle, Dataset.iter, etc
    """

    datadir: pathlib.Path | None
    batch_size: int

    _splits: FlatWMT14DatasetDict

    def __init__(self, datadir: pathlib.Path | str, batch_size: int) -> None:
        """Create a wmt14 datamodule. Has train, val and test dataloaders.

        Args:
            datadir: The root dataset where datasets are stored. Should contain
                a directory called ``hf-datasets`` where the datasets is cached.
            batch_size: The batch_size for all the dataloaders.
        """
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = pathlib.Path(datadir)
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """Prepares the WMT14 data.

        Since the data is versioned using dvc, the easiest way to download the
        necessary data is just calling ``dvc pull``. This should download the
        cached ``datasets.Dataset`` data and the tokenizer.
        """
        subprocess.call(shlex.split("dvc pull"))

    @property
    def in_memory(self) -> bool:
        """An in-memory datamodule is not mapped to a location on disk."""
        return _is_path(self.datadir)

    def setup(self, stage: str | None = None) -> None:
        """Loads the dataset splits.

        Args:
            stage: (optional) Either ``fit``, ``validate``, ``test`` or
                ``predict``. Added to comply with pytorch lightning's
                signature, in any case all splits are loaded. ``datasets``'
                Dataset are lazy loaded anyways, so this doesn't incur in a
                memory penalty.
        """
        assert stage in ("fit", "validate", "test", "predict", None), "Invalid stage"
        # NOTE: equivalent to using self.in_memory, but then it doesn't work as
        # a type guard.
        if not _is_path(self.datadir):
            return

        self._splits = load_dataset(
            "wmt14",
            name="fr-en",
            cache_dir=str(self.datadir / "hf-datasets"),
            revision="ebb5f5979fd115cd1e9d2537103db12539f29822",
        ).flatten()
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(self.datadir / "tokenizer" / "tokenizer.model"),
            num_threads=0,
        )

    def collate_fn(self, batch: list[FlatWMT14Item[str]]) -> TrainItem:
        """Pack into PackedSequence the values of a FlatWMT14Item."""

        def collate_single(
            sentences: list[str], add_bos: bool, add_eos: bool
        ) -> rnnutils.PackedSequence:
            return rnnutils.pack_sequence(
                [
                    torch.tensor(ids)
                    for ids in self.tokenizer.encode(
                        sentences, add_bos=add_bos, add_eos=add_eos
                    )
                ],
                enforce_sorted=False,
            )

        fr_sents, en_sents = [], []
        for item in batch:
            fr_sents.append(item["translation.fr"])
            en_sents.append(item["translation.en"])
        return TrainItem(
            collate_single(fr_sents, add_bos=True, add_eos=True),
            collate_single(en_sents, add_bos=True, add_eos=False),
            collate_single(en_sents, add_bos=False, add_eos=True),
        )

    def _make_dataloader(
        self, stage: tp.Literal["train", "validation", "test"]
    ) -> torch.utils.data.DataLoader[TrainItem]:
        return torch.utils.data.DataLoader(
            _narrow_cast(self._splits[stage]),
            batch_size=self.batch_size,
            # shuffle=True if stage == "train" else False,
            drop_last=True if stage == "train" else False,
            num_workers=16,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[TrainItem]:
        """Return train dataloader."""
        return self._make_dataloader("train")

    def val_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[TrainItem]:
        """Return validation dataloader."""
        return self._make_dataloader("validation")

    def test_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[TrainItem]:
        """Return test dataloader."""
        return self._make_dataloader("test")
