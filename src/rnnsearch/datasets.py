"""LightningDataModule for WMT14."""
from __future__ import annotations

import dataclasses
import pathlib
import shlex
import subprocess
import typing as tp

import omegaconf
import pytorch_lightning as pl
import sentencepiece
import torch
import torch.nn.utils.rnn as rnnutils
from datasets import load_from_disk

if tp.TYPE_CHECKING:
    import torch.utils.data
    from datasets import DatasetDict, TranslationItem


@dataclasses.dataclass
class TrainItem:
    """A simple translation train item.

    The ``target`` attribute is used for teacher forcing.
    """

    en: rnnutils.PackedSequence
    fr: rnnutils.PackedSequence
    target: rnnutils.PackedSequence


def _is_path(maybe_path: pathlib.Path | None) -> tp.TypeGuard[pathlib.Path]:
    return maybe_path is not None


def _narrow_cast(
    x: torch.utils.data.Dataset[TranslationItem],
) -> torch.utils.data.Dataset[TrainItem]:
    return x  # type: ignore[return-value]


class WMT14(pl.LightningDataModule):
    """LightningDataModule implementing Dataloaders for WMT14.

    Todo:
        Maybe use datasets' native data loading capabilities? i.e,
        Dataset.shuffle, Dataset.iter, etc
    """

    datadir: pathlib.Path
    batch_size: int

    _splits: DatasetDict

    @dataclasses.dataclass
    class Config:
        """Config class for the WMT14 datamodule.

        Attribtues and respective types should be in sync with the arguments of
        __init__.
        """

        datadir: str = omegaconf.MISSING
        batch_size: int = omegaconf.MISSING

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

        self._splits = load_from_disk(str(self.datadir / "wmt14"))
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(self.datadir / "tokenizer" / "tokenizer.model"),
            num_threads=0,
        )

    def collate_fn(self, batch: list[TranslationItem]) -> TrainItem:
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
            fr_sents.append(item["fr"])
            en_sents.append(item["en"])
        return TrainItem(
            collate_single(en_sents, add_bos=True, add_eos=True),
            collate_single(fr_sents, add_bos=True, add_eos=False),
            collate_single(fr_sents, add_bos=False, add_eos=True),
        )

    def _make_dataloader(
        self, stage: tp.Literal["train", "validation", "test"]
    ) -> torch.utils.data.DataLoader[TrainItem]:
        return torch.utils.data.DataLoader(
            _narrow_cast(self._splits[stage]),
            batch_size=self.batch_size,
            shuffle=True if stage == "train" else False,
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


class _DummyItem(tp.TypedDict):
    en: list[int]
    fr: list[int]


@dataclasses.dataclass
class _DummyDataset:
    nexamples: int
    sentence_length: int
    vocab_size: int

    base_seed: int = 1234

    def __getitem__(self, key: int) -> _DummyItem:
        if not 0 <= key < len(self):
            raise IndexError()
        gen = torch.Generator()
        gen.manual_seed(self.base_seed + key)
        both = torch.randint(
            0, self.vocab_size, (2, self.sentence_length), generator=gen
        ).tolist()
        return {"en": both[0], "fr": both[1]}

    def __len__(self) -> int:
        return self.nexamples


class Dummy(pl.LightningDataModule):
    """Dummy datamodule to test GPU memory limitations with large sequence lengths."""

    def __init__(self, *_: tp.Any, batch_size: int, **__: tp.Any) -> None:  # noqa: D107
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

    def setup(self, *_: tp.Any, **__: tp.Any) -> None:  # noqa: D102
        self.dataset = _DummyDataset(
            nexamples=1_000_000, sentence_length=110, vocab_size=30_000
        )

    @staticmethod
    def _narrow_cast(x: _DummyDataset) -> torch.utils.data.Dataset[TrainItem]:
        return x  # type: ignore[return-value]

    @staticmethod
    def collate_fn(batch: list[_DummyItem]) -> TrainItem:
        """Convert list of items to a single batch for training."""
        src_idss = [item["en"] for item in batch]
        tgt_idss = [item["fr"] for item in batch]
        src_packed = rnnutils.pack_sequence(
            [torch.tensor(src_ids) for src_ids in src_idss], enforce_sorted=False
        )
        tgt_packed = rnnutils.pack_sequence(
            [torch.tensor(tgt_ids) for tgt_ids in tgt_idss], enforce_sorted=False
        )
        # NOTE: the same as tgt_packed
        teacher_forcing = rnnutils.pack_sequence(
            [torch.tensor(tgt_ids) for tgt_ids in tgt_idss], enforce_sorted=False
        )

        return TrainItem(en=src_packed, fr=tgt_packed, target=teacher_forcing)

    def train_dataloader(self) -> torch.utils.data.DataLoader[TrainItem]:
        """Dummy train dataloader."""
        return torch.utils.data.DataLoader(
            self._narrow_cast(self.dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
