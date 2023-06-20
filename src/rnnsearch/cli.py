"""CLI module implementing commands to train the network."""
import pathlib
import shlex
import subprocess

import pytorch_lightning as pl
import sentencepiece as spm
import torch
import typer

import rnnsearch.datasets as datasets
import rnnsearch.nn as nn

DATADIR = pathlib.Path("data")


cli = typer.Typer(
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)


def _get_vocab_size() -> int:
    subprocess.call(shlex.split("dvc pull data/tokenizer"))
    vocab_size = spm.SentencePieceProcessor(
        model_file=str(DATADIR / "tokenizer" / "tokenizer.model")
    ).vocab_size()
    return vocab_size


@cli.command()
def train(
    embedding_dim: int = 620,
    hidden_size: int = 1000,
    alignment_dim: int = 1000,
    learn_rate: float = 1e-3,
    batch_size: int = 80,
) -> None:
    """Command to train the network."""
    torch.set_float32_matmul_precision("medium")
    vocab_size = _get_vocab_size()
    model = nn.RNNSearch(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_dim=vocab_size,
        alignment_dim=alignment_dim,
        learn_rate=learn_rate,
    )
    datamodule = datasets.WMT14(datadir=DATADIR, batch_size=batch_size)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
    )
    trainer.fit(model, datamodule)


@cli.callback()
def _dummy() -> None:
    pass
