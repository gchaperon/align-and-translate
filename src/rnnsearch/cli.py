"""CLI module implementing commands to train the network."""
import pathlib

import click
import pytorch_lightning as pl
import sentencepiece as spm

import rnnsearch.datasets as datasets
import rnnsearch.nn as nn

DATADIR = pathlib.Path("data")


@click.group()
def cli() -> None:
    """Main entrypoint for the application."""
    pass


@cli.command()
def train() -> None:
    """Command to train the network."""
    vocab_size = spm.SentencePieceProcessor(
        model_file=str(DATADIR / "tokenizer" / "tokenizer.model")
    ).vocab_size()
    model = nn.RNNSearch(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_size=100,
        output_dim=vocab_size,
        alignment_dim=100,
        learn_rate=1e-3,
    )
    datamodule = datasets.WMT14(datadir=DATADIR, batch_size=32)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
    )
    trainer.fit(model, datamodule)
