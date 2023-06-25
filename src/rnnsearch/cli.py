"""CLI module implementing commands to train the network."""
import dataclasses
import pathlib
import shlex
import subprocess
import typing as tp

import pytorch_lightning as pl
import sentencepiece as spm
import torch
import typer
import typing_extensions as tpx
from omegaconf import OmegaConf

import rnnsearch.datasets as datasets
import rnnsearch.nn as nn

DATADIR = "data"
cli = typer.Typer(
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)


def _get_vocab_size() -> int:
    subprocess.run(shlex.split("dvc pull data/tokenizer"), capture_output=True)
    vocab_size = spm.SentencePieceProcessor(
        model_file=str(pathlib.Path(DATADIR) / "tokenizer" / "tokenizer.model")
    ).vocab_size()
    return vocab_size


@dataclasses.dataclass
class TrainerConfig:
    """Trainer config."""

    deterministic: bool = True
    max_epochs: int = 100
    max_steps: int = -1
    val_check_interval: int = 1000


@dataclasses.dataclass
class TrainingConfig:
    """Config class for training related params."""

    seed: int = 1234
    # NOTE: uses iso-8601 durations:
    # https://en.wikipedia.org/wiki/ISO_8601#Durations
    checkpoint_time_interval: str = "PT1M"
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)


@dataclasses.dataclass
class Settings:
    """Settings class."""

    model: nn.RNNSearch.Config = dataclasses.field(default_factory=nn.RNNSearch.Config)
    dataset: datasets.WMT14.Config = dataclasses.field(
        default_factory=datasets.WMT14.Config
    )
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)


def _merge_settings(
    config: pathlib.Path,
    cmd_options: tp.List[str],
    extra_options: tp.Dict[str, tp.Any],
) -> Settings:
    defaults = OmegaConf.structured(Settings)
    from_file = OmegaConf.load(config)
    from_cmd = OmegaConf.from_dotlist(cmd_options)
    from_extras = OmegaConf.create(extra_options)
    merged: Settings = OmegaConf.merge(  # type: ignore[assignment]
        defaults,
        from_extras,
        from_file,
        from_cmd,
    )
    return merged


@cli.command()
def train(
    *,
    config: tpx.Annotated[
        pathlib.Path, typer.Option(show_default=False)
    ] = pathlib.Path("/dev/null"),
    options: tpx.Annotated[
        tp.List[str],
        typer.Option(
            "--option",
            "-o",
            help="""Override options using dotlist style keys. Can be used
            multiple times. Example: -o dataset.batch_size=32 -o
            model.hidden_size=100""",
            show_default=False,
            default_factory=list,
        ),
    ],
) -> None:
    """Command to train the network."""
    vocab_size = _get_vocab_size()
    settings = _merge_settings(
        config,
        options,
        dict(
            model=dict(vocab_size=vocab_size, output_dim=vocab_size),
            dataset=dict(datadir=DATADIR),
        ),
    )
    print("Config\n", OmegaConf.to_yaml(settings).strip())  # noqa: T201
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(settings.training.seed)
    # NOTE: on ignore[arg-type], this works, but OmegaConf doesn't play well
    # with mypy (specially nested configs)
    model = nn.RNNSearch(**settings.model)  # type: ignore[arg-type]
    datamodule = datasets.WMT14(**settings.dataset)  # type: ignore[arg-type]
    trainer = pl.Trainer(  # type: ignore[arg-type]
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                every_n_train_steps=settings.training.trainer.val_check_interval + 1,
            ),
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs", name="", default_hp_metric=False
        ),
        **settings.training.trainer,
    )
    trainer.fit(model, datamodule)


@cli.callback()
def _dummy() -> None:
    pass
