# noqa: D100, this is a script, no intention of documenting it as a module
import argparse
import io
import pathlib
import shutil
import typing as tp

import sentencepiece as spm
from datasets import load_dataset

OUTPUT_DIR = "data/tokenizer"
# NOTE: this is chosen because is the max number that makes the algorithm fit
# in the 32gb + 32gb(swap) of memory that I have (i'm poor).
MAX_SENTENCES = 8_000_000

parser = argparse.ArgumentParser(
    prog="train_tokenizer",
    description=(
        "Command to train a tokenizer for the wmt14 task. "
        "Only customization available is the vocabulary size."
    ),
)
parser.add_argument(
    "--vocab-size", help="Vocabulary size of the resulting tokenizer.", default=60_000
)


def train_tokenizer(
    sentences: tp.Iterable[str],
    vocab_size: int,
) -> tp.BinaryIO:
    """Trains a sentencepiece tokenizer using the ``sentences`` string iterable.

    Args:
        sentences: An iterable of sentences, used for training the tokenizer model.
        vocab_size: The vocabulary size of the model.

    Returns:
        The model file
    """
    buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences), model_writer=buffer, vocab_size=vocab_size
    )
    return buffer


def main(vocab_size: int) -> None:
    """Builds tokenizer for wmt14, fr-en split.

    This funcition builds a sentencepiece (unigram) tokenizer using a subset of
    the sentences in the wmt14 task and saves it to ``OUTPUT_DIR`` (default
    ``data/tokenizer``). The vocabulary size of the tokenizer can be configured
    via the ``vocab_size`` argument.

    Args:
        vocab_size: The size of the vocabulary for the tokenizer.
    """
    dset = load_dataset(
        "wmt14",
        name="fr-en",
        split=f"train[:{MAX_SENTENCES//2}]",
        cache_dir="data/hf-datasets/",
    )
    sentences = (
        sent
        for batch in dset.flatten().iter(batch_size=1000)
        for sent in batch["translation.en"] + batch["translation.fr"]
    )

    model = train_tokenizer(sentences, vocab_size)
    output_path = pathlib.Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    with open(output_path / "tokenizer.model", "wb") as of:
        print("Saving to", of.name)
        model.seek(0)
        shutil.copyfileobj(model, of)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
