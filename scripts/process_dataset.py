"""Script to create the WMT14 dataset.

The preprocessing done in the paper is quite complex, but here the only thing
I'm doing is tokenizing using a "modern" tokenizer and filtering long
sentences. I'm doing this because of memory contrains in the hardware I'm
using.

The number of sentences filtered is very little, less than 0.1% of the total
pairs, and so the impact on training performance should be negligible.
"""
import os
import pathlib
import typing as tp

import sentencepiece as spm
from datasets import load_dataset

OUTPUT_DIR = pathlib.Path("data/wmt14")
TOKENIZER_DIR = pathlib.Path("data/tokenizer")

# NOTE: largest number that fit in my RTX3060, i know, i'm poor
FILTER_THRESHOLD = 100


def main() -> None:
    """Main function to process the WMT14 dataset.

    Tokenized the whole dataset, filters long sentences (too many tokens) and
    saves a sorted by length version of the dataset.
    """
    dataset = load_dataset(
        "wmt14",
        name="fr-en",
        revision="ebb5f5979fd115cd1e9d2537103db12539f29822",
    )
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_DIR / "tokenizer.model"), num_threads=0
    )

    dataset = (
        dataset.flatten()
        .rename_column("translation.en", "en")
        .rename_column("translation.fr", "fr")
    )

    def process_fn(
        batch: dict[str, list[str]]
    ) -> dict[str, list[list[str]] | list[int]]:
        en_tokens = tokenizer.encode(batch["en"], out_type=str)
        fr_tokens = tokenizer.encode(batch["fr"], out_type=str)
        return {
            "en_tokens": en_tokens,
            "en_count": [len(toks) for toks in en_tokens],
            "fr_tokens": fr_tokens,
            "fr_count": [len(toks) for toks in fr_tokens],
        }

    def filter_fn(items: dict[str, tp.Any]) -> list[bool]:
        return [
            en_count <= FILTER_THRESHOLD and fr_count <= FILTER_THRESHOLD
            for en_count, fr_count in zip(items["en_count"], items["fr_count"])
        ]

    tokenized = dataset.map(process_fn, batched=True, num_proc=os.cpu_count())
    filtered = tokenized
    filtered["train"] = tokenized["train"].filter(
        filter_fn, batched=True, num_proc=os.cpu_count()
    )
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    filtered.remove_columns(
        ["en_tokens", "fr_tokens", "en_count", "fr_count"]
    ).save_to_disk(OUTPUT_DIR)
    print("Saved dataset", filtered)
    print("to", OUTPUT_DIR)
    print(
        "Original dataset reduced to",
        f"{len(filtered['train']) / len(dataset['train']):.0%}",
    )


if __name__ == "__main__":
    main()
