# mypy: ignore-errors
"""Script to create the WMT14 dataset.

The preprocessing done in the paper is quite complex, but here the only thing
I'm doing is tokenizing using a "modern" tokenizer and filtering long
sentences. I'm doing this because of memory contrains in the hardware I'm
using.

The number of sentences filtered is very little, less than 0.1% of the total
pairs, and so the impact on training performance should be negligible.
"""
import pathlib

import sentencepiece as spm
from datasets import load_dataset

OUTPUT_DIR = pathlib.Path("data/wmt14")
TOKENIZER_DIR = pathlib.Path("data/tokenizer")

FILTER_THRESHOLD = 300


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
        model_file=str(TOKENIZER_DIR / "tokenizer.model")
    )

    dataset = (
        dataset.flatten()
        .rename_column("translation.en", "en")
        .rename_column("translation.fr", "fr")
    )

    def process_fn(batch):
        en_tokens = tokenizer.encode(batch["en"], out_type=str)
        fr_tokens = tokenizer.encode(batch["fr"], out_type=str)
        return {"en_tokens": en_tokens, "fr_tokens": fr_tokens}

    def filter_fn(items):
        return [
            len(en_tokens) <= FILTER_THRESHOLD and len(fr_tokens) <= FILTER_THRESHOLD
            for en_tokens, fr_tokens in zip(items["en_tokens"], items["fr_tokens"])
        ]

    tokenized = dataset.map(process_fn, batched=True)
    filtered = tokenized.filter(filter_fn, batched=True, num_proc=8)

    # Could've been done in process_fn, but I already had that in cache and I
    # didn't want to process de whole dataset again
    def counts(batch):
        return {
            "en_count": [len(item) for item in batch["en_tokens"]],
            "fr_counts": [len(item) for item in batch["fr_tokens"]],
        }

    with_counts = filtered.map(counts, batched=True, num_proc=8)
    sorted_dataset = with_counts.sort(
        ["en_count", "fr_counts"], reverse=True, writer_batch_size=10_000
    )

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    sorted_dataset.remove_columns(
        ["en_tokens", "fr_tokens", "en_count", "fr_counts"]
    ).save_to_disk(OUTPUT_DIR, max_shard_size="1GB")
    print("Saved dataset", filtered)
    print("to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
