"""Create deterministic song-level train, validation and test splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed songs and create a train-only vocabulary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/songs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--minimum-frequency",
        type=int,
        default=1,
        help="Minimum train frequency for a token to enter the vocabulary.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed corpus not found: {path}. "
            "Run preprocess_abc.py first."
        )

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_sizes(total: int, train_ratio: float, validation_ratio: float) -> tuple[int, int]:
    if total < 3:
        raise ValueError("At least three usable songs are required for splitting.")

    train_size = max(1, int(total * train_ratio))
    validation_size = max(1, int(total * validation_ratio))

    if train_size + validation_size >= total:
        train_size = total - 2
        validation_size = 1

    return train_size, validation_size


def build_vocabulary(
    train_songs: list[dict],
    minimum_frequency: int,
) -> tuple[dict[str, int], Counter]:
    frequencies = Counter(
        token
        for song in train_songs
        for token in song["sequence_tokens"]
    )

    vocabulary = {
        token: index for index, token in enumerate(SPECIAL_TOKENS)
    }

    eligible = [
        token
        for token, count in frequencies.items()
        if count >= minimum_frequency and token not in vocabulary
    ]

    # Stable ordering: most frequent first, then alphabetically.
    eligible.sort(key=lambda token: (-frequencies[token], token))

    for token in eligible:
        vocabulary[token] = len(vocabulary)

    return vocabulary, frequencies


def oov_statistics(songs: list[dict], vocabulary: dict[str, int]) -> dict:
    tokens = [
        token
        for song in songs
        for token in song["sequence_tokens"]
    ]
    unknown = sum(token not in vocabulary for token in tokens)

    return {
        "total_tokens": len(tokens),
        "unknown_tokens": unknown,
        "unknown_token_rate": unknown / len(tokens) if tokens else 0.0,
    }


def main() -> None:
    args = parse_args()

    if args.train_ratio <= 0 or args.validation_ratio <= 0:
        raise ValueError("Split ratios must be positive.")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError(
            "train-ratio + validation-ratio must be smaller than 1."
        )

    all_songs = load_jsonl(args.input)
    songs = [
        song
        for song in all_songs
        if song.get("usable_for_chord_model", False)
    ]

    rng = random.Random(args.seed)
    rng.shuffle(songs)

    train_size, validation_size = split_sizes(
        len(songs),
        args.train_ratio,
        args.validation_ratio,
    )

    train = songs[:train_size]
    validation = songs[train_size : train_size + validation_size]
    test = songs[train_size + validation_size :]

    args.output.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for name, records in splits.items():
        save_jsonl(records, args.output / f"{name}.jsonl")
        with (args.output / f"{name}_ids.txt").open(
            "w", encoding="utf-8"
        ) as file:
            for record in records:
                file.write(record["song_id"] + "\n")

    vocabulary, train_frequencies = build_vocabulary(
        train,
        args.minimum_frequency,
    )

    with (args.output / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocabulary, file, indent=2, ensure_ascii=False)

    with (args.output / "vocab_frequencies.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                token: train_frequencies[token]
                for token in vocabulary
                if token not in SPECIAL_TOKENS
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "seed": args.seed,
        "total_usable_songs": len(songs),
        "train_songs": len(train),
        "validation_songs": len(validation),
        "test_songs": len(test),
        "vocabulary_size": len(vocabulary),
        "minimum_frequency": args.minimum_frequency,
        "validation_oov": oov_statistics(validation, vocabulary),
        "test_oov": oov_statistics(test, vocabulary),
    }

    with (args.output / "split_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nSong-level split complete")
    print("-" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
