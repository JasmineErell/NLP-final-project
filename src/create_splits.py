"""Create deterministic song-level train, validation and test splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Callable


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[MASK]",
    "[CLS]",
    "[SEP]",
]

REPRESENTATIONS = {
    "atomic": {
        "sequence_field": "atomic_sequence_tokens",
        "is_prediction_target": lambda token: token.startswith("CHORD_"),
    },
    "factorized": {
        "sequence_field": "factorized_sequence_tokens",
        "is_prediction_target": lambda token: token.startswith("[EVENT_"),
    },
}


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Split processed songs and create separate train-only "
            "vocabularies for atomic and factorized representations."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/songs.jsonl"),
        help="Processed corpus created by preprocess_abc.py.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits"),
        help="Directory in which split files will be saved.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--minimum-frequency",
        type=int,
        default=1,
        help=(
            "Minimum training frequency required for a token "
            "to enter the vocabulary."
        ),
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSON Lines file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Processed corpus not found: {path}. "
            "Run preprocess_abc.py first."
        )

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_jsonl(
    records: list[dict],
    path: Path,
) -> None:
    """Save records in JSON Lines format."""

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_json(
    data: dict,
    path: Path,
) -> None:
    """Save a dictionary as formatted JSON."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def split_sizes(
    total: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[int, int]:
    """Calculate train and validation split sizes."""

    if total < 3:
        raise ValueError(
            "At least three usable songs are required for splitting."
        )

    train_size = max(
        1,
        int(total * train_ratio),
    )

    validation_size = max(
        1,
        int(total * validation_ratio),
    )

    # Always leave at least one song for validation and one for testing.
    if train_size + validation_size >= total:
        train_size = total - 2
        validation_size = 1

    return train_size, validation_size


def validate_song_fields(
    songs: list[dict],
) -> None:
    """Verify that preprocessing created both representations."""

    required_fields = {
        "song_id",
        "atomic_sequence_tokens",
        "factorized_sequence_tokens",
    }

    for song in songs:
        missing_fields = required_fields.difference(song)

        if missing_fields:
            raise KeyError(
                f"Song {song.get('song_id', 'UNKNOWN')} is missing "
                f"the following fields: {sorted(missing_fields)}. "
                "Run the updated preprocess_abc.py first."
            )


def build_vocabulary(
    train_songs: list[dict],
    sequence_field: str,
    minimum_frequency: int,
) -> tuple[dict[str, int], Counter]:
    """
    Build a vocabulary using training songs only.

    Tokens are ordered by:
    1. Frequency, from most common to least common.
    2. Alphabetical order for equal frequencies.
    """

    frequencies = Counter(
        token
        for song in train_songs
        for token in song[sequence_field]
    )

    vocabulary = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    eligible_tokens = [
        token
        for token, count in frequencies.items()
        if count >= minimum_frequency
        and token not in vocabulary
    ]

    eligible_tokens.sort(
        key=lambda token: (
            -frequencies[token],
            token,
        )
    )

    for token in eligible_tokens:
        vocabulary[token] = len(vocabulary)

    return vocabulary, frequencies


def collect_tokens(
    songs: list[dict],
    sequence_field: str,
    predicate: Callable[[str], bool] | None = None,
) -> list[str]:
    """Collect tokens from a representation, optionally using a filter."""

    tokens = [
        token
        for song in songs
        for token in song[sequence_field]
    ]

    if predicate is not None:
        tokens = [
            token
            for token in tokens
            if predicate(token)
        ]

    return tokens


def oov_statistics(
    songs: list[dict],
    vocabulary: dict[str, int],
    sequence_field: str,
    target_predicate: Callable[[str], bool],
) -> dict:
    """
    Calculate OOV statistics.

    Two rates are reported:

    sequence_oov:
        OOV rate across the complete input sequences.

    prediction_target_oov:
        OOV rate only for tokens that may become prediction targets.
        Atomic targets are CHORD_* tokens.
        Factorized targets are [EVENT_*] tokens.
    """

    sequence_tokens = collect_tokens(
        songs,
        sequence_field,
    )

    target_tokens = collect_tokens(
        songs,
        sequence_field,
        target_predicate,
    )

    sequence_unknown = sum(
        token not in vocabulary
        for token in sequence_tokens
    )

    target_unknown = sum(
        token not in vocabulary
        for token in target_tokens
    )

    return {
        "sequence_oov": {
            "total_tokens": len(sequence_tokens),
            "unknown_tokens": sequence_unknown,
            "unknown_token_rate": (
                sequence_unknown / len(sequence_tokens)
                if sequence_tokens
                else 0.0
            ),
        },
        "prediction_target_oov": {
            "total_tokens": len(target_tokens),
            "unknown_tokens": target_unknown,
            "unknown_token_rate": (
                target_unknown / len(target_tokens)
                if target_tokens
                else 0.0
            ),
        },
    }


def vocabulary_statistics(
    frequencies: Counter,
    vocabulary: dict[str, int],
) -> dict:
    """Calculate basic training-vocabulary statistics."""

    unique_training_tokens = len(frequencies)

    singletons = sum(
        count == 1
        for count in frequencies.values()
    )

    tokens_below_minimum = sum(
        token not in vocabulary
        for token in frequencies
    )

    return {
        "vocabulary_size_including_special_tokens": len(vocabulary),
        "unique_training_tokens": unique_training_tokens,
        "training_singletons": singletons,
        "training_singleton_rate": (
            singletons / unique_training_tokens
            if unique_training_tokens
            else 0.0
        ),
        "training_tokens_excluded_by_minimum_frequency": (
            tokens_below_minimum
        ),
    }


def save_vocabulary_files(
    representation: str,
    vocabulary: dict[str, int],
    frequencies: Counter,
    output_directory: Path,
) -> None:
    """Save the vocabulary and full training-token frequencies."""

    save_json(
        vocabulary,
        output_directory / f"vocab_{representation}.json",
    )

    ordered_frequencies = {
        token: count
        for token, count in frequencies.most_common()
    }

    save_json(
        ordered_frequencies,
        output_directory
        / f"vocab_frequencies_{representation}.json",
    )


def main() -> None:
    """Create shared song splits and separate vocabularies."""

    args = parse_args()

    if args.train_ratio <= 0:
        raise ValueError(
            "--train-ratio must be positive."
        )

    if args.validation_ratio <= 0:
        raise ValueError(
            "--validation-ratio must be positive."
        )

    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError(
            "train-ratio + validation-ratio must be smaller than 1."
        )

    if args.minimum_frequency <= 0:
        raise ValueError(
            "--minimum-frequency must be at least 1."
        )

    all_songs = load_jsonl(args.input)

    songs = [
        song
        for song in all_songs
        if song.get(
            "usable_for_chord_model",
            False,
        )
    ]

    validate_song_fields(songs)

    # Copy the list so shuffling does not alter all_songs.
    shuffled_songs = list(songs)

    rng = random.Random(args.seed)
    rng.shuffle(shuffled_songs)

    train_size, validation_size = split_sizes(
        total=len(shuffled_songs),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    train = shuffled_songs[:train_size]

    validation = shuffled_songs[
        train_size:
        train_size + validation_size
    ]

    test = shuffled_songs[
        train_size + validation_size:
    ]

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Save the same song splits for both representations.
    for split_name, records in splits.items():
        save_jsonl(
            records,
            args.output / f"{split_name}.jsonl",
        )

        with (
            args.output / f"{split_name}_ids.txt"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in records:
                file.write(
                    record["song_id"] + "\n"
                )

    representation_summaries: dict[str, dict] = {}

    for representation, settings in REPRESENTATIONS.items():
        sequence_field = settings["sequence_field"]
        target_predicate = settings["is_prediction_target"]

        vocabulary, frequencies = build_vocabulary(
            train_songs=train,
            sequence_field=sequence_field,
            minimum_frequency=args.minimum_frequency,
        )

        save_vocabulary_files(
            representation=representation,
            vocabulary=vocabulary,
            frequencies=frequencies,
            output_directory=args.output,
        )

        representation_summaries[representation] = {
            "sequence_field": sequence_field,
            **vocabulary_statistics(
                frequencies,
                vocabulary,
            ),
            "validation": oov_statistics(
                songs=validation,
                vocabulary=vocabulary,
                sequence_field=sequence_field,
                target_predicate=target_predicate,
            ),
            "test": oov_statistics(
                songs=test,
                vocabulary=vocabulary,
                sequence_field=sequence_field,
                target_predicate=target_predicate,
            ),
        }

    summary = {
        "seed": args.seed,
        "minimum_frequency": args.minimum_frequency,
        "total_usable_songs": len(shuffled_songs),
        "train_songs": len(train),
        "validation_songs": len(validation),
        "test_songs": len(test),
        "train_ratio": len(train) / len(shuffled_songs),
        "validation_ratio": (
            len(validation) / len(shuffled_songs)
        ),
        "test_ratio": len(test) / len(shuffled_songs),
        "representations": representation_summaries,
    }

    save_json(
        summary,
        args.output / "split_summary.json",
    )

    print("\nSong-level split complete")
    print("-" * 60)

    print(f"Seed: {args.seed}")
    print(f"Total usable songs: {len(shuffled_songs)}")
    print(f"Train songs: {len(train)}")
    print(f"Validation songs: {len(validation)}")
    print(f"Test songs: {len(test)}")

    for representation, result in representation_summaries.items():
        print(f"\n{representation.capitalize()} representation")
        print("-" * 60)

        print(
            "Vocabulary size: "
            f"{result['vocabulary_size_including_special_tokens']}"
        )

        print(
            "Validation sequence OOV rate: "
            f"{result['validation']['sequence_oov']['unknown_token_rate']:.4f}"
        )

        print(
            "Validation target OOV rate: "
            f"{result['validation']['prediction_target_oov']['unknown_token_rate']:.4f}"
        )

        print(
            "Test sequence OOV rate: "
            f"{result['test']['sequence_oov']['unknown_token_rate']:.4f}"
        )

        print(
            "Test target OOV rate: "
            f"{result['test']['prediction_target_oov']['unknown_token_rate']:.4f}"
        )

    print("\nCreated files")
    print("-" * 60)

    print(args.output / "train.jsonl")
    print(args.output / "validation.jsonl")
    print(args.output / "test.jsonl")
    print(args.output / "vocab_atomic.json")
    print(args.output / "vocab_factorized.json")
    print(args.output / "split_summary.json")


if __name__ == "__main__":
    main()