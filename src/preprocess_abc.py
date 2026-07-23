"""Convert Nottingham ABC files into atomic and factorized token sequences."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import traceback
from abc_utils import (
    iter_abc_files,
    parse_tune,
    split_abc_text,
    token_statistics,
    tune_to_dict,
)

from factorized_tokenizer import create_factorized_sequence


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Parse ABC tunes and create atomic and factorized "
            "musical token sequences."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ABC_cleaned"),
        help=(
            "Directory containing the Nottingham .abc files "
            "(default: ABC_cleaned)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help=(
            "Directory in which processed files will be saved "
            "(default: data/processed)."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of tunes to process. "
            "Useful for a small pilot run."
        ),
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help=(
            "Number of original and tokenized examples to save "
            "(default: 10)."
        ),
    )

    return parser.parse_args()


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save a list of dictionaries in JSON Lines format."""

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def save_frequency_csv(
    frequencies: Counter,
    output_path: Path,
) -> None:
    """Save token frequencies as a CSV file."""

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            [
                "token",
                "count",
            ]
        )

        for token, count in frequencies.most_common():
            writer.writerow(
                [
                    token,
                    count,
                ]
            )


def calculate_sequence_statistics(
    songs: list[dict],
    sequence_field: str,
) -> tuple[Counter, dict]:
    """
    Calculate vocabulary statistics for one token representation.

    sequence_field can be:
    - atomic_sequence_tokens
    - factorized_sequence_tokens
    """

    usable_songs = [
        song
        for song in songs
        if song["usable_for_chord_model"]
    ]

    frequencies = Counter(
        token
        for song in usable_songs
        for token in song.get(sequence_field, [])
    )

    total_tokens = sum(frequencies.values())
    unique_tokens = len(frequencies)

    singleton_count = sum(
        1
        for count in frequencies.values()
        if count == 1
    )

    rare_under_five = sum(
        1
        for count in frequencies.values()
        if count < 5
    )

    sequence_lengths = [
        len(song.get(sequence_field, []))
        for song in usable_songs
    ]

    average_length = (
        sum(sequence_lengths) / len(sequence_lengths)
        if sequence_lengths
        else 0.0
    )

    statistics = {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "tokens_appearing_once": singleton_count,
        "tokens_appearing_fewer_than_five_times": (
            rare_under_five
        ),
        "singleton_rate": (
            singleton_count / unique_tokens
            if unique_tokens
            else 0.0
        ),
        "average_sequence_length": average_length,
        "minimum_sequence_length": (
            min(sequence_lengths)
            if sequence_lengths
            else 0
        ),
        "maximum_sequence_length": (
            max(sequence_lengths)
            if sequence_lengths
            else 0
        ),
    }

    return frequencies, statistics


def save_examples(
    songs: list[dict],
    output_path: Path,
    number_of_examples: int,
) -> None:
    """
    Save readable examples for manual validation.

    Each example includes:
    - Metadata
    - Original ABC body
    - Structured chord-melody segments
    - Atomic representation
    - Factorized representation
    """

    usable_songs = [
        song
        for song in songs
        if song["usable_for_chord_model"]
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for song in usable_songs[:number_of_examples]:
            output_file.write("=" * 80 + "\n")

            output_file.write(
                f"SONG ID: {song['song_id']}\n"
            )

            output_file.write(
                f"TITLE: {song['title']}\n"
            )

            output_file.write(
                f"KEY: {song['key']}\n"
            )

            output_file.write(
                f"RHYTHM: {song['rhythm']}\n"
            )

            output_file.write(
                f"METER: {song['meter']}\n"
            )

            output_file.write(
                "DEFAULT NOTE LENGTH: "
                f"{song['default_note_length']}\n\n"
            )

            output_file.write("ORIGINAL ABC BODY:\n")
            output_file.write(song["raw_body"] + "\n\n")

            output_file.write("STRUCTURED SEGMENTS:\n")

            for index, segment in enumerate(
                song["segments"],
                start=1,
            ):
                output_file.write(
                    f"\nSegment {index}\n"
                )

                output_file.write(
                    f"Chord: {segment['chord']}\n"
                )

                output_file.write(
                    "Raw melody: "
                    f"{segment['melody_raw']}\n"
                )

                output_file.write(
                    "Normalized melody: "
                    f"{segment['melody_normalized']}\n"
                )

                output_file.write(
                    "Atomic token: "
                    f"{segment['atomic_token']}\n"
                )

            output_file.write(
                "\nATOMIC SEQUENCE TOKENS:\n"
            )

            for token in song["atomic_sequence_tokens"]:
                output_file.write(token + "\n")

            output_file.write(
                "\nFACTORIZED SEQUENCE TOKENS:\n"
            )

            for token in song["factorized_sequence_tokens"]:
                output_file.write(token + "\n")

            output_file.write("\n\n")


def process_song(
    tune_text: str,
    source_filename: str,
) -> dict:
    """
    Parse one tune and create both token representations.

    Representation A:
    One complete chord-melody segment is one atomic token.

    Representation B:
    Chords and individual musical events are separate tokens.
    """

    parsed_tune = parse_tune(
        tune_text,
        source_filename,
    )

    song = tune_to_dict(parsed_tune)

    # Representation A:
    # Metadata followed by complete chord-melody tokens.
    song["atomic_sequence_tokens"] = list(
        song["sequence_tokens"]
    )

    # Representation B:
    # Metadata, chord tokens and individual melody-event tokens.
    if song["usable_for_chord_model"]:
        song["factorized_sequence_tokens"] = (
            create_factorized_sequence(song)
        )
    else:
        song["factorized_sequence_tokens"] = []

    return song


def main() -> None:
    """Run the full ABC preprocessing stage."""

    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    if args.examples < 0:
        raise ValueError(
            "--examples cannot be negative."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    songs: list[dict] = []
    errors: list[dict] = []

    processed_count = 0

    for abc_path in iter_abc_files(args.input):
        abc_text = abc_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tune_texts = split_abc_text(abc_text)

        for tune_index, tune_text in enumerate(
            tune_texts,
            start=1,
        ):
            if (
                args.limit is not None
                and processed_count >= args.limit
            ):
                break

            try:
                song = process_song(
                    tune_text,
                    abc_path.name,
                )

                songs.append(song)

            except Exception as error:
                errors.append(
                    {
                        "source_file": abc_path.name,
                        "tune_index_in_file": tune_index,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )

            processed_count += 1

        if (
            args.limit is not None
            and processed_count >= args.limit
        ):
            break

    # Existing atomic chord-segment statistics from abc_utils.py.
    _, general_statistics = token_statistics(songs)

    atomic_frequencies, atomic_statistics = (
        calculate_sequence_statistics(
            songs,
            "atomic_sequence_tokens",
        )
    )

    factorized_frequencies, factorized_statistics = (
        calculate_sequence_statistics(
            songs,
            "factorized_sequence_tokens",
        )
    )

    statistics = {
        **general_statistics,
        "parse_errors": len(errors),
        "input_directory": str(args.input),
        "output_directory": str(args.output),
        "atomic_representation": atomic_statistics,
        "factorized_representation": (
            factorized_statistics
        ),
    }

    save_jsonl(
        songs,
        args.output / "songs.jsonl",
    )

    save_jsonl(
        errors,
        args.output / "errors.jsonl",
    )

    save_examples(
        songs,
        args.output / "tokenization_examples.txt",
        args.examples,
    )

    save_frequency_csv(
        atomic_frequencies,
        args.output / "token_frequencies_atomic.csv",
    )

    save_frequency_csv(
        factorized_frequencies,
        args.output / "token_frequencies_factorized.csv",
    )

    with (
        args.output / "stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            statistics,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nPreprocessing complete")
    print("-" * 60)

    print(
        f"Total songs: "
        f"{statistics['total_songs']}"
    )

    print(
        f"Usable songs: "
        f"{statistics['usable_songs']}"
    )

    print(
        f"Songs without chords: "
        f"{statistics['songs_without_chords']}"
    )

    print(
        f"Parse errors: "
        f"{statistics['parse_errors']}"
    )

    print("\nAtomic representation")
    print("-" * 60)

    for key, value in atomic_statistics.items():
        print(f"{key}: {value}")

    print("\nFactorized representation")
    print("-" * 60)

    for key, value in factorized_statistics.items():
        print(f"{key}: {value}")

    print("\nCreated files")
    print("-" * 60)

    print(
        args.output / "songs.jsonl"
    )

    print(
        args.output / "stats.json"
    )

    print(
        args.output / "token_frequencies_atomic.csv"
    )

    print(
        args.output
        / "token_frequencies_factorized.csv"
    )

    print(
        args.output / "tokenization_examples.txt"
    )

    print(
        args.output / "errors.jsonl"
    )


if __name__ == "__main__":
    main()