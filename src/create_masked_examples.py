"""Create fixed single-mask validation and test examples for both representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Callable


REPRESENTATIONS = {
    "atomic": {
        "sequence_field": "atomic_sequence_tokens",
        "vocabulary_file": "vocab_atomic.json",
        "is_maskable": lambda token: token.startswith("CHORD_"),
    },
    "factorized": {
        "sequence_field": "factorized_sequence_tokens",
        "vocabulary_file": "vocab_factorized.json",
        "is_maskable": lambda token: token.startswith("[EVENT_"),
    },
}


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible masked validation and test examples "
            "for atomic and factorized musical representations."
        )
    )

    parser.add_argument(
        "--splits-directory",
        type=Path,
        default=Path("data/splits"),
        help=(
            "Directory containing train, validation, test and "
            "vocabulary files."
        ),
    )

    parser.add_argument(
        "--mask-rate",
        type=float,
        default=0.15,
        help=(
            "Percentage of eligible positions selected from each song. "
            "Each saved example contains exactly one masked position."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to select fixed mask positions.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSON Lines file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Run create_splits.py first."
        )

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_json(path: Path) -> dict:
    """Load a regular JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing vocabulary file: {path}. "
            "Run the updated create_splits.py first."
        )

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_jsonl(
    records: list[dict],
    path: Path,
) -> None:
    """Save records using JSON Lines format."""

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
    """Save formatted JSON."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def validate_sequence_field(
    songs: list[dict],
    sequence_field: str,
) -> None:
    """Verify that every song contains the requested token sequence."""

    for song in songs:
        if sequence_field not in song:
            raise KeyError(
                f"Song {song.get('song_id', 'UNKNOWN')} is missing "
                f"'{sequence_field}'. Run preprocess_abc.py again."
            )


def select_mask_positions(
    tokens: list[str],
    mask_rate: float,
    rng: random.Random,
    is_maskable: Callable[[str], bool],
) -> list[int]:
    """Select deterministic eligible mask positions for one song."""

    eligible_positions = [
        index
        for index, token in enumerate(tokens)
        if is_maskable(token)
    ]

    if not eligible_positions:
        return []

    number_to_mask = max(
        1,
        round(len(eligible_positions) * mask_rate),
    )

    number_to_mask = min(
        number_to_mask,
        len(eligible_positions),
    )

    return sorted(
        rng.sample(
            eligible_positions,
            number_to_mask,
        )
    )


def create_examples(
    songs: list[dict],
    representation: str,
    sequence_field: str,
    vocabulary: dict[str, int],
    mask_rate: float,
    rng: random.Random,
    is_maskable: Callable[[str], bool],
) -> list[dict]:
    """
    Create fixed masked examples.

    Several positions may be selected from one song, but every saved example
    contains exactly one [MASK] token. This makes evaluation straightforward
    and allows all models to receive the same context and target.
    """

    examples: list[dict] = []

    for song in songs:
        tokens = song[sequence_field]

        selected_positions = select_mask_positions(
            tokens=tokens,
            mask_rate=mask_rate,
            rng=rng,
            is_maskable=is_maskable,
        )

        for position in selected_positions:
            target_token = tokens[position]

            masked_tokens = list(tokens)
            masked_tokens[position] = "[MASK]"

            target_in_vocabulary = (
                target_token in vocabulary
            )

            examples.append(
                {
                    "song_id": song["song_id"],
                    "title": song.get("title"),
                    "representation": representation,
                    "sequence_field": sequence_field,
                    "position": position,
                    "sequence_length": len(tokens),
                    "target_token": target_token,
                    "target_token_id": (
                        vocabulary[target_token]
                        if target_in_vocabulary
                        else vocabulary["[UNK]"]
                    ),
                    "target_in_vocabulary": (
                        target_in_vocabulary
                    ),
                    "input_tokens": masked_tokens,
                    "left_context": tokens[:position],
                    "right_context": tokens[position + 1 :],
                    "key": song.get("key"),
                    "meter": song.get("meter"),
                    "rhythm": song.get("rhythm"),
                    "default_note_length": song.get(
                        "default_note_length"
                    ),
                }
            )

    return examples


def summarize_examples(
    examples: list[dict],
) -> dict:
    """Calculate summary statistics for masked examples."""

    total_examples = len(examples)

    known_targets = sum(
        example["target_in_vocabulary"]
        for example in examples
    )

    unknown_targets = (
        total_examples - known_targets
    )

    return {
        "total_examples": total_examples,
        "known_targets": known_targets,
        "unknown_targets": unknown_targets,
        "target_oov_rate": (
            unknown_targets / total_examples
            if total_examples
            else 0.0
        ),
    }


def main() -> None:
    """Create fixed masked examples for every split and representation."""

    args = parse_args()

    if not 0 < args.mask_rate <= 1:
        raise ValueError(
            "--mask-rate must be greater than 0 and at most 1."
        )

    summary = {
        "seed": args.seed,
        "mask_rate": args.mask_rate,
        "representations": {},
    }

    split_offsets = {
        "validation": 0,
        "test": 1000,
    }

    representation_offsets = {
        "atomic": 0,
        "factorized": 100,
    }

    for representation, settings in REPRESENTATIONS.items():
        sequence_field = settings["sequence_field"]
        is_maskable = settings["is_maskable"]

        vocabulary_path = (
            args.splits_directory
            / settings["vocabulary_file"]
        )

        vocabulary = load_json(vocabulary_path)

        if "[UNK]" not in vocabulary:
            raise KeyError(
                f"The vocabulary {vocabulary_path} "
                "does not contain [UNK]."
            )

        summary["representations"][representation] = {}

        for split_name in ("validation", "test"):
            input_path = (
                args.splits_directory
                / f"{split_name}.jsonl"
            )

            songs = load_jsonl(input_path)

            validate_sequence_field(
                songs,
                sequence_field,
            )

            rng = random.Random(
                args.seed
                + split_offsets[split_name]
                + representation_offsets[representation]
            )

            examples = create_examples(
                songs=songs,
                representation=representation,
                sequence_field=sequence_field,
                vocabulary=vocabulary,
                mask_rate=args.mask_rate,
                rng=rng,
                is_maskable=is_maskable,
            )

            output_path = (
                args.splits_directory
                / f"{split_name}_masked_"
                  f"{representation}.jsonl"
            )

            save_jsonl(
                examples,
                output_path,
            )

            example_summary = summarize_examples(
                examples
            )

            summary["representations"][
                representation
            ][split_name] = example_summary

            print(
                f"{representation} {split_name}: "
                f"created {example_summary['total_examples']} "
                f"fixed masked examples"
            )

            print(
                f"  Known targets: "
                f"{example_summary['known_targets']}"
            )

            print(
                f"  Unknown targets: "
                f"{example_summary['unknown_targets']}"
            )

            print(
                f"  Target OOV rate: "
                f"{example_summary['target_oov_rate']:.4f}"
            )

            print(
                f"  Output: {output_path}\n"
            )

    summary_path = (
        args.splits_directory
        / "masked_examples_summary.json"
    )

    save_json(
        summary,
        summary_path,
    )

    print("Masked-example creation complete")
    print("-" * 60)

    print(
        args.splits_directory
        / "validation_masked_atomic.jsonl"
    )

    print(
        args.splits_directory
        / "test_masked_atomic.jsonl"
    )

    print(
        args.splits_directory
        / "validation_masked_factorized.jsonl"
    )

    print(
        args.splits_directory
        / "test_masked_factorized.jsonl"
    )

    print(summary_path)


if __name__ == "__main__":
    main()