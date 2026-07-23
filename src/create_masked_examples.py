"""Create fixed single-mask validation and test examples for fair evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible masked examples from validation/test songs."
    )
    parser.add_argument(
        "--splits-directory",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument(
        "--mask-rate",
        type=float,
        default=0.15,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_examples(
    songs: list[dict],
    mask_rate: float,
    rng: random.Random,
) -> list[dict]:
    examples: list[dict] = []

    for song in songs:
        tokens = song["sequence_tokens"]
        eligible_positions = [
            index
            for index, token in enumerate(tokens)
            if token.startswith("CHORD_")
        ]

        if not eligible_positions:
            continue

        number_to_mask = max(
            1,
            round(len(eligible_positions) * mask_rate),
        )
        number_to_mask = min(number_to_mask, len(eligible_positions))

        selected = sorted(
            rng.sample(eligible_positions, number_to_mask)
        )

        # One masked position per example makes all baselines easy to compare.
        for position in selected:
            masked_tokens = list(tokens)
            target = masked_tokens[position]
            masked_tokens[position] = "[MASK]"

            examples.append(
                {
                    "song_id": song["song_id"],
                    "title": song["title"],
                    "position": position,
                    "target_token": target,
                    "input_tokens": masked_tokens,
                    "left_context": tokens[:position],
                    "right_context": tokens[position + 1 :],
                }
            )

    return examples


def main() -> None:
    args = parse_args()

    if not 0 < args.mask_rate <= 1:
        raise ValueError("mask-rate must be between 0 and 1.")

    for offset, split_name in enumerate(("validation", "test")):
        input_path = args.splits_directory / f"{split_name}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {input_path}. "
                "Run create_splits.py first."
            )

        songs = load_jsonl(input_path)
        rng = random.Random(args.seed + offset)
        examples = create_examples(songs, args.mask_rate, rng)

        output_path = (
            args.splits_directory / f"{split_name}_masked.jsonl"
        )
        save_jsonl(examples, output_path)

        print(
            f"{split_name}: created {len(examples)} fixed masked "
            f"examples at {output_path}"
        )


if __name__ == "__main__":
    main()
