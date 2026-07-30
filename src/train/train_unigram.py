import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable


REPRESENTATIONS = {
    "atomic": {
        "sequence_field": "atomic_sequence_tokens",
        "is_target": lambda token: token.startswith("CHORD_"),
    },
    "factorized": {
        "sequence_field": "factorized_sequence_tokens",
        "is_target": lambda token: token.startswith("[EVENT_"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a unigram model on the training split."
    )

    parser.add_argument(
        "--representation",
        choices=["atomic", "factorized"],
        default="factorized",
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/splits/train.jsonl"),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=1e-5,
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def calculate_probabilities(
    songs: list[dict],
    sequence_field: str,
    is_target: Callable[[str], bool],
    smoothing: float,
) -> tuple[Counter, dict[str, float]]:
    """
    Count eligible target tokens and calculate unigram probabilities.
    """

    counts: Counter = Counter()

    for song in songs:
        sequence = song[sequence_field]

        for token in sequence:
            if is_target(token):
                counts[token] += 1

    if not counts:
        raise ValueError(
            f"No target tokens were found in '{sequence_field}'."
        )

    total_tokens = sum(counts.values())
    vocabulary_size = len(counts)

    denominator = (
        total_tokens
        + smoothing * vocabulary_size
    )

    probabilities = {
        token: (count + smoothing) / denominator
        for token, count in counts.items()
    }

    return counts, probabilities


def main() -> None:
    args = parse_args()

    settings = REPRESENTATIONS[args.representation]
    sequence_field = settings["sequence_field"]
    is_target = settings["is_target"]

    output_file = args.output_file

    if output_file is None:
        output_file = Path(
            f"results/unigram/"
            f"train_{args.representation}_model.json"
        )

    print(f"Loading training data from: {args.train_file}")

    songs = load_jsonl(args.train_file)

    counts, probabilities = calculate_probabilities(
        songs=songs,
        sequence_field=sequence_field,
        is_target=is_target,
        smoothing=args.smoothing,
    )

    model_data = {
        "model": "unigram",
        "representation": args.representation,
        "sequence_field": sequence_field,
        "smoothing": args.smoothing,
        "number_of_songs": len(songs),
        "total_target_tokens": sum(counts.values()),
        "vocabulary_size": len(counts),
        "token_counts": dict(counts),
        "probabilities": probabilities,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            model_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    most_common = counts.most_common(5)

    print("Unigram training complete")
    print(f"Representation: {args.representation}")
    print(f"Training songs: {len(songs)}")
    print(f"Target tokens: {sum(counts.values())}")
    print(f"Vocabulary size: {len(counts)}")
    print("Most common tokens:")

    for token, count in most_common:
        print(
            f"  {token}: "
            f"count={count}, "
            f"probability={probabilities[token]:.6f}"
        )

    print(f"Model saved to: {output_file}")


if __name__ == "__main__":
    main()