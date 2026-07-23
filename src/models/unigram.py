"""Frequency-based unigram baseline for masked symbolic-music prediction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    REPRESENTATIONS,
    calculate_ranking_metrics,
    ranked_tokens_from_counts,
    read_jsonl,
    validate_representation,
    write_json,
    write_jsonl,
)


class UnigramBaseline:
    """
    Predict every masked target using the same training-frequency ranking.

    The model ignores musical context. Its purpose is to establish the
    simplest baseline that context-aware models should outperform.
    """

    def __init__(self, representation: str = "factorized") -> None:
        validate_representation(representation)

        self.representation = representation
        self.sequence_field = REPRESENTATIONS[representation]["sequence_field"]
        self.is_target = REPRESENTATIONS[representation]["is_target"]

        self.counts: Counter[str] = Counter()
        self.ranked_tokens: list[str] = []
        self.is_fitted = False

    def fit(self, train_path: str | Path) -> "UnigramBaseline":
        """Count eligible target tokens in the training songs."""
        counts: Counter[str] = Counter()

        for song in read_jsonl(train_path):
            sequence = song.get(self.sequence_field)

            if sequence is None:
                raise KeyError(
                    f"Missing field {self.sequence_field!r} in a training row"
                )

            counts.update(
                token for token in sequence if self.is_target(token)
            )

        if not counts:
            raise ValueError(
                "No eligible prediction tokens were found in the training set"
            )

        self.counts = counts
        self.ranked_tokens = ranked_tokens_from_counts(dict(counts))
        self.is_fitted = True
        return self

    def predict_top_k(self, k: int = 5) -> list[str]:
        """Return the globally most frequent training targets."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_top_k()")

        if k <= 0:
            raise ValueError("k must be positive")

        return self.ranked_tokens[:k]

    def evaluate(
        self,
        masked_examples_path: str | Path,
        predictions_output_path: str | Path | None = None,
        top_ks: tuple[int, ...] = (1, 3, 5),
    ) -> dict[str, Any]:
        """Evaluate on fixed masked examples."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before evaluate()")

        totals = {
            "examples": 0,
            "known_examples": 0,
            "unknown_examples": 0,
            "reciprocal_rank_sum": 0.0,
            "known_reciprocal_rank_sum": 0.0,
        }

        for k in top_ks:
            totals[f"top_{k}_hits"] = 0
            totals[f"known_top_{k}_hits"] = 0

        prediction_rows: list[dict[str, Any]] = []

        for example in read_jsonl(masked_examples_path):
            target = example["target_token"]
            target_is_known = bool(
                example.get(
                    "target_in_vocabulary",
                    target in self.counts,
                )
            )

            ranking = calculate_ranking_metrics(
                target_token=target,
                ranked_predictions=self.ranked_tokens,
                top_ks=top_ks,
            )

            totals["examples"] += 1
            totals["reciprocal_rank_sum"] += float(
                ranking["reciprocal_rank"]
            )

            if target_is_known:
                totals["known_examples"] += 1
                totals["known_reciprocal_rank_sum"] += float(
                    ranking["reciprocal_rank"]
                )
            else:
                totals["unknown_examples"] += 1

            for k in top_ks:
                hit = int(ranking[f"top_{k}_hit"])
                totals[f"top_{k}_hits"] += hit

                if target_is_known:
                    totals[f"known_top_{k}_hits"] += hit

            prediction_rows.append(
                {
                    "song_id": example.get("song_id"),
                    "position": example.get("position"),
                    "representation": self.representation,
                    "target_token": target,
                    "target_in_vocabulary": target_is_known,
                    "top_predictions": self.ranked_tokens[: max(top_ks)],
                    **ranking,
                }
            )

        number_of_examples = totals["examples"]
        number_of_known_examples = totals["known_examples"]

        if number_of_examples == 0:
            raise ValueError("The masked examples file contains no examples")

        metrics: dict[str, Any] = {
            "model": "unigram",
            "representation": self.representation,
            "training_target_types": len(self.counts),
            "training_target_occurrences": sum(self.counts.values()),
            "most_frequent_targets": [
                {
                    "token": token,
                    "count": self.counts[token],
                }
                for token in self.ranked_tokens[:10]
            ],
            "total_examples": number_of_examples,
            "known_examples": number_of_known_examples,
            "unknown_examples": totals["unknown_examples"],
            "mrr_overall": (
                totals["reciprocal_rank_sum"] / number_of_examples
            ),
            "mrr_known_targets": (
                totals["known_reciprocal_rank_sum"]
                / number_of_known_examples
                if number_of_known_examples
                else None
            ),
        }

        for k in top_ks:
            metrics[f"top_{k}_accuracy_overall"] = (
                totals[f"top_{k}_hits"] / number_of_examples
            )
            metrics[f"top_{k}_accuracy_known_targets"] = (
                totals[f"known_top_{k}_hits"]
                / number_of_known_examples
                if number_of_known_examples
                else None
            )

        if predictions_output_path is not None:
            write_jsonl(predictions_output_path, prediction_rows)

        return metrics

    def save(self, path: str | Path) -> None:
        """Save model counts and deterministic ranking."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before save()")

        write_json(
            path,
            {
                "model": "unigram",
                "representation": self.representation,
                "counts": dict(self.counts),
                "ranked_tokens": self.ranked_tokens,
            },
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the unigram baseline"
    )
    parser.add_argument(
        "--train",
        default="data/splits/train.jsonl",
        help="Training JSONL file",
    )
    parser.add_argument(
        "--masked",
        required=True,
        help="Fixed masked validation or test JSONL file",
    )
    parser.add_argument(
        "--representation",
        choices=sorted(REPRESENTATIONS),
        default="factorized",
    )
    parser.add_argument(
        "--output-dir",
        default="results/unigram",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = UnigramBaseline(args.representation)
    model.fit(args.train)

    metrics = model.evaluate(
        masked_examples_path=args.masked,
        predictions_output_path=output_dir / "predictions.jsonl",
    )

    model.save(output_dir / "model.json")
    write_json(output_dir / "metrics.json", metrics)

    print("Unigram evaluation complete")
    print(f"Representation: {args.representation}")
    print(f"Examples: {metrics['total_examples']}")
    print(
        "Top-1 overall accuracy: "
        f"{metrics['top_1_accuracy_overall']:.4f}"
    )
    print(
        "Top-5 overall accuracy: "
        f"{metrics['top_5_accuracy_overall']:.4f}"
    )


if __name__ == "__main__":
    main()
