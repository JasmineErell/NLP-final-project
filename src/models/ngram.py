"""n-gram baseline for masked symbolic-music prediction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


class NGramBaseline:
    """
    Predict a target token using only tokens that appear before it.

    Definitions:
        order=1: unigram, no context
        order=2: bigram, one previous token
        order=3: trigram, two previous tokens
        order=4: 4-gram, three previous tokens

    This class is intended for order >= 2 because the separate
    UnigramBaseline already handles order 1.

    When the full context was never observed in training, the model backs off
    to a shorter left context and finally to global unigram frequencies.
    """

    def __init__(
        self,
        representation: str = "factorized",
        order: int = 3,
    ) -> None:
        validate_representation(representation)

        if order < 2:
            raise ValueError(
                "NGramBaseline requires order >= 2. "
                "Use UnigramBaseline for order 1."
            )

        self.representation = representation
        self.order = order
        self.maximum_context_length = order - 1
        self.sequence_field = REPRESENTATIONS[representation]["sequence_field"]
        self.is_target = REPRESENTATIONS[representation]["is_target"]

        self.context_counts: dict[
            tuple[str, ...],
            Counter[str],
        ] = defaultdict(Counter)

        self.global_counts: Counter[str] = Counter()
        self.global_ranking: list[str] = []
        self.is_fitted = False

    @staticmethod
    def _left_context(
        sequence: list[str],
        position: int,
        context_length: int,
    ) -> tuple[str, ...]:
        """Return up to context_length tokens immediately before position."""
        start = max(0, position - context_length)
        return tuple(sequence[start:position])

    def fit(self, train_path: str | Path) -> "NGramBaseline":
        """
        Count target frequencies after each observed left context.

        For a trigram model, for example, the training pattern is:

            previous_2 previous_1 -> target
        """
        for song in read_jsonl(train_path):
            sequence = song.get(self.sequence_field)

            if sequence is None:
                raise KeyError(
                    f"Missing field {self.sequence_field!r} in a training row"
                )

            for position, target_token in enumerate(sequence):
                if not self.is_target(target_token):
                    continue

                self.global_counts[target_token] += 1

                for context_length in range(
                    1,
                    self.maximum_context_length + 1,
                ):
                    context = self._left_context(
                        sequence=sequence,
                        position=position,
                        context_length=context_length,
                    )

                    # Near the beginning of a song, fewer context tokens may
                    # be available. Avoid storing duplicate shorter contexts.
                    if len(context) != context_length:
                        continue

                    self.context_counts[context][target_token] += 1

        if not self.global_counts:
            raise ValueError(
                "No eligible prediction targets were found in training"
            )

        self.global_ranking = ranked_tokens_from_counts(
            dict(self.global_counts)
        )
        self.is_fitted = True
        return self

    def rank_predictions(
        self,
        masked_sequence: list[str],
        mask_position: int,
    ) -> tuple[list[str], int]:
        """
        Rank targets using only context before the mask.

        Returns:
            ranked predictions and the context length actually used.
            A used context length of 0 means unigram fallback.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before rank_predictions()")

        maximum_available = min(
            self.maximum_context_length,
            mask_position,
        )

        for context_length in range(maximum_available, 0, -1):
            context = self._left_context(
                sequence=masked_sequence,
                position=mask_position,
                context_length=context_length,
            )
            counts = self.context_counts.get(context)

            if counts:
                context_ranking = ranked_tokens_from_counts(dict(counts))
                context_tokens = set(context_ranking)

                # Tokens never seen after this context are appended using
                # global unigram frequency, producing a complete ranking.
                full_ranking = context_ranking + [
                    token
                    for token in self.global_ranking
                    if token not in context_tokens
                ]

                return full_ranking, context_length

        return self.global_ranking, 0

    def evaluate(
        self,
        masked_examples_path: str | Path,
        predictions_output_path: str | Path | None = None,
        top_ks: tuple[int, ...] = (1, 3, 5),
    ) -> dict[str, Any]:
        """Evaluate on the project's fixed masked examples."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before evaluate()")

        total = 0
        known = 0
        unknown = 0
        reciprocal_rank_sum = 0.0
        known_reciprocal_rank_sum = 0.0

        backoff_counts: Counter[int] = Counter()
        top_hits: Counter[int] = Counter()
        known_top_hits: Counter[int] = Counter()
        prediction_rows: list[dict[str, Any]] = []

        for example in read_jsonl(masked_examples_path):
            sequence = list(example["input_tokens"])

            try:
                mask_position = sequence.index("[MASK]")
            except ValueError as error:
                raise ValueError(
                    "A masked example does not contain [MASK]"
                ) from error

            target_token = example["target_token"]
            target_is_known = bool(
                example.get(
                    "target_in_vocabulary",
                    target_token in self.global_counts,
                )
            )

            ranking, used_context_length = self.rank_predictions(
                masked_sequence=sequence,
                mask_position=mask_position,
            )

            result = calculate_ranking_metrics(
                target_token=target_token,
                ranked_predictions=ranking,
                top_ks=top_ks,
            )

            total += 1
            backoff_counts[used_context_length] += 1
            reciprocal_rank_sum += float(result["reciprocal_rank"])

            if target_is_known:
                known += 1
                known_reciprocal_rank_sum += float(
                    result["reciprocal_rank"]
                )
            else:
                unknown += 1

            for k in top_ks:
                hit = int(result[f"top_{k}_hit"])
                top_hits[k] += hit

                if target_is_known:
                    known_top_hits[k] += hit

            prediction_rows.append(
                {
                    "song_id": example.get("song_id"),
                    "position": example.get("position"),
                    "representation": self.representation,
                    "model_order": self.order,
                    "target_token": target_token,
                    "target_in_vocabulary": target_is_known,
                    "used_left_context_length": used_context_length,
                    "top_predictions": ranking[: max(top_ks)],
                    **result,
                }
            )

        if total == 0:
            raise ValueError("The masked examples file contains no examples")

        metrics: dict[str, Any] = {
            "model": "ngram",
            "direction": "left_to_right",
            "representation": self.representation,
            "order": self.order,
            "maximum_left_context_length": self.maximum_context_length,
            "total_examples": total,
            "known_examples": known,
            "unknown_examples": unknown,
            "backoff_counts_by_left_context_length": {
                str(length): count
                for length, count in sorted(backoff_counts.items())
            },
            "mrr_overall": reciprocal_rank_sum / total,
            "mrr_known_targets": (
                known_reciprocal_rank_sum / known if known else None
            ),
        }

        for k in top_ks:
            metrics[f"top_{k}_accuracy_overall"] = top_hits[k] / total
            metrics[f"top_{k}_accuracy_known_targets"] = (
                known_top_hits[k] / known if known else None
            )

        if predictions_output_path is not None:
            write_jsonl(predictions_output_path, prediction_rows)

        return metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a standard left-to-right n-gram"
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
        "--order",
        type=int,
        default=3,
        help=(
            "N-gram order: 2=bigram, 3=trigram, "
            "4=four-gram, and so on"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/ngram",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = NGramBaseline(
        representation=args.representation,
        order=args.order,
    )
    model.fit(args.train)

    metrics = model.evaluate(
        masked_examples_path=args.masked,
        predictions_output_path=output_dir / "predictions.jsonl",
    )

    write_json(output_dir / "metrics.json", metrics)

    print("N-gram evaluation complete")
    print("Direction: left-to-right")
    print(f"Representation: {args.representation}")
    print(f"Order: {args.order}")
    print(f"Examples: {metrics['total_examples']}")
    print(
        "Top-1 overall accuracy: "
        f"{metrics['top_1_accuracy_overall']:.4f}"
    )


if __name__ == "__main__":
    main()
