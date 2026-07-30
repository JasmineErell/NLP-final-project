"""Evaluate trained LSTM models on masked validation or test examples."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.models.lstm import MusicLSTMModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM models.")
    parser.add_argument(
        "--representation",
        choices=["atomic", "factorized", "both"],
        default="factorized",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
    )
    return parser.parse_args()


def load_jsonl(relative_path: str) -> list[dict]:
    full_path = PROJECT_ROOT / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")

    data = []
    with full_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_model(
    model: MusicLSTMModel,
    examples: list[dict],
    show_samples: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "loaded_examples": len(examples),
        "evaluated_examples": 0,
        "skipped_examples": 0,
        "known_targets": 0,
        "unknown_targets": 0,
        "top_1_correct": 0,
        "top_3_correct": 0,
        "top_5_correct": 0,
    }

    reciprocal_rank_sum = 0.0
    negative_log_likelihood_sum = 0.0
    displayed = 0

    for example_index, sample in enumerate(examples):
        sequence = sample.get("input_tokens", [])
        target_index = sample.get("position", -1)
        true_token = sample.get("target_token", "")

        if not isinstance(target_index, int) and "[MASK]" in sequence:
            target_index = sequence.index("[MASK]")

        if (
            not sequence
            or not isinstance(target_index, int)
            or target_index < 0
            or target_index >= len(sequence)
            or not true_token
        ):
            metrics["skipped_examples"] += 1
            continue

        probabilities = model.predict_token_probabilities(sequence, target_index)
        ranked_predictions = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked_tokens = [token for token, _ in ranked_predictions]

        metrics["evaluated_examples"] += 1

        if true_token in probabilities:
            metrics["known_targets"] += 1
            rank = ranked_tokens.index(true_token) + 1
            reciprocal_rank_sum += 1.0 / rank

            true_probability = max(probabilities[true_token], 1e-12)
            negative_log_likelihood_sum += -math.log(true_probability)
        else:
            metrics["unknown_targets"] += 1
            rank = None

        if ranked_tokens and true_token == ranked_tokens[0]:
            metrics["top_1_correct"] += 1
        if true_token in ranked_tokens[:3]:
            metrics["top_3_correct"] += 1
        if true_token in ranked_tokens[:5]:
            metrics["top_5_correct"] += 1

        if displayed < show_samples:
            left_context = sequence[max(0, target_index - 4) : target_index]
            print(f"\n  Sample {displayed + 1}")
            print(f"  Left context: {left_context} [MASK]")
            print(f"  Target: {true_token}")
            print(f"  Target rank: {rank if rank is not None else 'OOV'}")
            print("  Top 5 Predictions:")
            for prediction_rank, (token, probability) in enumerate(
                ranked_predictions[:5],
                start=1,
            ):
                print(
                    f"    {prediction_rank}. {token:<25} "
                    f"({probability:.4f})"
                )
            displayed += 1

        if (example_index + 1) % 500 == 0:
            print(f"  Processed {example_index + 1}/{len(examples)} examples...")

    evaluated = metrics["evaluated_examples"]
    known = metrics["known_targets"]

    if evaluated == 0:
        raise ValueError("No valid evaluation examples were found.")

    metrics["target_oov_rate"] = metrics["unknown_targets"] / evaluated
    metrics["top_1_accuracy"] = metrics["top_1_correct"] / evaluated
    metrics["top_3_accuracy"] = metrics["top_3_correct"] / evaluated
    metrics["top_5_accuracy"] = metrics["top_5_correct"] / evaluated
    metrics["mean_reciprocal_rank"] = reciprocal_rank_sum / evaluated

    if known > 0:
        average_nll = negative_log_likelihood_sum / known
        metrics["known_target_average_nll"] = average_nll
        metrics["known_target_perplexity"] = math.exp(average_nll)
    else:
        metrics["known_target_average_nll"] = None
        metrics["known_target_perplexity"] = None

    return metrics


def evaluate_representation(
    version: str,
    split: str,
    show_samples: int,
    device: str | None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"EVALUATING LSTM: {version.upper()} ({split.upper()})")
    print(f"{'=' * 60}")

    checkpoint_path = PROJECT_ROOT / "results" / "lstm" / version / "checkpoint.pt"
    masked_path = f"data/splits/{split}_masked_{version}.jsonl"

    print(f"Loading checkpoint: {checkpoint_path}")
    model = MusicLSTMModel.load(checkpoint_path, device=device)
    print(f"Device: {model.device}")
    print(f"Valid prediction targets: {model.target_prefix}*")

    print(f"Loading evaluation data: {masked_path}")
    examples = load_jsonl(masked_path)

    metrics = evaluate_model(model, examples, show_samples)

    output_path = (
        PROJECT_ROOT
        / "results"
        / "lstm"
        / version
        / f"{split}_metrics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nFull Evaluation Results:")
    print(f"  Loaded examples:   {metrics['loaded_examples']}")
    print(f"  Evaluated examples:{metrics['evaluated_examples']}")
    print(f"  Skipped examples:  {metrics['skipped_examples']}")
    print(f"  Target OOV rate:   {metrics['target_oov_rate']:.4f}")
    print(f"  Top-1 accuracy:    {metrics['top_1_accuracy']:.4f}")
    print(f"  Top-3 accuracy:    {metrics['top_3_accuracy']:.4f}")
    print(f"  Top-5 accuracy:    {metrics['top_5_accuracy']:.4f}")
    print(f"  MRR:               {metrics['mean_reciprocal_rank']:.4f}")
    if metrics["known_target_perplexity"] is not None:
        print(
            "  Known-target PPL:  "
            f"{metrics['known_target_perplexity']:.4f}"
        )
    print(f"Metrics saved to: {output_path}")


def main() -> None:
    args = parse_args()
    versions = (
        ["atomic", "factorized"]
        if args.representation == "both"
        else [args.representation]
    )

    for version in versions:
        evaluate_representation(
            version=version,
            split=args.split,
            show_samples=args.show_samples,
            device=args.device,
        )


if __name__ == "__main__":
    main()