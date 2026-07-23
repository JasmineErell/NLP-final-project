"""Train and compare unigram and standard left-to-right n-gram baselines.

Run from the project root:

    python3 -m src.evaluate_unigram_ngram

Defaults:
    - factorized representation
    - validation split
    - bigram, trigram, and 4-gram
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .models.ngram import NGramBaseline
from .models.unigram import UnigramBaseline


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def ngram_name(order: int) -> str:
    names = {
        1: "Unigram",
        2: "Bigram",
        3: "Trigram",
        4: "4-gram",
    }
    return names.get(order, f"{order}-gram")


def evaluate_unigram(
    train_path: Path,
    masked_path: Path,
    representation: str,
    split: str,
    results_root: Path,
) -> dict[str, Any]:
    output_dir = results_root / "unigram" / f"{split}_{representation}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = UnigramBaseline(representation=representation)
    model.fit(train_path)

    metrics = model.evaluate(
        masked_examples_path=masked_path,
        predictions_output_path=output_dir / "predictions.jsonl",
    )

    model.save(output_dir / "model.json")
    write_json(output_dir / "metrics.json", metrics)

    return {
        "model_name": "Unigram",
        "model_type": "unigram",
        "direction": "none",
        "representation": representation,
        "split": split,
        "order": 1,
        "metrics_path": str(output_dir / "metrics.json"),
        **metrics,
    }


def evaluate_ngram(
    train_path: Path,
    masked_path: Path,
    representation: str,
    split: str,
    order: int,
    results_root: Path,
) -> dict[str, Any]:
    output_dir = (
        results_root
        / "ngram"
        / f"order_{order}_{split}_{representation}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = NGramBaseline(
        representation=representation,
        order=order,
    )
    model.fit(train_path)

    metrics = model.evaluate(
        masked_examples_path=masked_path,
        predictions_output_path=output_dir / "predictions.jsonl",
    )

    write_json(output_dir / "metrics.json", metrics)

    return {
        "model_name": ngram_name(order),
        "model_type": "ngram",
        "direction": "left_to_right",
        "representation": representation,
        "split": split,
        "order": order,
        "metrics_path": str(output_dir / "metrics.json"),
        **metrics,
    }


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'Model':<16}"
        f"{'Representation':<18}"
        f"{'Top-1':>10}"
        f"{'Top-3':>10}"
        f"{'Top-5':>10}"
        f"{'MRR':>10}"
        f"{'Known T1':>12}"
    )
    print("-" * 86)

    for row in rows:
        print(
            f"{row['model_name']:<16}"
            f"{row['representation']:<18}"
            f"{row['top_1_accuracy_overall']:>10.4f}"
            f"{row['top_3_accuracy_overall']:>10.4f}"
            f"{row['top_5_accuracy_overall']:>10.4f}"
            f"{row['mrr_overall']:>10.4f}"
            f"{row['top_1_accuracy_known_targets']:>12.4f}"
        )


def save_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "model_type",
        "direction",
        "representation",
        "split",
        "order",
        "total_examples",
        "known_examples",
        "unknown_examples",
        "top_1_accuracy_overall",
        "top_3_accuracy_overall",
        "top_5_accuracy_overall",
        "mrr_overall",
        "top_1_accuracy_known_targets",
        "top_3_accuracy_known_targets",
        "top_5_accuracy_known_targets",
        "mrr_known_targets",
        "metrics_path",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {field: row.get(field) for field in fieldnames}
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare unigram and standard left-to-right "
            "n-gram baselines."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/train.jsonl"),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=("atomic", "factorized"),
        default=["factorized"],
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="N-gram orders to evaluate, for example 2 3 4",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.train.exists():
        raise FileNotFoundError(f"Training file not found: {args.train}")

    if any(order < 2 for order in args.orders):
        raise ValueError(
            "N-gram orders must be at least 2. "
            "The unigram is evaluated separately."
        )

    rows: list[dict[str, Any]] = []

    for representation in args.representations:
        masked_path = Path(
            f"data/splits/{args.split}_masked_{representation}.jsonl"
        )

        if not masked_path.exists():
            raise FileNotFoundError(
                f"Masked examples file not found: {masked_path}"
            )

        print(
            f"\nEvaluating {representation} representation "
            f"on {args.split}..."
        )

        unigram_row = evaluate_unigram(
            train_path=args.train,
            masked_path=masked_path,
            representation=representation,
            split=args.split,
            results_root=args.results_root,
        )
        rows.append(unigram_row)

        print(
            "  Unigram complete: "
            f"Top-1={unigram_row['top_1_accuracy_overall']:.4f}"
        )

        for order in sorted(set(args.orders)):
            ngram_row = evaluate_ngram(
                train_path=args.train,
                masked_path=masked_path,
                representation=representation,
                split=args.split,
                order=order,
                results_root=args.results_root,
            )
            rows.append(ngram_row)

            print(
                f"  {ngram_name(order)} complete: "
                f"Top-1={ngram_row['top_1_accuracy_overall']:.4f}"
            )

    rows.sort(
        key=lambda row: (
            row["representation"],
            row["order"],
        )
    )

    print_comparison(rows)

    summary_dir = args.results_root / "baseline_comparison"
    summary_json = summary_dir / f"{args.split}_left_to_right_summary.json"
    summary_csv = summary_dir / f"{args.split}_left_to_right_summary.csv"

    write_json(
        summary_json,
        {
            "split": args.split,
            "direction": "left_to_right",
            "representations": args.representations,
            "ngram_orders": sorted(set(args.orders)),
            "results": rows,
        },
    )
    save_comparison_csv(summary_csv, rows)

    print("\nBest left-to-right n-gram per representation:")

    for representation in args.representations:
        candidates = [
            row
            for row in rows
            if row["representation"] == representation
            and row["model_type"] == "ngram"
        ]

        best = max(
            candidates,
            key=lambda row: (
                row["top_1_accuracy_overall"],
                row["mrr_overall"],
            ),
        )

        print(
            f"  {representation}: {best['model_name']} "
            f"(Top-1={best['top_1_accuracy_overall']:.4f}, "
            f"MRR={best['mrr_overall']:.4f})"
        )

    print(f"\nSaved comparison JSON: {summary_json}")
    print(f"Saved comparison CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
