"""Shared utilities for symbolic-music models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


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


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_path}"
                ) from error


def write_json(path: str | Path, data: Any) -> None:
    """Write formatted JSON and create parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_representation(representation: str) -> None:
    """Raise a clear error for an unsupported representation."""
    if representation not in REPRESENTATIONS:
        allowed = ", ".join(sorted(REPRESENTATIONS))
        raise ValueError(
            f"Unsupported representation: {representation!r}. "
            f"Choose one of: {allowed}"
        )


def ranked_tokens_from_counts(
    counts: dict[str, int],
) -> list[str]:
    """
    Sort by descending frequency and then lexicographically.

    Lexicographic tie-breaking makes results deterministic.
    """
    return [
        token
        for token, _ in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def calculate_ranking_metrics(
    target_token: str,
    ranked_predictions: list[str],
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float | int | None]:
    """Calculate hits and reciprocal rank for one prediction."""
    metrics: dict[str, float | int | None] = {}

    for k in top_ks:
        metrics[f"top_{k}_hit"] = int(
            target_token in ranked_predictions[:k]
        )

    try:
        rank = ranked_predictions.index(target_token) + 1
        metrics["rank"] = rank
        metrics["reciprocal_rank"] = 1.0 / rank
    except ValueError:
        metrics["rank"] = None
        metrics["reciprocal_rank"] = 0.0

    return metrics
