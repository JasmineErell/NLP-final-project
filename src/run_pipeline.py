"""Run the complete preprocessing pipeline from the project root."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ABC preprocessing, splitting and fixed masking."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ABC_cleaned"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use a small number such as 20 for a pilot run.",
    )
    return parser.parse_args()

def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)

def main() -> None:
    args = parse_args()
    script_directory = Path(__file__).resolve().parent

    preprocess_command = [
        sys.executable,
        str(script_directory / "preprocess_abc.py"),
        "--input",
        str(args.input),
    ]
    if args.limit is not None:
        preprocess_command.extend(["--limit", str(args.limit)])

    run(preprocess_command)
    run([sys.executable, str(script_directory / "create_splits.py")])
    run([sys.executable, str(script_directory / "create_masked_examples.py")])

    print("\nData Pipeline finished successfully.")
    print("Inspect data/processed/tokenization_examples.txt first.")
    print("You can now run the training script: python -m src.train_bert")

if __name__ == "__main__":
    main()