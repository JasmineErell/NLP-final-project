"""Run the complete preprocessing pipeline from the project root."""

from __future__ import annotations
import os
from src.models.unigram import UnigramModel
from src.models.ngram import NGramModel
from src.models.lstm import LSTMModel
from src.models.bert import BertMaskedModel
from src.create_splits import load_data_splits

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
    run(
        [
            sys.executable,
            str(script_directory / "create_splits.py"),
        ]
    )
    run(
        [
            sys.executable,
            str(script_directory / "create_masked_examples.py"),
        ]
    )

    print("\nPipeline finished successfully.")
    print("Inspect data/processed/tokenization_examples.txt first.")

    print("Loading data splits...")
    train_atomic, eval_atomic = load_data_splits(version="atomic")
    train_factorized, eval_factorized = load_data_splits(version="factorized")

    # Paths to the tokenizers generated during preprocessing
    atomic_tokenizer_path = "data/tokenizers/atomic_tokenizer.json"
    factorized_tokenizer_path = "data/tokenizers/factorized_tokenizer.json"

    # 2. Instantiate BERT Models
    atomic_bert = BertMaskedModel(
        name="BERT_Atomic",
        tokenizer_file_path=atomic_tokenizer_path,
        is_factorized=False,
        output_dir="./results_bert_atomic",
    )

    factorized_bert = BertMaskedModel(
        name="BERT_Factorized",
        tokenizer_file_path=factorized_tokenizer_path,
        is_factorized=True,
        output_dir="./results_bert_factorized",
    )

    # 3. Fit BERT Models
    print("Fitting Atomic BERT...")
    atomic_bert.fit(train_atomic, epochs=5, batch_size=32)

    print("Fitting Factorized BERT...")
    factorized_bert.fit(train_factorized, epochs=5, batch_size=32)

    # 4. Universal Model List for Evaluation
    # Because all models inherit from BaseMaskedModel, you can iterate through them together
    models_to_evaluate = [
        # UnigramModel("Unigram"),
        # NGramModel("NGram"),
        atomic_bert,
        factorized_bert,
    ]

    # 5. Run Evaluation Loop (Masked Accuracy & Plausibility Analysis)
    for model in models_to_evaluate:
        print(f"\nEvaluating {model.name}...")

        # Example: Test prediction on a sample sequence and target index
        sample_seq = (
            eval_atomic[0]
            if not getattr(model, "is_factorized", False)
            else eval_factorized[0]
        )
        target_idx = 2  # Index of note/chord to mask

        top_k_preds = model.predict_top_k(sample_seq, target_index=target_idx, k=5)
        print(f"Top 5 predictions for index {target_idx}: {top_k_preds}")


if __name__ == "__main__":
    main()