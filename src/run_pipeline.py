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

    # 2. Instantiate Transfer Learning BERT Models
    atomic_bert = BertPretrainedFineTuner(
        name="BERT_Transfer_Atomic",
        is_factorized=False,
        output_dir="./results_bert_transfer_atomic",
    )

    factorized_bert = BertPretrainedFineTuner(
        name="BERT_Transfer_Factorized",
        is_factorized=True,
        output_dir="./results_bert_transfer_factorized",
    )

    # 3. Fit Models (Notice lowered batch size to 16 for heavier pre-trained weights)
    print("Fine-tuning Pre-trained BERT on Atomic tokens...")
    atomic_bert.fit(train_atomic, epochs=5, batch_size=16)

    print("Fine-tuning Pre-trained BERT on Factorized tokens...")
    factorized_bert.fit(train_factorized, epochs=5, batch_size=16)

    # 4. Universal Model List for Evaluation
    models_to_evaluate = [ 
        # UnigramModel("Unigram"), 
        # NGramModel("NGram"),
        atomic_bert, 
        factorized_bert, 
    ] 

    # 5. Run Evaluation Loop (Masked Accuracy & Plausibility Analysis)
    for model in models_to_evaluate: 
        print(f"\nEvaluating {model.name}...")

        sample_seq = ( 
            eval_atomic[0]
            if not getattr(model, "is_factorized", False) 
            else eval_factorized[0] 
        )
        target_idx = 2  

        top_k_preds = model.predict_top_k(sample_seq, target_index=target_idx, k=5) 
        print(f"Top 5 predictions for index {target_idx}: {top_k_preds}") 


if __name__ == "__main__": 
    main()