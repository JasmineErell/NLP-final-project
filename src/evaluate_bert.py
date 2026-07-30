"""Evaluate trained BERT models on the validation or test splits."""

import argparse
from pathlib import Path
from src.models.bert import BertPretrainedFineTuner
from src.create_splits import load_jsonl

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BERT models.")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Which data split to evaluate on.",
    )
    return parser.parse_args()

def load_data_splits(version: str, split: str) -> list[list[str]]:
    """
    Loads the processed JSONL splits and extracts the token sequences 
    for the specified representation version.
    """
    records = load_jsonl(Path(f"data/splits/{split}.jsonl"))
    sequence_field = f"{version}_sequence_tokens"
    return [record[sequence_field] for record in records]

def main() -> None:
    args = parse_args()

    print(f"Loading {args.split} splits...")
    eval_atomic = load_data_splits(version="atomic", split=args.split)
    eval_factorized = load_data_splits(version="factorized", split=args.split)

    # 1. Instantiate Models
    atomic_bert = BertPretrainedFineTuner(
        name="BERT_Transfer_Atomic",
        is_factorized=False,
    )

    factorized_bert = BertPretrainedFineTuner(
        name="BERT_Transfer_Factorized",
        is_factorized=True,
    )

    # 2. Load Saved Weights from Disk
    try:
        atomic_bert.load("./results_bert_transfer_atomic/final_model")
        factorized_bert.load("./results_bert_transfer_factorized/final_model")
    except Exception as e:
        print(f"Error loading models. Have you trained them yet?\nDetails: {e}")
        return

    # 3. Universal Evaluation Loop
    models_to_evaluate = [atomic_bert, factorized_bert]

    for model in models_to_evaluate:
        print(f"\nEvaluating {model.name} on {args.split} set...")

        sample_seq = (
            eval_atomic[0]
            if not getattr(model, "is_factorized", False)
            else eval_factorized[0]
        )
        target_idx = 2 

        try:
            top_k_preds = model.predict_top_k(sample_seq, target_index=target_idx, k=5)
            print(f"Top 5 predictions for index {target_idx}: {top_k_preds}")
        except Exception as e:
            print(f"Evaluation encountered an error: {e}")

if __name__ == "__main__":
    main()