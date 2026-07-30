"""Train the BERT models using cross-domain transfer learning."""

import argparse
from pathlib import Path
from src.models.bert import BertPretrainedFineTuner
from src.create_splits import load_jsonl

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on ABC data.")
    parser.add_argument(
        "--limit",
        action="store_true",
        help="Run a quick pilot test with reduced epochs and batch size.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()

def load_data_splits(version: str) -> tuple[list[list[str]], list[list[str]]]:
    train_records = load_jsonl(Path("data/splits/train.jsonl"))
    eval_records = load_jsonl(Path("data/splits/validation.jsonl"))

    sequence_field = f"{version}_sequence_tokens"

    train_sequences = [record[sequence_field] for record in train_records]
    eval_sequences = [record[sequence_field] for record in eval_records]

    return train_sequences, eval_sequences

def main() -> None:
    args = parse_args()

    print("Loading data splits...")
    train_atomic, eval_atomic = load_data_splits(version="atomic")
    train_factorized, eval_factorized = load_data_splits(version="factorized")

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

    epochs = 1 if args.limit else args.epochs
    batch_size = 4 if args.limit else args.batch_size

    print(f"\nFine-tuning Pre-trained BERT on Atomic tokens (Epochs: {epochs}, Batch: {batch_size})...")
    atomic_bert.fit(train_atomic, epochs=epochs, batch_size=batch_size)

    print(f"\nFine-tuning Pre-trained BERT on Factorized tokens (Epochs: {epochs}, Batch: {batch_size})...")
    factorized_bert.fit(train_factorized, epochs=epochs, batch_size=batch_size)

    print("\nTraining complete! Run 'python -m src.evaluate_bert' to test the models.")

if __name__ == "__main__":
    main()