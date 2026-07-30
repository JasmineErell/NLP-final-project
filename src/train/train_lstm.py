"""Train unidirectional LSTM models on atomic and/or factorized sequences."""

import argparse
import json
import random
from pathlib import Path

import torch

from src.models.lstm import MusicLSTMModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

REPRESENTATION_SETTINGS = {
    "atomic": {
        "sequence_key": "atomic_sequence_tokens",
        "target_prefix": "CHORD_",
    },
    "factorized": {
        "sequence_key": "factorized_sequence_tokens",
        "target_prefix": "[EVENT_",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM music language models.")
    parser.add_argument(
        "--representation",
        choices=["atomic", "factorized", "both"],
        default="factorized",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Optional small subset for a quick debugging run.",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_representation(
    version: str,
    train_data: list[dict],
    args: argparse.Namespace,
) -> None:
    settings = REPRESENTATION_SETTINGS[version]
    sequence_key = settings["sequence_key"]
    target_prefix = settings["target_prefix"]

    train_sequences = [item[sequence_key] for item in train_data]

    # Sorting makes token IDs reproducible across training and evaluation.
    vocab = sorted({token for sequence in train_sequences for token in sequence})

    print(f"\n{'=' * 60}")
    print(f"TRAINING LSTM: {version.upper()}")
    print(f"{'=' * 60}")
    print(f"Songs: {len(train_sequences)}")
    print(f"Corpus vocabulary: {len(vocab)}")
    print(f"Valid masked targets: {target_prefix}*")

    lstm_model = MusicLSTMModel(
        vocab=vocab,
        target_prefix=target_prefix,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        context_length=args.seq_len,
        device=args.device,
        name=f"lstm_{version}",
    )

    print(f"Device: {lstm_model.device}")
    print(f"Model vocabulary with special tokens: {len(lstm_model.vocab)}")
    print("\nTraining model...")

    stats = lstm_model.fit(
        train_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    output_dir = PROJECT_ROOT / "results" / "lstm" / version
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "training_metrics.json"

    lstm_model.save(
        checkpoint_path,
        extra={
            "representation": version,
            "training_stats": stats,
            "seed": args.seed,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "representation": version,
                "target_prefix": target_prefix,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "learning_rate": args.lr,
                **stats,
            },
            file,
            indent=2,
        )

    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"Training metrics saved to: {metrics_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading training dataset...")
    train_data = load_jsonl("data/splits/train.jsonl")

    if args.max_songs is not None:
        train_data = train_data[: args.max_songs]
        print(f"Debug subset enabled: using {len(train_data)} songs.")

    versions = (
        ["atomic", "factorized"]
        if args.representation == "both"
        else [args.representation]
    )

    for version in versions:
        train_representation(version, train_data, args)


if __name__ == "__main__":
    main()