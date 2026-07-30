import json
import torch
from pathlib import Path
from src.models.lstm import MusicLSTMModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_json(relative_path):
    full_path = PROJECT_ROOT / relative_path
    data = []
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def main():
    print("1. Loading training dataset...")
    train_data = load_json("data/splits/train.jsonl")

    train_sequences = [item["factorized_sequence_tokens"] for item in train_data]
    vocab = list(set([token for seq in train_sequences for token in seq]))

    print("\n2. Initializing LSTM Model...")
    lstm_model = MusicLSTMModel(vocab=vocab, embed_dim=64, hidden_dim=128)

    print("\n3. Training Model...")
    lstm_model.fit(train_sequences, epochs=5, batch_size=32)

    print("\n4. Saving trained LSTM weights...")
    weight_dir = PROJECT_ROOT / "results"
    weight_dir.mkdir(parents=True, exist_ok=True)

    torch.save(lstm_model.model.state_dict(), weight_dir / "lstm_checkpoint.pt")
    print(f"Checkpoint saved successfully to {weight_dir / 'lstm_checkpoint.pt'}!")


if __name__ == "__main__":
    main()