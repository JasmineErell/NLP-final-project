import json
from pathlib import Path
from src.models.lstm import MusicLSTMModel
from src.evaluator import MusicEvaluator

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
    print("1. Loading datasets...")
    train_data = load_json("data/splits/train.jsonl")
    masked_val_data = load_json("data/splits/validation_masked_factorized.jsonl")

    # Extract just the sequences for training
    print("Available keys in dataset:", train_data[0].keys())
    train_sequences = [item["factorized_sequence_tokens"] for item in train_data]
    vocab = list(set([token for seq in train_sequences for token in seq]))

    print("\n2. Initializing Models...")
    models = {
        "LSTM": MusicLSTMModel(vocab=vocab, embed_dim=64, hidden_dim=128)
        # TODO: Add "BERT": MusicBERTModel(...) here when ready
    }

    print("\n3. Training Models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_sequences, epochs=5, batch_size=32)



if __name__ == "__main__":
    main()