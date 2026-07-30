import json
import torch
from pathlib import Path
from src.evaluator import MusicEvaluator
from src.models.unigram import UnigramModel
from src.models.ngram import NGramModel
from src.models.lstm import MusicLSTMModel

# from src.models.bert import MusicBERTModel  <-- Import BERT here when ready

# Automatically locate the main project root (NLP-final-project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_jsonl(relative_path):
    full_path = PROJECT_ROOT / relative_path
    data = []
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    print("1. Loading datasets for evaluation...")
    train_data = load_jsonl("data/splits/train.jsonl")
    masked_val_data = load_jsonl("data/splits/validation_masked_factorized.jsonl")

    # Extract vocabulary from training data to match what the models expect
    train_sequences = [item["factorized_sequence_tokens"] for item in train_data]
    vocab = list(set([token for seq in train_sequences for token in seq]))

    print("\n2. Initializing Models for Evaluation...")
    models = {
        "Unigram": UnigramModel(),
        "N-Gram (R=2)": NGramModel(radius=2),
        "LSTM": MusicLSTMModel(vocab=vocab, embed_dim=64, hidden_dim=128),
        # "BERT": MusicBERTModel(...)  <-- Add BERT here when ready
    }

    print("\n3. Preparing Models (Building stats / Loading saved weights)...")
    for name, model in models.items():
        if name in ["Unigram", "N-Gram (R=2)"]:
            print(f"   -> Building frequency tables for {name}...")
            model.fit(train_sequences)
        elif name == "LSTM":
            print(f"   -> Loading trained weights for {name}...")
            weight_path = PROJECT_ROOT / "results/lstm_checkpoint.pt"
            if weight_path.exists():
                # Load state dict into the inner model attribute
                model.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
                print("   -> Checkpoint loaded successfully!")
            else:
                print("   -> Warning: No checkpoint found! Running with untrained weights.")

    print("\n4. Running Unified Evaluation across all models...")
    all_results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluator = MusicEvaluator(model, masked_val_data)
        all_results[name] = evaluator.run_evaluation()

    print("\n5. Final Comparative Evaluation Results:")
    print(all_results)

    # Automatically save results to your results folder for your report
    output_path = PROJECT_ROOT / "results/baseline_comparison/validation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved evaluation summary report to: {output_path}")


if __name__ == "__main__":
    main()