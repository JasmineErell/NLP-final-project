"""Standalone evaluation script to test the N-Gram model on both representations."""

import json
from pathlib import Path
from src.models.ngram import NGramModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_jsonl(relative_path: str) -> list[dict]:
    full_path = PROJECT_ROOT / relative_path
    data = []
    if not full_path.exists():
        print(f"Warning: File not found -> {full_path}")
        return data
    
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main() -> None:
    print("Loading training dataset...")
    train_data = load_jsonl("data/splits/train.jsonl")

    if not train_data:
        print("Error: Could not load training data.")
        return

    representations = ["atomic", "factorized"]

    for version in representations:
        print(f"\n{'='*50}")
        print(f"EVALUATING REPRESENTATION: {version.upper()}")
        print(f"{'='*50}")

        # 1. Load specific version data
        masked_val_data = load_jsonl(f"data/splits/validation_masked_{version}.jsonl")
        sequence_key = f"{version}_sequence_tokens"
        train_sequences = [item[sequence_key] for item in train_data]

        # 2. Fit N-Gram Model
        # Define the strict target prefix based on your rules
        if version == "atomic":
            target_prefix = "CHORD_"
        else:
            target_prefix = "[EVENT_"

        # 2. Fit N-Gram Model with the prefix filter
        print(f"Fitting N-Gram Model (Radius 2) on {version} data (Prefix: {target_prefix})...")
        ngram_model = NGramModel(radius=2, name=f"ngram_{version}")
        
        # Pass the target_prefix into the fit method!
        stats = ngram_model.fit(train_sequences, target_prefix=target_prefix)
        print(f"   -> Model Stats: {stats}")

        # 3. Evaluate first 3 samples
        print("\nEvaluating First 3 Samples:")
        for i, sample in enumerate(masked_val_data[:3]):
            seq = sample.get("input_tokens", [])
            target_idx = sample.get("position", 0)
            true_token = sample.get("target_token", "UNKNOWN")

            if not isinstance(target_idx, int) and "[MASK]" in seq:
                target_idx = seq.index("[MASK]")

            print(f"\n  Sample {i + 1}")
            left_ctx = seq[max(0, target_idx - 2):target_idx]
            right_ctx = seq[target_idx + 1:target_idx + 3]
            print(f"  Context: {left_ctx} [MASK] {right_ctx}")
            print(f"  Target:  {true_token}")
            
            top_k = ngram_model.predict_top_k(seq, target_index=target_idx, k=3)
            
            print("  Top 3 Predictions:")
            for rank, (token, prob) in enumerate(top_k, 1):
                print(f"    {rank}. {token:<15} ({prob:.4f})")

if __name__ == "__main__":
    main()