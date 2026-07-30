import argparse
import json
from pathlib import Path
from transformers import BertTokenizer

from src.models.bert import MusicBERT

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_jsonl(file_path: Path) -> list:
    data = []
    if not file_path.exists():
        raise FileNotFoundError(f"Missing data file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description="Train Music BERT")
    parser.add_argument("--representation", type=str, choices=["factorized", "atomic"], default="factorized")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", action="store_true", help="Run a tiny pilot test")
    args = parser.parse_args()

    print(f"=== Training BERT ({args.representation.upper()}) ===")
    
    # 1. Load Data
    train_data = load_jsonl(PROJECT_ROOT / "data/splits/train.jsonl")
    val_data = load_jsonl(PROJECT_ROOT / f"data/splits/validation_masked_{args.representation}.jsonl")
    
    seq_key = f"{args.representation}_sequence_tokens"
    train_sequences = [item[seq_key] for item in train_data]

    if args.limit:
        train_sequences = train_sequences[:64]
        val_data = val_data[:128]
        args.epochs = 1
        print("LIMIT FLAG ACTIVE: Running pilot test...")

    # 2. Build Vocabulary and Tokenizer
    print("Building vocabulary...")
    vocab = set()
    for seq in train_sequences:
        vocab.update(seq)
    
    # Add special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab_list = special_tokens + sorted(list(vocab))
    
    # Save vocab temporarily for the tokenizer
    vocab_file = PROJECT_ROOT / f"data/bert_vocab_{args.representation}.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token in vocab_list:
            f.write(token + "\n")

    tokenizer = BertTokenizer(
        vocab_file=str(vocab_file),
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]"
    )

    # 3. Setup Model
    target_prefix = "CHORD_" if args.representation == "atomic" else "[EVENT_"
    output_dir = str(PROJECT_ROOT / f"results/bert/{args.representation}")
    
    bert_model = MusicBERT(tokenizer=tokenizer, output_dir=output_dir)

    # 4. Train
    results = bert_model.fit(
        train_sequences=train_sequences,
        validation_examples=val_data,
        target_prefix=target_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mask_probability=args.mask_probability,
        seed=args.seed
    )

    print(f"\nTraining Complete! Best Validation Loss: {results.get('best_eval_loss')}")

if __name__ == "__main__":
    main()