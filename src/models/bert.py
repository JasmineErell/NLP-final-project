import torch
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from typing import List, Dict, Any, Optional
import os

from .common import BaseMaskedModel

class TargetOnlyMaskingCollator:
    """Custom collator that ONLY masks target tokens (e.g., [EVENT_*] or CHORD_*)."""
    def __init__(self, tokenizer, target_token_ids: List[int], mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.target_token_ids = set(target_token_ids)
        self.target_token_ids_tensor = torch.tensor(target_token_ids, dtype=torch.long)
        self.mlm_probability = mlm_probability

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # If labels are already provided (Validation set), just dynamically pad
        if "labels" in features[0]:
            return self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # Dynamically pad the training batch
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Find eligible positions for masking
        eligible = torch.zeros_like(input_ids, dtype=torch.bool)
        for t_id in self.target_token_ids:
            eligible |= input_ids.eq(t_id)

        # Select 15% of ELIGIBLE tokens
        probabilities = torch.full(labels.shape, self.mlm_probability)
        masked = torch.bernoulli(probabilities).bool() & eligible

        # Set non-masked tokens to -100 so loss is ignored
        labels[~masked] = -100

        # 80% MASK, 10% Random Target Token, 10% Unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        batch["input_ids"][indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~indices_replaced
        random_targets = self.target_token_ids_tensor[torch.randint(0, len(self.target_token_ids), labels.shape)]
        batch["input_ids"][indices_random] = random_targets[indices_random]

        batch["labels"] = labels
        return batch


class MusicBERT(BaseMaskedModel):
    def __init__(self, tokenizer, name: str = "bert_model", output_dir: str = "results/bert"):
        super().__init__(name=name)
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.model = None

    def fit(
        self,
        train_sequences: List[List[str]],
        validation_examples: List[Dict[str, Any]],
        target_prefix: str,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        mask_probability: float = 0.15,
        seed: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """Trains BERT using target-only masking and fixed validation."""
        
        # 1. Initialize fresh BERT Model
        config = BertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=256,     # Scale down for faster Colab training
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512
        )
        self.model = BertForMaskedLM(config)

        # 2. Identify target token IDs for the collator
        target_token_ids = []
        for token, token_id in self.tokenizer.get_vocab().items():
            if "EVENT_" in token:  # More robust substring check
                target_token_ids.append(token_id)
                
        print(f"Found {len(target_token_ids)} valid target tokens containing 'EVENT_'")
        
        # 3. Prepare Training Dataset (Unmasked - Collator handles masking)
        train_features = []
        truncated_count = 0
        for seq in train_sequences:
            if len(seq) > 510:
                truncated_count += 1
            enc = self.tokenizer(seq, is_split_into_words=True, truncation=True, max_length=512)
            train_features.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        
        if truncated_count > 0:
            print(f"Warning: {truncated_count} training sequences were truncated to 512 tokens.")
            
        train_dataset = Dataset.from_list(train_features)

        # 4. Prepare Fixed Validation Dataset (Pre-masked)
        eval_features = []
        for ex in validation_examples:
            seq = ex.get("input_tokens", [])
            target = ex.get("target_token", "")
            
            # Find the [MASK] position
            try:
                mask_idx = seq.index("[MASK]") + 1 # +1 for [CLS]
            except ValueError:
                continue
                
            if mask_idx >= 512:
                continue # Skip if mask is truncated
                
            enc = self.tokenizer(seq, is_split_into_words=True, truncation=True, max_length=512)
            labels = [-100] * len(enc["input_ids"])
            
            target_id = self.tokenizer.convert_tokens_to_ids(target)
            if target_id != self.tokenizer.unk_token_id:
                labels[mask_idx] = target_id
                eval_features.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": labels
                })
                
        eval_dataset = Dataset.from_list(eval_features)
        print(f"Loaded {len(train_dataset)} train examples and {len(eval_dataset)} validation examples.")

        # 5. Setup Collator and Trainer
        collator = TargetOnlyMaskingCollator(
            tokenizer=self.tokenizer,
            target_token_ids=target_token_ids,
            mlm_probability=mask_probability
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            data_seed=seed,
            eval_strategy="epoch",        # Updated for transformers v5+ (formerly evaluation_strategy)
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=10,
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )

        # 6. Train!
        print("Starting BERT training...")
        trainer.train()

        # 7. Save best model and tokenizer
        final_dir = os.path.join(self.output_dir, "best_model")
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        return {"status": "success", "best_eval_loss": trainer.state.best_metric}

    def predict_token_probabilities(self, sequence: List[str], target_index: int) -> Dict[str, float]:
            """
            Predicts the probability distribution for the masked token during the final evaluation phase.
            """
            if self.model is None:
                raise RuntimeError("Model is not loaded. Train or load a model first.")

            # Ensure model is in evaluation mode (turns off dropout)
            self.model.eval()
            
            # Ensure the sequence actually contains the [MASK] token at the target index
            seq_copy = sequence.copy()
            if seq_copy[target_index] != "[MASK]":
                seq_copy[target_index] = "[MASK]"

            # Tokenize the sequence
            encoding = self.tokenizer(
                seq_copy, 
                is_split_into_words=True, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Move inputs to the same device as the model (GPU/CPU)
            encoding = {k: v.to(self.model.device) for k, v in encoding.items()}

            # Find the exact index of the [MASK] token in the resulting tensor
            # (It will likely be target_index + 1 because BERT adds a [CLS] token at the start)
            input_ids = encoding["input_ids"][0]
            mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_positions) == 0:
                return {}  # The mask was likely cut off by truncation
                
            mask_idx = mask_positions[0].item()

            # Run the model without calculating gradients to save memory
            with torch.no_grad():
                outputs = self.model(**encoding)
                
            # Extract the raw logits for the specific masked position
            logits = outputs.logits[0, mask_idx, :]
            
            # Convert logits to a standard probability distribution (0.0 to 1.0)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Map the probabilities back to the human-readable string tokens
            vocab = self.tokenizer.get_vocab()
            token_probs = {}
            
            for token, token_id in vocab.items():
                # Skip special BERT tokens so they don't skew the evaluation metrics
                if not token.startswith(("[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]")):
                    token_probs[token] = probs[token_id].item()

            return token_probs