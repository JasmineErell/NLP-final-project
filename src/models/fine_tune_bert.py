"""BERT model for masked musical-token prediction in ABC notation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM

from .common import BaseMaskedModel


# ---------------------------------------------------------------------------
# Factory function: creates a fresh (random-weights) BertForMaskedLM
# ---------------------------------------------------------------------------

def create_folk_music_bert(
        vocab_size: int,
        pad_token_id: int,
        mask_token_id: int,
        cls_token_id: int,
        sep_token_id: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 258,
        hidden_dropout_prob: float = 0.2,
        attention_probs_dropout_prob: float = 0.2,
) -> BertForMaskedLM:
    """
    Initializes a BERT model configured for Masked Language Modeling (MLM).

    Architecture choices (overfitting mitigation):
    -----------------------------------------------
    - 4 layers / 4 heads instead of 6/8: reduces model capacity to better
      match our tiny dataset (~184K training tokens). A model that is too
      large relative to the data memorises training examples rather than
      learning generalisable musical patterns.
    - max_position_embeddings=258: matches MAX_SEQ_LEN (256) + 2 for
      [CLS]/[SEP]. Using 512 wastes ~50% of the positional embedding
      parameters on positions the model will never see.
    - hidden_dropout_prob=0.2 and attention_probs_dropout_prob=0.2:
      increased from the default 0.1. Higher dropout is a standard
      regularisation technique for small-data regimes — it forces the
      model to be robust to missing activations rather than relying on
      co-adapted features.
    """

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
    )

    model = BertForMaskedLM(config)
    return model


# ---------------------------------------------------------------------------
# Wrapper class: conforms to BaseMaskedModel interface + evaluation pipeline
# ---------------------------------------------------------------------------

class BertPretrainedFineTuner(BaseMaskedModel):
    """
    Wraps a HuggingFace BertForMaskedLM to conform to the project's
    BaseMaskedModel interface used by the evaluation scripts.

    Supports:
    - Loading a trained checkpoint from disk
    - Predicting token probabilities for a single masked position
    - Top-k prediction
    """

    def __init__(
        self,
        name: str = "BERT_MLM",
        is_factorized: bool = True,
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.is_factorized = is_factorized
        self.model: Optional[BertForMaskedLM] = None
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.device = device or self._detect_device()

        if vocab_path:
            self._load_vocab(vocab_path)

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_vocab(self, vocab_path: str) -> None:
        """Load token-to-id mapping from a JSON file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        if isinstance(vocab_data, dict) and "token_to_id" in vocab_data:
            self.token2id = vocab_data["token_to_id"]
        elif isinstance(vocab_data, dict):
            self.token2id = vocab_data
        else:
            self.token2id = {token: idx for idx, token in enumerate(vocab_data)}

        # Ensure special tokens
        for st in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            if st not in self.token2id:
                self.token2id[st] = len(self.token2id)

        self.id2token = {v: k for k, v in self.token2id.items()}

    def load(self, model_dir: str) -> None:
        """Load a pretrained/fine-tuned model and its vocabulary from disk."""
        model_path = Path(model_dir)

        # Load vocabulary if present alongside the model
        # Support both from-scratch (vocab.json) and fine-tuned (musical_vocab.json)
        if not self.token2id:
            vocab_file = model_path / "musical_vocab.json"
            if not vocab_file.exists():
                vocab_file = model_path / "vocab.json"
            if vocab_file.exists():
                self._load_vocab(str(vocab_file))

        # Load the HuggingFace model
        self.model = BertForMaskedLM.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

    def fit(self, train_sequences: List[List[str]], **kwargs) -> Dict[str, Any]:
        """
        Training is handled by train_bert.py. This method exists to satisfy
        the BaseMaskedModel interface. Calling it here raises an error to
        prevent confusion.
        """
        raise NotImplementedError(
            "Use src/train/train_bert.py to train the BERT model. "
            "This class is for inference/evaluation only."
        )

    def predict_token_probabilities(
        self, sequence: List[str], target_index: int
    ) -> Dict[str, float]:
        """
        Given a token sequence with a [MASK] at target_index, return a
        probability distribution over the vocabulary for that position.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load(model_dir) first.")

        pad_id = self.token2id.get("[PAD]", 0)
        unk_id = self.token2id.get("[UNK]", 1)
        cls_id = self.token2id["[CLS]"]
        sep_id = self.token2id["[SEP]"]
        mask_id = self.token2id["[MASK]"]

        # Build input: [CLS] + tokens + [SEP], masking the target position
        tokens = list(sequence)
        tokens[target_index] = "[MASK]"

        input_ids = [cls_id] + [self.token2id.get(t, unk_id) for t in tokens] + [sep_id]
        # Adjust target_index to account for [CLS] prefix
        masked_position = target_index + 1

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
            logits = outputs.logits[0, masked_position, :]
            probs = torch.softmax(logits, dim=-1).cpu()

        # Convert to dict
        prob_dict: Dict[str, float] = {}
        for token_id, prob in enumerate(probs.tolist()):
            if prob > 1e-6:  # Skip negligible probabilities
                token = self.id2token.get(token_id, f"[ID_{token_id}]")
                prob_dict[token] = prob

        return prob_dict

    def predict_top_k(
        self, sequence: List[str], target_index: int, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict top-k tokens for a masked position."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load(model_dir) first.")

        pad_id = self.token2id.get("[PAD]", 0)
        unk_id = self.token2id.get("[UNK]", 1)
        cls_id = self.token2id["[CLS]"]
        sep_id = self.token2id["[SEP]"]

        # Build input with mask
        tokens = list(sequence)
        tokens[target_index] = "[MASK]"

        input_ids = [cls_id] + [self.token2id.get(t, unk_id) for t in tokens] + [sep_id]
        masked_position = target_index + 1

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)
            logits = outputs.logits[0, masked_position, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=k)

        results = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            token = self.id2token.get(idx, f"[ID_{idx}]")
            results.append((token, prob))

        return results
