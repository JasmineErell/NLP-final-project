import math
from collections import Counter
from typing import List, Dict, Any, Optional

from .common import BaseMaskedModel


class UnigramModel(BaseMaskedModel):
    """
    Unigram Baseline Model.
    
    Predicts token probabilities based purely on relative unigram frequencies 
    in the training set, ignoring any surrounding context.
    """

    def __init__(self, smoothing: float = 1e-5, name: str = "unigram"):
        super().__init__(name=name)
        self.smoothing = smoothing
        self.token_counts: Counter = Counter()
        self.total_tokens: int = 0
        self.vocab: List[str] = []
        self.probabilities: Dict[str, float] = {}

    def fit(self, train_sequences: List[List[str]], **kwargs) -> Dict[str, Any]:
        """Fits unigram frequencies on training sequences."""
        self.token_counts = Counter()
        for seq in train_sequences:
            self.token_counts.update(seq)

        self.total_tokens = sum(self.token_counts.values())
        self.vocab = list(self.token2id.keys()) if hasattr(self, "token2id") else list(self.token_counts.keys())

        # Precompute probabilities with Add-k / Laplace smoothing
        vocab_size = len(self.vocab)
        denom = self.total_tokens + (self.smoothing * vocab_size)

        self.probabilities = {}
        for token in self.vocab:
            count = self.token_counts.get(token, 0)
            self.probabilities[token] = (count + self.smoothing) / denom if denom > 0 else 1.0 / max(1, vocab_size)

        return {
            "total_tokens": self.total_tokens,
            "vocab_size": vocab_size
        }

    def predict_token_probabilities(
        self, sequence: List[str], target_index: int
    ) -> Dict[str, float]:
        """
        Returns the token probability distribution for the target position.
        
        Since Unigram is context-independent, it returns the global token probabilities.
        """
        if not self.probabilities:
            # Fallback if fit hasn't been called or probabilities empty
            fallback_prob = 1.0 / max(1, len(sequence))
            return {token: fallback_prob for token in sequence}

        return self.probabilities.copy()