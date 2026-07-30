import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

from .common import BaseMaskedModel


class NGramModel(BaseMaskedModel):
    """
    N-gram Language Model supporting both:
    1. Unidirectional context (Order N)
    2. Bidirectional context (Radius R around target_index)
    """

    def __init__(
        self, 
        order: Optional[int] = 3, 
        radius: Optional[int] = None, 
        smoothing: float = 1e-5,
        name: str = "ngram"
    ):
        super().__init__(name=name)
        self.order = order
        self.radius = radius
        self.smoothing = smoothing
        
        # Frequency counts
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def _extract_context(self, sequence: List[str], target_index: int) -> Tuple[str, ...]:
        """Extracts context tuple based on radius (bidirectional) or order (unidirectional)."""
        seq_len = len(sequence)
        
        if self.radius is not None:
            # Bidirectional context window around target_index
            left_start = max(0, target_index - self.radius)
            right_end = min(seq_len, target_index + self.radius + 1)
            
            left_ctx = sequence[left_start:target_index]
            right_ctx = sequence[target_index + 1:right_end]
            
            # Use special placeholder for target position in context key
            return tuple(left_ctx + ["<TARGET>"] + right_ctx)
        
        elif self.order is not None:
            # Left-to-right unidirectional context of length (order - 1)
            left_start = max(0, target_index - (self.order - 1))
            return tuple(sequence[left_start:target_index])
        
        else:
            return ()

    def fit(self, train_sequences: List[List[str]], **kwargs) -> Dict[str, Any]:
        """Extracts N-gram counts from training sequences."""
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocab.clear()

        for seq in train_sequences:
            self.vocab.update(seq)
            seq_len = len(seq)
            
            for i in range(seq_len):
                target_token = seq[i]
                context = self._extract_context(seq, i)
                
                self.ngram_counts[context][target_token] += 1
                self.context_counts[context] += 1

        return {
            "vocab_size": len(self.vocab),
            "num_contexts": len(self.context_counts)
        }

    def predict_token_probabilities(
        self, sequence: List[str], target_index: int
    ) -> Dict[str, float]:
        """Predicts probability distribution for a target position using N-gram context."""
        context = self._extract_context(sequence, target_index)
        counts = self.ngram_counts.get(context, Counter())
        total_context_count = self.context_counts.get(context, 0)

        vocab_list = list(self.vocab) if self.vocab else list(set(sequence))
        vocab_size = len(vocab_list)

        probs = {}
        denom = total_context_count + (self.smoothing * vocab_size)

        for token in vocab_list:
            count = counts.get(token, 0)
            if denom > 0:
                probs[token] = (count + self.smoothing) / denom
            else:
                probs[token] = 1.0 / max(1, vocab_size)

        return probs