import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional

from .common import BaseMaskedModel


class NGramModel(BaseMaskedModel):
    """
    Production-grade N-gram Language Model supporting both:
    1. Unidirectional context with Backoff (Order N)
    2. Bidirectional context with Backoff & Padding (Radius R around target_index)
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
        
        # Frequency counts mapped by context tuple -> Counter(target_token -> frequency)
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def _extract_context(
        self, 
        sequence: List[str], 
        target_index: int, 
        current_radius: Optional[int] = None, 
        current_order: Optional[int] = None
    ) -> Tuple[str, ...]:
        """
        Extracts context tuple with explicit <BOS> / <EOS> boundary padding 
        to guarantee fixed-length context windows.
        """
        seq_len = len(sequence)
        r = current_radius if current_radius is not None else self.radius
        o = current_order if current_order is not None else self.order

        if r is not None:
            # --- Bidirectional Context (Radius R) ---
            # Extract available left tokens and pad with <BOS> if needed
            left_tokens = sequence[max(0, target_index - r): target_index]
            pad_left_count = r - len(left_tokens)
            left_ctx = ["<BOS>"] * pad_left_count + left_tokens

            # Extract available right tokens and pad with <EOS> if needed
            right_tokens = sequence[target_index + 1: min(seq_len, target_index + r + 1)]
            pad_right_count = r - len(right_tokens)
            right_ctx = right_tokens + ["<EOS>"] * pad_right_count

            return tuple(left_ctx + ["<TARGET>"] + right_ctx)

        elif o is not None:
            # --- Unidirectional Context (Order N) ---
            context_len = max(0, o - 1)
            if context_len == 0:
                return ()
            left_tokens = sequence[max(0, target_index - context_len): target_index]
            pad_count = context_len - len(left_tokens)
            return tuple(["<BOS>"] * pad_count + left_tokens)

        return ()

    def fit(self, train_sequences: List[List[str]], **kwargs) -> Dict[str, Any]:
        """
        Extracts N-gram counts from training sequences for ALL context levels
        (down to radius 0 / unigram) to enable fast backoff during prediction.
        """
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocab.clear()

        for seq in train_sequences:
            self.vocab.update(seq)
            seq_len = len(seq)
            
            for i in range(seq_len):
                target_token = seq[i]
                
                if self.radius is not None:
                    # Index counts for radii from self.radius down to 0 (Unigram)
                    for r in range(self.radius, -1, -1):
                        ctx = self._extract_context(seq, i, current_radius=r)
                        self.ngram_counts[ctx][target_token] += 1
                        self.context_counts[ctx] += 1
                elif self.order is not None:
                    # Index counts for orders from self.order down to 1
                    for o in range(self.order, 0, -1):
                        ctx = self._extract_context(seq, i, current_order=o)
                        self.ngram_counts[ctx][target_token] += 1
                        self.context_counts[ctx] += 1

        return {
            "vocab_size": len(self.vocab),
            "num_contexts": len(self.context_counts)
        }

    def predict_token_probabilities(
        self, sequence: List[str], target_index: int
    ) -> Dict[str, float]:
        """
        Predicts probability distribution using Stupid Backoff. 
        Falls back to smaller context windows if higher-order context is unseen.
        """
        vocab_list = sorted(list(self.vocab)) if self.vocab else sorted(list(set(sequence)))
        vocab_size = len(vocab_list)

        if vocab_size == 0:
            return {}

        counts = Counter()
        total_context_count = 0

        # Backoff Search: Find highest-order context present in training history
        if self.radius is not None:
            for r in range(self.radius, -1, -1):
                ctx = self._extract_context(sequence, target_index, current_radius=r)
                if ctx in self.context_counts and self.context_counts[ctx] > 0:
                    counts = self.ngram_counts[ctx]
                    total_context_count = self.context_counts[ctx]
                    break
        elif self.order is not None:
            for o in range(self.order, 0, -1):
                ctx = self._extract_context(sequence, target_index, current_order=o)
                if ctx in self.context_counts and self.context_counts[ctx] > 0:
                    counts = self.ngram_counts[ctx]
                    total_context_count = self.context_counts[ctx]
                    break

        # Calculate Laplace-smoothed probabilities for the matched context level
        denom = total_context_count + (self.smoothing * vocab_size)
        probs = {}

        for token in vocab_list:
            count = counts.get(token, 0)
            if denom > 0:
                probs[token] = (count + self.smoothing) / denom
            else:
                probs[token] = 1.0 / vocab_size

        return probs