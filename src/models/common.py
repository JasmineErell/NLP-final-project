from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseMaskedModel(ABC):
    """Abstract base class for all masked music models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, train_sequences: List[List[str]], **kwargs) -> Dict[str, Any]:
        """Fit model on training sequences."""
        pass

    @abstractmethod
    def predict_token_probabilities(
        self, sequence: List[str], target_index: int
    ) -> Dict[str, float]:
        """Predict probability distribution over vocabulary for a target position."""
        pass

    def predict_top_k(
        self, sequence: List[str], target_index: int, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict top-k tokens and their probabilities for a target position."""
        probs = self.predict_token_probabilities(sequence, target_index)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]