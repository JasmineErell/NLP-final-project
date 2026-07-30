"""Unidirectional LSTM language model for symbolic-music token prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .common import BaseMaskedModel


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>"]


def choose_device(requested_device: Optional[str] = None) -> str:
    """Choose CUDA, Apple Silicon MPS, or CPU."""
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SequenceDataset(Dataset):
    """Create next-token training chunks from token-ID sequences."""

    def __init__(
        self,
        sequences: List[List[int]],
        seq_len: int = 64,
        pad_id: int = 0,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be at least 1.")

        self.samples = []
        self.seq_len = seq_len
        self.pad_id = pad_id

        for sequence in sequences:
            if len(sequence) < 2:
                continue

            for start in range(0, len(sequence) - 1, seq_len):
                chunk_x = sequence[start : start + seq_len]
                chunk_y = sequence[start + 1 : start + 1 + seq_len]

                chunk_x += [pad_id] * (seq_len - len(chunk_x))
                chunk_y += [pad_id] * (seq_len - len(chunk_y))

                self.samples.append(
                    (
                        torch.tensor(chunk_x, dtype=torch.long),
                        torch.tensor(chunk_y, dtype=torch.long),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class PyTorchLSTMModule(nn.Module):
    """Embedding -> unidirectional LSTM -> vocabulary projection."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embeddings = self.embedding(x)
        output, hidden = self.lstm(embeddings, hidden)
        logits = self.fc(output)
        return logits, hidden


class MusicLSTMModel(BaseMaskedModel):
    """
    Unidirectional LSTM language model.

    Training uses next-token prediction. During masked evaluation, the model
    predicts the masked token using only tokens to the left of target_index.
    This makes it directly comparable with a left-to-right N-gram model.
    """

    def __init__(
        self,
        vocab: List[str],
        target_prefix: Optional[str] = None,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        context_length: int = 64,
        device: Optional[str] = None,
        name: str = "lstm",
    ):
        super().__init__(name=name)

        # Use a deterministic vocabulary and reserve stable special-token IDs.
        unique_vocab = list(dict.fromkeys(vocab))
        content_tokens = [token for token in unique_vocab if token not in SPECIAL_TOKENS]
        self.vocab = SPECIAL_TOKENS + content_tokens

        self.token2id = {token: index for index, token in enumerate(self.vocab)}
        self.id2token = {index: token for token, index in self.token2id.items()}

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.pad_id = self.token2id[self.pad_token]
        self.unk_id = self.token2id[self.unk_token]
        self.bos_id = self.token2id[self.bos_token]

        self.target_prefix = target_prefix
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.context_length = context_length
        self.device = choose_device(device)

        self.model = PyTorchLSTMModule(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=self.pad_id,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def fit(
        self,
        train_sequences: List[List[str]],
        epochs: int = 10,
        batch_size: int = 32,
        seq_len: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train on the training split using left-to-right next-token loss."""
        effective_seq_len = seq_len or self.context_length
        self.context_length = effective_seq_len

        numeric_sequences = []
        for sequence in train_sequences:
            token_ids = [self.token2id.get(token, self.unk_id) for token in sequence]
            numeric_sequences.append([self.bos_id] + token_ids)

        dataset = SequenceDataset(
            numeric_sequences,
            seq_len=effective_seq_len,
            pad_id=self.pad_id,
        )
        if len(dataset) == 0:
            raise ValueError("No LSTM training samples were created.")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        history = []
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(x_batch)

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(dataloader)
            history.append(average_loss)
            print(
                f"   [LSTM] Epoch [{epoch + 1}/{epochs}] "
                f"- Loss: {average_loss:.4f}"
            )

        return {
            "train_loss": history,
            "vocab_size": len(self.vocab),
            "training_chunks": len(dataset),
            "device": self.device,
            "context_length": self.context_length,
        }

    def _candidate_token_ids(self) -> List[int]:
        """Return IDs allowed as masked targets for this representation."""
        candidate_ids = []

        for token, token_id in self.token2id.items():
            if token in SPECIAL_TOKENS:
                continue
            if self.target_prefix is not None and not token.startswith(self.target_prefix):
                continue
            candidate_ids.append(token_id)

        return candidate_ids

    def predict_token_probabilities(
        self,
        sequence: List[str],
        target_index: int,
    ) -> Dict[str, float]:
        """Predict the target from its left context and return valid-target probabilities."""
        if not isinstance(target_index, int):
            raise TypeError("target_index must be an integer.")
        if target_index < 0 or target_index >= len(sequence):
            raise IndexError("target_index is outside the sequence.")

        self.model.eval()

        left_tokens = sequence[:target_index]
        context_ids = [self.bos_id] + [
            self.token2id.get(token, self.unk_id) for token in left_tokens
        ]
        context_ids = context_ids[-self.context_length :]

        x = torch.tensor([context_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.model(x)
            last_logits = logits[0, -1, :]
            full_probabilities = torch.softmax(last_logits, dim=-1)

        candidate_ids = self._candidate_token_ids()
        if not candidate_ids:
            return {}

        candidate_tensor = full_probabilities[candidate_ids]
        candidate_total = candidate_tensor.sum()

        # Renormalize over tokens that are valid masked targets.
        if candidate_total.item() > 0:
            candidate_tensor = candidate_tensor / candidate_total
        else:
            candidate_tensor = torch.full_like(
                candidate_tensor,
                fill_value=1.0 / len(candidate_ids),
            )

        candidate_probabilities = candidate_tensor.detach().cpu().tolist()
        return {
            self.id2token[token_id]: float(probability)
            for token_id, probability in zip(candidate_ids, candidate_probabilities)
        }

    def save(self, checkpoint_path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save everything needed to reconstruct the trained model."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "vocab": self.vocab,
                "config": {
                    "target_prefix": self.target_prefix,
                    "embed_dim": self.embed_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "lr": self.lr,
                    "context_length": self.context_length,
                    "name": self.name,
                },
                "extra": extra or {},
            },
            checkpoint_path,
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: Path,
        device: Optional[str] = None,
    ) -> "MusicLSTMModel":
        """Load a checkpoint created by save()."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        map_location = choose_device(device)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        config = checkpoint["config"]
        model = cls(
            vocab=checkpoint["vocab"],
            target_prefix=config.get("target_prefix"),
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            lr=config.get("lr", 1e-3),
            context_length=config.get("context_length", 64),
            device=map_location,
            name=config.get("name", "lstm"),
        )
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        return model