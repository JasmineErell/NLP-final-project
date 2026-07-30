import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any
import numpy as np

from .common import BaseModel  # Assuming BaseModel is defined in src/models/common.py


class SequenceDataset(Dataset):
    """Dataset wrapper for token ID sequences."""

    def __init__(self, sequences: List[List[int]], seq_len: int = 64, pad_id: int = 0):
        self.samples = []
        self.seq_len = seq_len
        self.pad_id = pad_id

        # Chunk sequences into fixed-length inputs and targets for language modeling
        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(0, len(seq) - 1, seq_len):
                chunk_x = seq[i : i + seq_len]
                chunk_y = seq[i + 1 : i + 1 + seq_len]

                # Pad if necessary
                if len(chunk_x) < seq_len:
                    chunk_x = chunk_x + [pad_id] * (seq_len - len(chunk_x))
                    chunk_y = chunk_y + [pad_id] * (seq_len - len(chunk_y))

                self.samples.append((torch.tensor(chunk_x, dtype=torch.long), 
                                     torch.tensor(chunk_y, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PyTorchLSTMModule(nn.Module):
    """Core PyTorch LSTM Architecture."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, hidden = self.lstm(embeds, hidden)  # (batch_size, seq_len, hidden_dim)
        logits = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return logits, hidden


class MusicLSTMModel(BaseModel):
    """Wrapper class integrating PyTorch LSTM into the project pipeline."""

    def __init__(
        self,
        vocab: List[str],
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str = None
    ):
        super().__init__()
        self.vocab = vocab
        self.token2id = {token: idx for idx, token in enumerate(vocab)}
        self.id2token = {idx: token for idx, token in enumerate(vocab)}
        
        self.pad_token = "<PAD>" if "<PAD>" in self.token2id else vocab[0]
        self.pad_id = self.token2id[self.pad_token]
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchLSTMModule(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=self.pad_id
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def train_model(
        self,
        train_sequences: List[List[str]],
        epochs: int = 10,
        batch_size: int = 32,
        seq_len: int = 64
    ) -> Dict[str, Any]:
        """Trains the PyTorch LSTM on tokenized ABC sequences."""
        # Convert string tokens to IDs
        numeric_sequences = [
            [self.token2id[t] for t in seq if t in self.token2id]
            for seq in train_sequences
        ]

        dataset = SequenceDataset(numeric_sequences, seq_len=seq_len, pad_id=self.pad_id)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(x_batch)

                # Reshape for CrossEntropyLoss: (batch_size * seq_len, vocab_size) vs (batch_size * seq_len)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(dataloader))
            history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return {"train_loss": history}

    def predict_next_token_probs(self, context_tokens: List[str]) -> Dict[str, float]:
        """Returns next-token probability distribution given a context sequence."""
        self.model.eval()
        numeric_context = [self.token2id[t] for t in context_tokens if t in self.token2id]

        if not numeric_context:
            # Fallback to uniform if no valid context tokens
            uniform_prob = 1.0 / len(self.vocab)
            return {token: uniform_prob for token in self.vocab}

        x = torch.tensor([numeric_context], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(x)
            last_token_logits = logits[0, -1, :]  # Logits for the next prediction
            probs = torch.softmax(last_token_logits, dim=-1).cpu().numpy()

        return {self.id2token[i]: float(probs[i]) for i in range(len(probs))}