"""
Text encoder for query processing.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    GRU-based text encoder for Korean queries.

    Encodes variable-length token sequences into fixed-dimension embeddings.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden: int = 256):
        """
        Initialize text encoder.

        Args:
            vocab_size: Size of vocabulary
            emb_dim: Embedding dimension
            hidden: GRU hidden size
        """
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden * 2, emb_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode text queries.

        Args:
            tokens: (B, L) long tensor of token IDs, padded with 0
            lengths: (B,) long tensor of valid lengths

        Returns:
            (B, D) float tensor of query embeddings
        """
        x = self.emb(tokens)  # (B, L, E)

        # Pack padded sequence for efficient GRU processing
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process with GRU
        out, h = self.gru(packed)

        # Concatenate final hidden states from both directions
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)

        # Project to embedding dimension
        q = self.proj(h_cat)  # (B, D)

        return q
