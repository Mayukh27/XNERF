from __future__ import annotations

import torch
from torch import nn

from xnerf.utils.base import BaseModule


class APIEncoder(BaseModule):
    """Transformer encoder for API call sequences.

    Inputs:
        api_ids: LongTensor [B, T]
    Outputs:
        embedding: FloatTensor [B, 512]
    Forward:
        forward(api_ids) -> embedding
    Usage:
        model = APIEncoder(vocab_size=4096)
        z = model(torch.randint(0, 4096, (8, 256)))
    """

    def __init__(self, vocab_size: int = 8192, hidden_dim: int = 256, out_dim: int = 512, layers: int = 4, heads: int = 8, max_len: int = 1024):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, api_ids: torch.Tensor) -> torch.Tensor:
        b, t = api_ids.shape
        pos = torch.arange(t, device=api_ids.device).unsqueeze(0).expand(b, t)
        mask = api_ids.eq(0)
        h = self.token(api_ids) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        denom = (~mask).sum(dim=1, keepdim=True).clamp_min(1)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / denom
        return self.proj(pooled)

