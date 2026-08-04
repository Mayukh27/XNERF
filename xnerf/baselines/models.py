from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from xnerf.encoders.api import APIEncoder
from xnerf.encoders.binary_image import BinaryImageEncoder
from xnerf.encoders.cfg import CFGEncoder
from xnerf.utils.base import BaseModule


class Byte_CNN(BaseModule):
    """CNN byte-image baseline. Input [B,1,H,W], output logits [B,C]."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image[:, :1])


class Transformer_Api(BaseModule):
    """Transformer token baseline. Input token ids [B,T], output logits [B,C]."""

    def __init__(self, vocab_size: int = 8192, num_classes: int = 2, num_families: int | None = None):
        super().__init__()
        self.encoder = APIEncoder(vocab_size=vocab_size)
        self.head = nn.Linear(512, num_classes)
        self.family_head = nn.Linear(512, num_families) if num_families else None

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(token_ids)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.encode(token_ids)
        logits = self.head(features)
        if self.family_head is None:
            return logits
        return {"malware_logits": logits, "family_logits": self.family_head(features)}


class LateFusion(BaseModule):
    """Hybrid static/dynamic baseline. Inputs image [B,1,H,W], api_ids [B,T]."""

    def __init__(self, num_classes: int = 2, num_families: int | None = None):
        super().__init__()
        self.image = BinaryImageEncoder()
        self.api = APIEncoder()
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, num_classes))
        self.family_head = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, num_families)) if num_families else None

    def encode(self, image: torch.Tensor, api_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.image(image), self.api(api_ids)], dim=-1)

    def forward(self, image: torch.Tensor, api_ids: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.encode(image, api_ids)
        logits = self.head(features)
        if self.family_head is None:
            return logits
        return {"malware_logits": logits, "family_logits": self.family_head(features)}

"""
class SAFE(BaseModule):
    """SAFE-style self-attentive sequence baseline for tokenized instruction/API streams."""

    def __init__(self, vocab_size: int = 8192, embed_dim: int = 256, hidden_dim: int = 256, num_classes: int = 2, num_families: int | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Linear(hidden_dim * 2, num_classes)
        self.family_head = nn.Linear(hidden_dim * 2, num_families) if num_families else None

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.clamp_min(0)
        mask = token_ids.ne(0)
        encoded, _ = self.encoder(self.embedding(token_ids))
        scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (encoded * weights).sum(dim=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.encode(token_ids)
        logits = self.head(features)
        if self.family_head is None:
            return logits
        return {"malware_logits": logits, "family_logits": self.family_head(features)}
"""

class CrossArchitectureSiamese(BaseModule):
    """Siamese cross-architecture baseline. Inputs two feature tensors [B,D]."""

    def __init__(self, input_dim: int = 512, embed_dim: int = 256):
        super().__init__()
        self.tower = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, embed_dim))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        za, zb = F.normalize(self.tower(a), dim=-1), F.normalize(self.tower(b), dim=-1)
        return {"za": za, "zb": zb, "similarity": (za * zb).sum(dim=-1)}


class CFG_GNN(BaseModule):
    """GNN CFG baseline. Inputs PyG graph tensors, output logits [B,C]."""

    def __init__(self, node_dim: int = 64, num_classes: int = 2, num_families: int | None = None):
        super().__init__()
        self.cfg = CFGEncoder(node_dim=node_dim)
        self.head = nn.Linear(512, num_classes)
        self.family_head = nn.Linear(512, num_families) if num_families else None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.cfg(x, edge_index, batch)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.encode(x, edge_index, batch)
        logits = self.head(features)
        if self.family_head is None:
            return logits
        return {"malware_logits": logits, "family_logits": self.family_head(features)}
