from __future__ import annotations

import torch
from torch import nn

from xnerf.alignment.adversarial import CrossArchitectureAligner
from xnerf.encoders.api import APIEncoder
from xnerf.encoders.binary_image import BinaryImageEncoder
from xnerf.encoders.memory import MemoryEncoder
from xnerf.encoders.network import NetworkEncoder
from xnerf.encoders.isr import ISREncoder
from xnerf.fields.mnef import MNEF
from xnerf.renderer.trajectory_decoder import TrajectoryDecoder
from xnerf.synchronization.sfs import SemanticFieldSynchronizer
from xnerf.utils.base import BaseModule
from xnerf.encoders.cfg import CFGEncoder

class XNERFPlusPlus(BaseModule):
    """End-to-end X-NERF++ model.

    Inputs:
        batch dict with binary_image [B,1,H,W], api_ids [B,T], memory_trace [B,T,C],
        network_ids [B,T], arch_id [B].
    Outputs:
        malware_logits [B,num_classes], family_logits [B,num_families],
        zero_shot_embedding [B,2048], trajectory logits, MNEF field.
    Tensor dimensions:
        synchronized state [B,field_time,2048], field [B,field_time,1024].
    Usage:
        model = XNERFPlusPlus(num_classes=2, num_families=32)
        out = model(batch)
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_families: int = 32,
        field_time: int = 16,
        use_binary: bool = True,
        disabled_modalities: list[str] | tuple[str, ...] | set[str] | None = None,
        use_alignment: bool = True,
        use_grl: bool = True,
        use_sfs: bool = True,
        use_mnef: bool = True,
    ):
        super().__init__()
        disabled = {str(name).lower() for name in (disabled_modalities or [])}
        self.use_binary = use_binary
        self.disabled_modalities = disabled
        self.use_alignment = bool(use_alignment)
        self.use_grl = bool(use_grl)
        self.use_sfs = bool(use_sfs)
        self.use_mnef = bool(use_mnef)
        self.field_time = field_time
        self.binary = BinaryImageEncoder() if use_binary and "binary" not in disabled else None
        self.api = APIEncoder() if "api" not in disabled else None
        self.graph = CFGEncoder(node_dim=4) if "cfg" not in disabled else None
        self.memory = MemoryEncoder() if "memory" not in disabled else None
        self.network = NetworkEncoder() if "network" not in disabled else None
        self.isr = ISREncoder() if "isr" not in disabled else None
        self.sfs = SemanticFieldSynchronizer() if self.use_sfs else None
        self.simple_fusion = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
        ) if not self.use_sfs else None
        self.arch_embed = nn.Embedding(7, 64)
        self.memory_context = nn.Linear(512, 512)
        self.mnef = MNEF() if self.use_mnef else None
        self.aligner = CrossArchitectureAligner(feature_dim=2048, num_arch=7) if self.use_alignment else None
        self.renderer = TrajectoryDecoder() if self.use_mnef else None
        self.malware_head = nn.Linear(2048, num_classes)
        self.family_head = nn.Linear(2048, num_families)

    def _zero_embedding(self, batch: dict[str, torch.Tensor], dim: int = 512) -> torch.Tensor:
        reference = batch.get("api_ids")
        if not torch.is_tensor(reference):
            for value in batch.values():
                if torch.is_tensor(value) and value.dim() > 0:
                    reference = value
                    break
        if not torch.is_tensor(reference):
            raise ValueError("cannot infer batch size/device for zero ablation embedding")
        return torch.zeros(reference.shape[0], dim, device=reference.device, dtype=torch.float32)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = {}
        if self.api is not None:
            embeddings["api"] = self.api(batch["api_ids"])
        if self.memory is not None:
            embeddings["memory"] = self.memory(batch["memory_trace"])
        if self.network is not None:
            embeddings["network"] = self.network(batch["network_ids"])
        if self.binary is not None and "binary_image" in batch:
            embeddings["binary"] = self.binary(batch["binary_image"])

        if (
            self.graph is not None
            and "graph_x" in batch
            and "graph_edge_index" in batch
            and batch["graph_x"].numel() > 0
        ):
               cfg = self.graph( batch["graph_x"], batch["graph_edge_index"], batch["graph_batch"],)  
               batch_size = batch["api_ids"].shape[0]
               cfg_full = torch.zeros( batch_size, cfg.shape[1], device=cfg.device, dtype=cfg.dtype,)
               cfg_full[batch["graph_sample_ids"]] = cfg
               embeddings["cfg"] = cfg_full

        elif "cfg" not in self.disabled_modalities:
             embeddings["cfg"] = torch.zeros( batch["api_ids"].shape[0], 512, device=batch["api_ids"].device, )      
        
        
        if self.isr is not None and "isr" in batch and batch["isr"].numel() > 0:
            embeddings["isr"] = self.isr(batch["isr"])

        if not embeddings:
            raise ValueError("all modalities are disabled; at least one encoder must remain active")
        if self.sfs is not None:
            semantic = self.sfs(embeddings, time_steps=self.field_time)
        else:
            fused = torch.stack(list(embeddings.values()), dim=1).mean(dim=1)
            semantic = self.simple_fusion(fused).unsqueeze(1).expand(-1, self.field_time, -1)
        pooled = semantic.mean(dim=1)
        if self.aligner is not None:
            aligned = self.aligner(pooled, grl_lambda=1.0 if self.use_grl else 0.0)
            task_features = aligned["aligned"]
            out = {"arch_logits": aligned["arch_logits"]}
        else:
            task_features = pooled
            out = {}

        out.update(
            {
                "malware_logits": self.malware_head(task_features),
                "family_logits": self.family_head(task_features),
                "zero_shot_embedding": task_features,
            }
        )
        if self.mnef is not None and self.renderer is not None:
            b, t, _ = semantic.shape
            coords = torch.linspace(0, 1, t, device=semantic.device).view(1, t, 1).expand(b, t, 1)
            arch = self.arch_embed(batch["arch_id"].clamp(min=0, max=self.arch_embed.num_embeddings - 1)).unsqueeze(1).expand(b, t, 64)
            memory_embedding = embeddings.get("memory")
            if memory_embedding is None:
                memory_embedding = self._zero_embedding(batch, dim=512).to(device=semantic.device, dtype=semantic.dtype)
            mem = self.memory_context(memory_embedding).unsqueeze(1).expand(b, t, 512)
            field_out = self.mnef(coords, coords, semantic, mem, arch)
            traj = self.renderer(field_out["field"])
            out.update(field_out)
            out.update(traj)
        return out
