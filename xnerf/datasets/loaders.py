from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from xnerf.preprocessing.ontology import ARCH_TO_ID
from xnerf.utils.base import DatasetLoader
from xnerf.utils.io import read_jsonl


class MalwareManifestDataset(DatasetLoader):
    """JSONL-backed unified malware dataset.

    Inputs:
        manifest_path with fields path, label, family, arch, optional *_path tensors.
    Outputs:
        dict with tensors: binary_image [1,H,W], api_ids [T], memory_trace [T,C],
        network_ids [T], isr [T,4], arch_id [], label [].
    Tensor dimensions:
        binary image [1,256,256], api/network [256], memory [512,8], isr [1024,4].
    Usage:
        ds = MalwareManifestDataset("data/processed/manifest.jsonl")
        item = ds[0]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 256,
        seq_len: int = 256,
        memory_len: int = 512,
        isr_len: int = 1024,
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.image_size = image_size
        self.seq_len = seq_len
        self.memory_len = memory_len
        self.isr_len = isr_len

    def __len__(self) -> int:
        return len(self.rows)

    def _binary_image(self, path: Path) -> torch.Tensor:
        data = np.frombuffer(path.read_bytes()[: self.image_size * self.image_size], dtype=np.uint8)
        if data.size == 0:
            data = np.zeros(1, dtype=np.uint8)
        data = np.pad(data, (0, max(0, self.image_size * self.image_size - data.size)))[: self.image_size * self.image_size]
        return torch.from_numpy(data.reshape(1, self.image_size, self.image_size).astype("float32") / 255.0)

    def _memory_trace(self, row: dict[str, Any]) -> torch.Tensor:
        out = torch.zeros(self.memory_len, 8, dtype=torch.float32)
        feature_path = row.get("feature_path")
        if feature_path and Path(feature_path).exists():
            loaded = torch.load(feature_path, map_location="cpu").float()
            loaded = torch.nan_to_num(loaded, nan=0.0, posinf=0.0, neginf=0.0)
            if loaded.dim() == 1:
                loaded = torch.nn.functional.pad(loaded, (0, max(0, self.memory_len * 8 - loaded.numel())))[: self.memory_len * 8].view(self.memory_len, 8)
            out[: min(self.memory_len, loaded.shape[0]), : min(8, loaded.shape[1])] = loaded[: self.memory_len, :8]
        return out

    def _load_ids(self, row: dict[str, Any], key: str) -> torch.Tensor:
        values = row.get(key, [])
        out = torch.zeros(self.seq_len, dtype=torch.long)
        if values:
            vals = torch.tensor(values[: self.seq_len], dtype=torch.long)
            out[: vals.numel()] = vals
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["path"])
        isr = torch.zeros(self.isr_len, 4, dtype=torch.long)
        if row.get("isr_path") and Path(row["isr_path"]).exists():
            loaded = torch.load(row["isr_path"], map_location="cpu")
            if loaded.is_floating_point():
                loaded = torch.nan_to_num(loaded, nan=0.0, posinf=0.0, neginf=0.0)
            isr[: min(self.isr_len, loaded.shape[0])] = loaded[: self.isr_len]
        data_type = row.get("data_type")
        return {
            "binary_image": torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
            if data_type in {"feature_csv", "feature_parquet"}
            else self._binary_image(path),
            "api_ids": self._load_ids(row, "api_ids"),
            "network_ids": self._load_ids(row, "network_ids"),
            "memory_trace": self._memory_trace(row),
            "isr": isr,
            "arch_id": torch.tensor(ARCH_TO_ID.get(row.get("arch", "x86"), 0), dtype=torch.long),
            "label": torch.tensor(int(row.get("label", 0)), dtype=torch.long),
            "dataset": row.get("dataset", "unknown"),
            "family": row.get("family", "unknown"),
            "path": row.get("path", ""),
            "row_index": row.get("row_index", index),
            "sample_id": row.get("sample_id", ""),
            "sha256": row.get("sha256", ""),
        }


class UnifiedMalwareDataset(MalwareManifestDataset):
    """Alias for the production multimodal loader."""
