from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xnerf.preprocessing.ontology import ARCH_TO_ID
from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline


SUPPORTED_EXTENSIONS = {
    "",
    ".bin",
    ".dat",
    ".dll",
    ".elf",
    ".exe",
    ".scr",
    ".so",
    ".sys",
    ".apk",
}


class FeatureExtractionError(RuntimeError):
    pass


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not path.is_file():
        raise FeatureExtractionError(f"unsupported input, expected a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise FeatureExtractionError(f"unsupported file extension '{path.suffix or '<none>'}' for {path}")
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError as exc:
        raise FeatureExtractionError(f"could not read file: {path}") from exc


def binary_image_from_bytes(data: bytes, image_size: int = 256) -> torch.Tensor:
    values = np.frombuffer(data[: image_size * image_size], dtype=np.uint8)
    if values.size == 0:
        values = np.zeros(1, dtype=np.uint8)
    values = np.pad(values, (0, max(0, image_size * image_size - values.size)))[: image_size * image_size]
    return torch.from_numpy(values.reshape(1, image_size, image_size).astype("float32") / 255.0)


def memory_trace_from_bytes(data: bytes, rows: int = 512, cols: int = 8) -> torch.Tensor:
    out = torch.zeros(rows * cols, dtype=torch.float32)
    if data:
        values = torch.tensor(list(data[: rows * cols]), dtype=torch.float32)
        if values.numel() > 1:
            values = (values - values.mean()) / values.std().clamp_min(1e-6)
        out[: values.numel()] = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return out.view(rows, cols)


def extract_modalities(path: str | Path, arch: str = "x86") -> dict[str, Any]:
    sample_path = Path(path)
    validate_input_file(sample_path)
    try:
        data = sample_path.read_bytes()
    except OSError as exc:
        raise FeatureExtractionError(f"feature extraction failed while reading {sample_path}") from exc

    try:
        isr = ArchitectureNormalizationPipeline(arch=arch).process({"bytes": data, "arch": arch})
    except Exception as exc:
        raise FeatureExtractionError(f"ISR feature extraction failed for {sample_path}: {type(exc).__name__}: {exc}") from exc

    sha256 = hashlib.sha256(data).hexdigest()
    return {
        "binary_image": binary_image_from_bytes(data),
        "memory_trace": memory_trace_from_bytes(data),
        "api_ids": torch.zeros(256, dtype=torch.long),
        "network_ids": torch.zeros(256, dtype=torch.long),
        "isr": isr[:1024].long(),
        "arch_id": torch.tensor(ARCH_TO_ID.get(arch, 0), dtype=torch.long),
        "label": torch.tensor(0, dtype=torch.long),
        "metadata": {
            "path": str(sample_path),
            "file_name": sample_path.name,
            "size_bytes": len(data),
            "sha256": sha256,
            "arch": arch,
        },
    }


def make_model_batch(features: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
    return batch

