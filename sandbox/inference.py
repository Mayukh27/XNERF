from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from config import SandboxConfig
from feature_extractor import extract_modalities, make_model_batch
from xnerf.datasets.validation import family_names_from_metadata
from xnerf.model import XNERFPlusPlus
from xnerf.preprocessing.ontology import ARCH_TO_ID


ID_TO_ARCH = {idx: name for name, idx in ARCH_TO_ID.items()}


class InferenceError(RuntimeError):
    pass


def family_name(index: int, checkpoint_meta: dict[str, Any] | None = None) -> str:
    families = family_names_from_metadata(checkpoint_meta)
    if isinstance(families, list) and 0 <= index < len(families):
        return str(families[index])

    sidecar_candidates = [
        Path("data/processed/family_vocab.json"),
        Path("data/family_vocab.json"),
        Path("models/family_vocab.json"),
        Path("models/xnerf_local_inference.families.json"),
        Path("models/xnerf_local_inference.family_names.json"),
        Path("models/xnerf_local_inference.family_vocab.json"),
        Path("models/best.families.json"),
        Path("models/best.family_names.json"),
        Path("models/best.family_vocab.json"),
    ]
    for sidecar in sidecar_candidates:
        if not sidecar.exists():
            continue
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, list) and 0 <= index < len(loaded):
            return str(loaded[index])
        if isinstance(loaded, dict):
            family_names = loaded.get("family_names") or loaded.get("id_to_family") or loaded.get("families")
            if isinstance(family_names, list) and 0 <= index < len(family_names):
                return str(family_names[index])

    return f"family_{index}"


def load_model(config: SandboxConfig, device: torch.device) -> tuple[XNERFPlusPlus, dict[str, Any]]:
    checkpoint = Path(config.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location=device)
    except Exception as exc:
        raise InferenceError(f"could not load checkpoint {checkpoint}: {type(exc).__name__}: {exc}") from exc

    payload_dict = payload if isinstance(payload, dict) else {}
    model_cfg = payload_dict.get("model_config", {})
    model = XNERFPlusPlus(
        num_classes=int(model_cfg.get("num_classes", config.num_classes)),
        num_families=int(model_cfg.get("num_families", config.num_families)),
    ).to(device)
    state = payload_dict.get("model", payload_dict.get("state_dict", payload))
    if not isinstance(state, dict):
        raise InferenceError(f"checkpoint does not contain a model state dict: {checkpoint}")
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, payload_dict


def _check_outputs(outputs: dict[str, torch.Tensor]) -> None:
    for name, value in outputs.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise InferenceError(f"model produced non-finite output tensor: {name}")


@torch.no_grad()
def run_inference(file_path: str | Path, config: SandboxConfig) -> dict[str, Any]:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    started = time.perf_counter()
    features = extract_modalities(file_path, arch=config.arch)
    model, checkpoint_meta = load_model(config, device)
    batch = make_model_batch(features, device)
    try:
        outputs = model(batch)
    except Exception as exc:
        raise InferenceError(f"model inference failed: {type(exc).__name__}: {exc}") from exc
    _check_outputs(outputs)

    malware_prob = torch.softmax(outputs["malware_logits"], dim=-1)[0, -1].item()
    class_probs = torch.softmax(outputs["malware_logits"], dim=-1)[0]
    family_idx = int(torch.softmax(outputs["family_logits"], dim=-1)[0].argmax().item())
    arch_idx = int(outputs["arch_logits"][0].argmax().item())
    elapsed = time.perf_counter() - started
    metadata = features["metadata"]

    return {
        "file_name": metadata["file_name"],
        "file_path": metadata["path"],
        "sha256": metadata["sha256"],
        "malware_probability": malware_prob,
        "decision": "Malware" if malware_prob >= config.decision_threshold else "Benign",
        "predicted_family": family_name(family_idx, checkpoint_meta),
        "predicted_architecture": ID_TO_ARCH.get(arch_idx, f"arch_{arch_idx}"),
        "confidence_score": float(class_probs.max().item()),
        "inference_time_seconds": elapsed,
        "checkpoint": str(config.checkpoint),
        "device": str(device),
    }


def format_terminal_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "XNERF Terminal Inference Report",
            "=" * 31,
            f"File Name: {result['file_name']}",
            f"Malware Probability: {result['malware_probability']:.6f}",
            f"Malware/Benign Decision: {result['decision']}",
            f"Predicted Family: {result['predicted_family']}",
            f"Predicted Architecture: {result['predicted_architecture']}",
            f"Confidence Score: {result['confidence_score']:.6f}",
            f"Inference Time: {result['inference_time_seconds']:.3f}s",
        ]
    )

