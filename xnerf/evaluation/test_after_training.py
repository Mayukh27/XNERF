from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.evaluate import evaluate_predictions, save_confusion_matrix, save_tsne, save_umap
from xnerf.model import XNERFPlusPlus
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config


def _safe_family_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    mask = y_true >= 0
    if not mask.any():
        return {"family_top1_accuracy": None, "family_top5_accuracy": None}
    y_true_valid = y_true[mask]
    y_prob_valid = y_prob[mask]
    y_pred = y_prob_valid.argmax(axis=1)
    metrics: dict[str, float | None] = {
        "family_top1_accuracy": float(accuracy_score(y_true_valid, y_pred)),
    }
    labels = np.arange(y_prob_valid.shape[1])
    k = min(5, y_prob_valid.shape[1])
    try:
        metrics["family_top5_accuracy"] = float(top_k_accuracy_score(y_true_valid, y_prob_valid, k=k, labels=labels))
    except ValueError:
        metrics["family_top5_accuracy"] = None
    return metrics


def load_model(checkpoint_path: Path, cfg: dict, device: torch.device) -> XNERFPlusPlus:
    model = XNERFPlusPlus(
        num_classes=int(cfg["model"].get("num_classes", 2)),
        num_families=int(cfg["model"].get("num_families", 32)),
        field_time=int(cfg["model"].get("field_time", 16)),
        use_binary=bool(cfg["model"].get("use_binary", True)),
        disabled_modalities=cfg["model"].get("disabled_modalities", []),
        use_alignment=bool(cfg["model"].get("use_alignment", True)),
        use_grl=bool(cfg["model"].get("use_grl", True)),
        use_sfs=bool(cfg["model"].get("use_sfs", True)),
        use_mnef=bool(cfg["model"].get("use_mnef", True)),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model", payload.get("state_dict", payload))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.eval()


@torch.no_grad()
def run_inference(
    cfg: dict,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path,
    predictions_name: str = "test_predictions.npz",
    metrics_name: str = "test_metrics.json",
    save_plots: bool = True,
    desc: str = "test",
) -> dict:
    test_manifest = str(manifest_path)
    if not test_manifest:
        raise ValueError("config data.test_manifest is required for test evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint_path)

    model = load_model(checkpoint, cfg, device)
    ds = MalwareManifestDataset(test_manifest, require_cache=True)
    loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 4), shuffle=False, num_workers=cfg["training"].get("num_workers", 2), collate_fn=collate_dicts)

    probs, labels, family_probs, family_labels, embeddings, arch_labels, sample_ids = [], [], [], [], [], [], []
    for batch in tqdm(loader, desc=desc):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        probs.append(torch.softmax(outputs["malware_logits"], dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        if "family_logits" in outputs and "family_label" in batch:
            family_probs.append(torch.softmax(outputs["family_logits"], dim=-1).cpu().numpy())
            family_labels.append(batch["family_label"].cpu().numpy())
        embeddings.append(outputs["zero_shot_embedding"].cpu().numpy())
        arch_labels.append(batch["arch_id"].cpu().numpy())
        sample_ids.extend([str(x) for x in batch.get("sample_id", batch.get("sha256", []))])

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    emb = np.concatenate(embeddings)
    arch_true = np.concatenate(arch_labels)
    y_pred = y_prob.argmax(axis=1)

    metrics = evaluate_predictions(y_true, y_prob, arch_true=arch_true)
    if family_probs and family_labels:
        metrics.update(_safe_family_metrics(np.concatenate(family_labels), np.concatenate(family_probs)))
    diagnostics = {
        "checkpoint": str(checkpoint),
        "test_manifest": str(test_manifest),
        "num_test_samples": int(len(y_true)),
        "first_sample_ids": sample_ids[:10],
        "prediction_counts": {str(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / metrics_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_path / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    payload = {
        "y_true": y_true,
        "y_prob": y_prob,
        "embeddings": emb,
        "arch_true": arch_true,
        "sample_ids": np.array(sample_ids),
    }
    if family_probs and family_labels:
        payload["family_true"] = np.concatenate(family_labels)
        payload["family_prob"] = np.concatenate(family_probs)
    np.savez_compressed(out_path / predictions_name, **payload)
    if save_plots:
        save_confusion_matrix(y_true, y_pred, out_path / "confusion_matrix.png")
    if save_plots and len(y_true) >= 3:
        save_tsne(emb, y_true, out_path / "tsne.png")
        save_umap(emb, y_true, out_path / "umap.png")
    return metrics


@torch.no_grad()
def run_test(
    config_path: str,
    checkpoint_path: str | None = None,
    out_dir: str | Path = "runs/test",
    test_manifest: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    test_manifest = test_manifest or cfg["data"].get("test_manifest")
    checkpoint = Path(checkpoint_path) if checkpoint_path else Path(cfg["training"]["checkpoint_dir"]) / "best.pt"
    return run_inference(
        cfg=cfg,
        checkpoint_path=checkpoint,
        manifest_path=test_manifest,
        out_dir=out_dir,
        predictions_name="test_predictions.npz",
        metrics_name="test_metrics.json",
        save_plots=True,
        desc="test",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best X-NERF++ checkpoint on the test split")
    parser.add_argument("--config", default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--out", default="/kaggle/working/runs/test")
    args = parser.parse_args()
    metrics = run_test(args.config, args.checkpoint, args.out, test_manifest=args.test_manifest)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
