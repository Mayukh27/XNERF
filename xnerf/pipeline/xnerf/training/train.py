"""train.py — standalone training and validation entry-points.

Called directly by kaggle_run.py / local_run.py, but also usable standalone:
    python -m xnerf.training.train --config config.yaml
    python -m xnerf.training.train --config config.yaml --validate-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.test_after_training import load_model
from xnerf.model import XNERFPlusPlus
from xnerf.training.trainer import XNerfTrainer
from xnerf.utils.base import move_to_device
from xnerf.utils.config import load_config
from xnerf.utils.seed import seed_everything


def run_training(config_path: str = "config.yaml") -> dict:
    """Train the model and return metrics dict.

    Writes:
        <checkpoint_dir>/best.pt          — best checkpoint
        runs/train_metrics.json           — metrics (also returned)
    """
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 1337))

    train_ds = MalwareManifestDataset(cfg["data"]["train_manifest"])
    val_path = cfg["data"].get("val_manifest")
    val_ds = MalwareManifestDataset(val_path) if val_path else None

    model = XNERFPlusPlus(
        num_classes=cfg["model"]["num_classes"],
        num_families=cfg["model"]["num_families"],
    )

    # Strip keys not accepted by XNerfTrainer so the config can include extras.
    trainer_keys = {
        "batch_size", "lr", "epochs", "grad_accum",
        "num_workers", "checkpoint_dir", "patience",
    }
    trainer_kwargs = {k: v for k, v in cfg["training"].items() if k in trainer_keys}

    trainer = XNerfTrainer(model, train_ds, val_ds, **trainer_kwargs)
    metrics = trainer.fit()

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


@torch.no_grad()
def run_validation(config_path: str = "config.yaml", checkpoint_path: str | None = None) -> dict:
    """Run the validation loop on the val split and return metrics.

    Writes:
        runs/val_metrics.json
    """
    from tqdm import tqdm

    cfg = load_config(config_path)
    val_manifest = cfg["data"].get("val_manifest")
    if not val_manifest:
        raise ValueError("config data.val_manifest is required for validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(
        checkpoint_path
        or cfg.get("export", {}).get("checkpoint")
        or (Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    )
    model = load_model(ckpt_path, cfg, device)

    from torch.utils.data import DataLoader
    import numpy as np
    from xnerf.evaluation.evaluate import evaluate_predictions

    ds = MalwareManifestDataset(val_manifest)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    probs, labels = [], []
    for batch in tqdm(loader, desc="validate"):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        probs.append(torch.softmax(outputs["malware_logits"], dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    metrics = evaluate_predictions(y_true, y_prob)

    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="X-NERF++ training / validation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip training; just run the validation loop on an existing checkpoint",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for --validate-only")
    args = parser.parse_args()

    if args.validate_only:
        metrics = run_validation(args.config, checkpoint_path=args.checkpoint)
    else:
        metrics = run_training(args.config)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
