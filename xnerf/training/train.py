from __future__ import annotations

import argparse
import json
from pathlib import Path

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.model import XNERFPlusPlus
from xnerf.training.trainer import XNerfTrainer
from xnerf.utils.config import load_config
from xnerf.utils.seed import seed_everything


def run_training(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 1337))
    train_ds = MalwareManifestDataset(cfg["data"]["train_manifest"])
    val_path = cfg["data"].get("val_manifest")
    val_ds = MalwareManifestDataset(val_path) if val_path else None
    model = XNERFPlusPlus(num_classes=cfg["model"]["num_classes"], num_families=cfg["model"]["num_families"])
    trainer = XNerfTrainer(model, train_ds, val_ds, **cfg["training"])
    metrics = trainer.fit()
    Path("runs").mkdir(exist_ok=True)
    Path("runs/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
