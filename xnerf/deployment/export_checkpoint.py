from __future__ import annotations

import argparse
from pathlib import Path

import torch

from xnerf.utils.config import load_config


def strip_module_prefix(state_dict: dict) -> dict:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def export_checkpoint(input_checkpoint: Path, config_path: Path, output_path: Path) -> Path:
    cfg = load_config(config_path)
    raw = torch.load(input_checkpoint, map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw))
    payload = {
        "format": "xnerf-local-inference-v1",
        "model_config": cfg.get("model", {}),
        "state_dict": strip_module_prefix(state),
        "epoch": raw.get("epoch"),
        "val_loss": raw.get("val_loss"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kaggle training checkpoint for local inference")
    parser.add_argument("--input", type=Path, default=Path("/kaggle/working/checkpoints/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("xnerf/configs/kaggle.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/export/xnerf_local_inference.pt"))
    args = parser.parse_args()
    out = export_checkpoint(args.input, args.config, args.output)
    print(f"exported {out}")


if __name__ == "__main__":
    main()

