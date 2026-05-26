from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from xnerf.datasets.build_dataset import build_manifest
from xnerf.datasets.extract_archives import extract_dataset_archives
from xnerf.deployment.export_checkpoint import export_checkpoint
from xnerf.evaluation.test_after_training import run_test
from xnerf.training.train import run_training
from xnerf.utils.config import load_config
from xnerf.zero_shot.build_prototypes import build_family_prototypes
from xnerf.zero_shot.evaluate_zero_shot import evaluate_zero_shot


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _manifests_exist(full: Path, train: Path, val: Path, test: Path) -> bool:
    return all(path.exists() for path in (full, train, val, test))


def run_kaggle_pipeline(
    config_path: str = "xnerf/configs/kaggle.yaml",
    skip_extract: bool = False,
    rebuild_manifests: bool = False,
) -> dict[str, Any]:
    """Run the complete Kaggle workflow once.

    Steps:
        1. Extract archives from data.archive_root into data.root/raw.
        2. Build full/train/val/test manifests.
        3. Train and checkpoint best.pt.
        4. Evaluate the held-out test split.
        5. Build and evaluate zero-shot prototype bank.
        6. Export local inference checkpoint.
        7. Write one combined summary JSON.
    """

    cfg = load_config(config_path)

    print("=== CONFIG DEBUG ===")
    print("Train:", cfg["data"]["train_manifest"])
    print("Val:", cfg["data"]["val_manifest"])
    print("Test:", cfg["data"]["test_manifest"])
    print("====================")


    data_root = Path(cfg["data"]["root"])
    archive_root = Path(cfg["data"]["archive_root"])
    full_manifest = Path(cfg["data"].get("full_manifest", data_root / "processed" / "manifest.jsonl"))
    train_manifest = Path(cfg["data"]["train_manifest"])
    val_manifest = Path(cfg["data"]["val_manifest"])
    test_manifest = Path(cfg["data"]["test_manifest"])
    checkpoint = Path(cfg["export"].get("checkpoint", Path(cfg["training"]["checkpoint_dir"]) / "best.pt"))
    export_path = Path(cfg["export"]["output"])
    test_dir = Path(cfg["outputs"].get("test_dir", "/kaggle/working/runs/test"))
    zero_shot_dir = Path(cfg["outputs"].get("zero_shot_dir", "/kaggle/working/runs/zero_shot"))
    prototype_bank = Path(cfg["outputs"].get("prototype_bank", zero_shot_dir / "prototypes.pt"))
    final_dir = Path(cfg["outputs"].get("final_dir", "/kaggle/working/xnerf_output"))
    final_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"config": config_path, "steps": {}}

    if skip_extract:
        summary["steps"]["extract"] = {"skipped": True}
    else:
        summary["steps"]["extract"] = extract_dataset_archives(archive_root, data_root)

    if rebuild_manifests or not _manifests_exist(full_manifest, train_manifest, val_manifest, test_manifest):
        build_manifest(
            root=data_root,
            out=full_manifest,
            make_splits=True,
            train_ratio=float(cfg.get("splits", {}).get("train_ratio", 0.8)),
            val_ratio=float(cfg.get("splits", {}).get("val_ratio", 0.1)),
            seed=int(cfg.get("seed", 1337)),
        )
        summary["steps"]["manifests"] = {"rebuilt": True}
    else:
        summary["steps"]["manifests"] = {"reused": True}
    summary["steps"]["manifests"].update(
        {
            "full": str(full_manifest),
            "train": str(train_manifest),
            "val": str(val_manifest),
            "test": str(test_manifest),
        }
    )

    train_metrics = run_training(config_path)
    summary["steps"]["train"] = train_metrics

    test_metrics = run_test(config_path=config_path, checkpoint_path=str(checkpoint), out_dir=test_dir)
    summary["steps"]["test"] = test_metrics

    build_family_prototypes(
        config_path=config_path,
        checkpoint_path=str(checkpoint),
        manifest_path=str(train_manifest),
        output_path=str(prototype_bank),
    )
    zero_shot_metrics = evaluate_zero_shot(
        config_path=config_path,
        checkpoint_path=str(checkpoint),
        manifest_path=str(test_manifest),
        prototype_path=str(prototype_bank),
        out_dir=zero_shot_dir,
    )
    summary["steps"]["zero_shot"] = zero_shot_metrics

    exported = export_checkpoint(checkpoint, Path(config_path), export_path)
    summary["steps"]["export"] = {"local_checkpoint": str(exported)}
    summary["metrics"] = {
        "accuracy": test_metrics.get("accuracy"),
        "precision": test_metrics.get("precision"),
        "recall": test_metrics.get("recall"),
        "f1": test_metrics.get("f1"),
        "roc_auc": test_metrics.get("roc_auc"),
        "cross_architecture_accuracy": test_metrics.get("cross_architecture_accuracy"),
        "zero_shot_accuracy": zero_shot_metrics.get("zero_shot_accuracy"),
        "zero_shot_f1": zero_shot_metrics.get("zero_shot_f1"),
    }

    final_files = {
        "local_checkpoint": final_dir / "xnerf_local_inference.pt",
        "test_metrics": final_dir / "test_metrics.json",
        "zero_shot_metrics": final_dir / "zero_shot_metrics.json",
        "prototype_bank": final_dir / "prototypes.pt",
        "confusion_matrix": final_dir / "confusion_matrix.png",
        "test_predictions": final_dir / "test_predictions.npz",
        "zero_shot_predictions": final_dir / "zero_shot_predictions.npz",
    }
    _copy_if_exists(export_path, final_files["local_checkpoint"])
    _copy_if_exists(test_dir / "test_metrics.json", final_files["test_metrics"])
    _copy_if_exists(zero_shot_dir / "zero_shot_metrics.json", final_files["zero_shot_metrics"])
    _copy_if_exists(prototype_bank, final_files["prototype_bank"])
    _copy_if_exists(test_dir / "confusion_matrix.png", final_files["confusion_matrix"])
    _copy_if_exists(test_dir / "test_predictions.npz", final_files["test_predictions"])
    _copy_if_exists(zero_shot_dir / "zero_shot_predictions.npz", final_files["zero_shot_predictions"])

    summary["final_output_dir"] = str(final_dir)
    summary["final_files"] = {k: str(v) for k, v in final_files.items() if v.exists()}
    _write_json(final_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete X-NERF++ Kaggle pipeline")
    parser.add_argument("--config", default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--skip-extract", action="store_true", help="Use existing /kaggle/working/data/raw contents")
    parser.add_argument(
        "--rebuild-manifests",
        action="store_true",
        help="Force manifest rebuild even if processed manifests already exist",
    )
    args = parser.parse_args()
    run_kaggle_pipeline(args.config, skip_extract=args.skip_extract, rebuild_manifests=args.rebuild_manifests)


if __name__ == "__main__":
    main()
