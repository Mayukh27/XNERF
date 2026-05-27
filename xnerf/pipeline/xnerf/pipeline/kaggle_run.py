"""kaggle_run.py — split pipeline for Kaggle.

Subcommands (run in order):
    build-manifest   Extract archives and build train/val/test manifests.
    train            Train the model; writes checkpoints/best.pt + runs/train_done.json.
    validate         Run validation loop on val split and report metrics.
    test             Evaluate best.pt on the test split.
    zero-shot        Build prototype bank then evaluate zero-shot accuracy.
    export           Package best.pt into a local-inference checkpoint.

    all              Run all of the above in sequence (legacy behaviour).

Examples:
    # Kaggle notebook cell 1
    !python -m xnerf.pipeline.kaggle_run build-manifest --config xnerf/configs/kaggle.yaml

    # Kaggle notebook cell 2
    !python -m xnerf.pipeline.kaggle_run train --config xnerf/configs/kaggle.yaml

    # After downloading / inspecting outputs:
    !python -m xnerf.pipeline.kaggle_run test --config xnerf/configs/kaggle.yaml
"""
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
from xnerf.training.train import run_training, run_validation
from xnerf.utils.config import load_config
from xnerf.zero_shot.build_prototypes import build_family_prototypes
from xnerf.zero_shot.evaluate_zero_shot import evaluate_zero_shot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _manifests_exist(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def _resolve_paths(cfg: dict) -> dict[str, Path]:
    """Return a flat dict of all well-known paths derived from config."""
    data_root = Path(cfg["data"]["root"])
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    zero_shot_dir = Path(cfg["outputs"].get("zero_shot_dir", "/kaggle/working/runs/zero_shot"))
    final_dir = Path(cfg["outputs"].get("final_dir", "/kaggle/working/xnerf_output"))
    return {
        "data_root": data_root,
        "archive_root": Path(cfg["data"]["archive_root"]),
        "full_manifest": Path(cfg["data"].get("full_manifest", data_root / "processed" / "manifest.jsonl")),
        "train_manifest": Path(cfg["data"]["train_manifest"]),
        "val_manifest": Path(cfg["data"]["val_manifest"]),
        "test_manifest": Path(cfg["data"]["test_manifest"]),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint": Path(cfg["export"].get("checkpoint", checkpoint_dir / "best.pt")),
        "export_output": Path(cfg["export"]["output"]),
        "train_metrics": Path(cfg["outputs"].get("train_metrics", "/kaggle/working/runs/metrics.json")),
        "test_dir": Path(cfg["outputs"].get("test_dir", "/kaggle/working/runs/test")),
        "zero_shot_dir": zero_shot_dir,
        "prototype_bank": Path(cfg["outputs"].get("prototype_bank", zero_shot_dir / "prototypes.pt")),
        "final_dir": final_dir,
        "train_done": checkpoint_dir / "train_done.json",
    }


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------

def cmd_build_manifest(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if args.skip_extract:
        result["extract"] = {"skipped": True}
    else:
        result["extract"] = extract_dataset_archives(p["archive_root"], p["data_root"])

    if args.rebuild_manifests or not _manifests_exist(
        p["full_manifest"], p["train_manifest"], p["val_manifest"], p["test_manifest"]
    ):
        build_manifest(
            root=p["data_root"],
            out=p["full_manifest"],
            make_splits=True,
            train_ratio=float(cfg.get("splits", {}).get("train_ratio", 0.8)),
            val_ratio=float(cfg.get("splits", {}).get("val_ratio", 0.1)),
            seed=int(cfg.get("seed", 1337)),
        )
        result["manifests"] = {"rebuilt": True}
    else:
        result["manifests"] = {"reused": True}

    result["manifests"].update({
        "full": str(p["full_manifest"]),
        "train": str(p["train_manifest"]),
        "val": str(p["val_manifest"]),
        "test": str(p["test_manifest"]),
    })

    _write_json(p["final_dir"] / "build_manifest_result.json", result)
    print(json.dumps(result, indent=2))
    return result


def cmd_train(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if p["train_done"].exists() and not args.force:
        print(f"[train] Checkpoint already exists at {p['checkpoint']}.")
        print(f"        Delete {p['train_done']} or pass --force to retrain.")
        return _read_json(p["train_done"])

    metrics = run_training(args.config)

    # Write sentinel so downstream commands know training completed.
    sentinel = {"status": "done", "checkpoint": str(p["checkpoint"]), **metrics}
    _write_json(p["train_done"], sentinel)
    _write_json(p["train_metrics"], metrics)
    _copy_if_exists(p["train_metrics"], p["final_dir"] / "train_metrics.json")

    print(json.dumps(sentinel, indent=2))
    return sentinel


def cmd_validate(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _require_checkpoint(p)
    metrics = run_validation(args.config, checkpoint_path=str(p["checkpoint"]))
    out = p["final_dir"] / "val_metrics.json"
    _write_json(out, metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_test(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _require_checkpoint(p)
    checkpoint = args.checkpoint or str(p["checkpoint"])
    metrics = run_test(config_path=args.config, checkpoint_path=checkpoint, out_dir=p["test_dir"])
    _copy_if_exists(p["test_dir"] / "test_metrics.json", p["final_dir"] / "test_metrics.json")
    _copy_if_exists(p["test_dir"] / "confusion_matrix.png", p["final_dir"] / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_zero_shot(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _require_checkpoint(p)
    checkpoint = args.checkpoint or str(p["checkpoint"])
    build_family_prototypes(
        config_path=args.config,
        checkpoint_path=checkpoint,
        manifest_path=str(p["train_manifest"]),
        output_path=str(p["prototype_bank"]),
    )
    metrics = evaluate_zero_shot(
        config_path=args.config,
        checkpoint_path=checkpoint,
        manifest_path=str(p["test_manifest"]),
        prototype_path=str(p["prototype_bank"]),
        out_dir=p["zero_shot_dir"],
    )
    _copy_if_exists(p["zero_shot_dir"] / "zero_shot_metrics.json", p["final_dir"] / "zero_shot_metrics.json")
    _copy_if_exists(p["prototype_bank"], p["final_dir"] / "prototypes.pt")
    print(json.dumps(metrics, indent=2))
    return metrics


def cmd_export(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _require_checkpoint(p)
    checkpoint = args.checkpoint or str(p["checkpoint"])
    exported = export_checkpoint(Path(checkpoint), Path(args.config), p["export_output"])
    _copy_if_exists(p["export_output"], p["final_dir"] / "xnerf_local_inference.pt")
    result = {"exported": str(exported)}
    print(json.dumps(result, indent=2))
    return result


def cmd_all(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {"config": args.config, "steps": {}}
    summary["steps"]["build_manifest"] = cmd_build_manifest(cfg, p, args)
    summary["steps"]["train"] = cmd_train(cfg, p, args)
    summary["steps"]["validate"] = cmd_validate(cfg, p, args)
    summary["steps"]["test"] = cmd_test(cfg, p, args)
    summary["steps"]["zero_shot"] = cmd_zero_shot(cfg, p, args)
    summary["steps"]["export"] = cmd_export(cfg, p, args)
    _write_json(p["final_dir"] / "summary.json", summary)
    print("\n=== Pipeline complete ===")
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def _require_checkpoint(p: dict[str, Path]) -> None:
    if not p["checkpoint"].exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {p['checkpoint']}\n"
            "Run 'train' first, or pass --checkpoint <path>."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--config", default="xnerf/configs/kaggle.yaml")


def _add_checkpoint_arg(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--checkpoint", default=None, help="Override checkpoint path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X-NERF++ Kaggle pipeline — run stages independently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # build-manifest
    p_bm = subs.add_parser("build-manifest", help="Extract archives and build manifests")
    _add_common_args(p_bm)
    p_bm.add_argument("--skip-extract", action="store_true")
    p_bm.add_argument("--rebuild-manifests", action="store_true")

    # train
    p_tr = subs.add_parser("train", help="Train the model")
    _add_common_args(p_tr)
    p_tr.add_argument("--force", action="store_true", help="Retrain even if train_done.json exists")

    # validate
    p_va = subs.add_parser("validate", help="Validate on the val split")
    _add_common_args(p_va)
    _add_checkpoint_arg(p_va)

    # test
    p_te = subs.add_parser("test", help="Evaluate on the test split")
    _add_common_args(p_te)
    _add_checkpoint_arg(p_te)

    # zero-shot
    p_zs = subs.add_parser("zero-shot", help="Build prototypes and evaluate zero-shot")
    _add_common_args(p_zs)
    _add_checkpoint_arg(p_zs)

    # export
    p_ex = subs.add_parser("export", help="Export local-inference checkpoint")
    _add_common_args(p_ex)
    _add_checkpoint_arg(p_ex)

    # all
    p_al = subs.add_parser("all", help="Run the full pipeline end-to-end")
    _add_common_args(p_al)
    p_al.add_argument("--skip-extract", action="store_true")
    p_al.add_argument("--rebuild-manifests", action="store_true")
    p_al.add_argument("--force", action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config)
    p = _resolve_paths(cfg)
    p["final_dir"].mkdir(parents=True, exist_ok=True)

    dispatch = {
        "build-manifest": cmd_build_manifest,
        "train": cmd_train,
        "validate": cmd_validate,
        "test": cmd_test,
        "zero-shot": cmd_zero_shot,
        "export": cmd_export,
        "all": cmd_all,
    }
    dispatch[args.command](cfg, p, args)


if __name__ == "__main__":
    main()
