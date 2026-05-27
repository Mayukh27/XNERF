"""local_run.py — split pipeline for local development / inference.

Subcommands (run in order):
    build-manifest   Scan data/raw and build train/val/test manifests.
    train            Train the model; writes checkpoints/best.pt + runs/train_done.json.
    validate         Run the validation loop on the val split.
    test             Evaluate best.pt on the test split.
    zero-shot        Build prototype bank then evaluate zero-shot accuracy.
    export           Package best.pt into a local-inference checkpoint.

    all              Run all of the above in sequence.

Examples:
    python -m xnerf.pipeline.local_run build-manifest --config config.yaml
    python -m xnerf.pipeline.local_run train           --config config.yaml
    python -m xnerf.pipeline.local_run validate        --config config.yaml
    python -m xnerf.pipeline.local_run test            --config config.yaml
    python -m xnerf.pipeline.local_run zero-shot       --config config.yaml
    python -m xnerf.pipeline.local_run export          --config config.yaml

    # Use a specific checkpoint for test/zero-shot/export without retraining:
    python -m xnerf.pipeline.local_run test --checkpoint checkpoints/epoch3.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from xnerf.datasets.build_dataset import build_manifest
from xnerf.deployment.export_checkpoint import export_checkpoint
from xnerf.evaluation.test_after_training import run_test
from xnerf.training.train import run_training, run_validation
from xnerf.utils.config import load_config
from xnerf.zero_shot.build_prototypes import build_family_prototypes
from xnerf.zero_shot.evaluate_zero_shot import evaluate_zero_shot

_DEFAULT_CONFIG = "config.yaml"


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
    data_root = Path(cfg["data"]["root"])
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    runs_dir = Path("runs")
    zero_shot_dir = runs_dir / "zero_shot"
    final_dir = runs_dir / "output"

    # Local config may not have export/outputs sections — provide sensible defaults.
    export_cfg = cfg.get("export", {})
    outputs_cfg = cfg.get("outputs", {})

    return {
        "data_root": data_root,
        "full_manifest": Path(cfg["data"].get("full_manifest", data_root / "processed" / "manifest.jsonl")),
        "train_manifest": Path(cfg["data"].get("train_manifest", data_root / "processed" / "train_manifest.jsonl")),
        "val_manifest": Path(cfg["data"].get("val_manifest", data_root / "processed" / "val_manifest.jsonl")),
        "test_manifest": Path(cfg["data"].get("test_manifest", data_root / "processed" / "test_manifest.jsonl")),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint": Path(export_cfg.get("checkpoint", checkpoint_dir / "best.pt")),
        "export_output": Path(export_cfg.get("output", "models/xnerf_local_inference.pt")),
        "train_metrics": Path(outputs_cfg.get("train_metrics", runs_dir / "train_metrics.json")),
        "test_dir": Path(outputs_cfg.get("test_dir", runs_dir / "test")),
        "zero_shot_dir": Path(outputs_cfg.get("zero_shot_dir", str(zero_shot_dir))),
        "prototype_bank": Path(outputs_cfg.get("prototype_bank", str(zero_shot_dir / "prototypes.pt"))),
        "final_dir": Path(outputs_cfg.get("final_dir", str(final_dir))),
        "train_done": checkpoint_dir / "train_done.json",
    }


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def cmd_build_manifest(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
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
        result = {"rebuilt": True}
    else:
        result = {"reused": True}

    result.update({
        "full": str(p["full_manifest"]),
        "train": str(p["train_manifest"]),
        "val": str(p["val_manifest"]),
        "test": str(p["test_manifest"]),
    })
    print(json.dumps(result, indent=2))
    return result


def cmd_train(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    if p["train_done"].exists() and not args.force:
        print(f"[train] Checkpoint already exists at {p['checkpoint']}.")
        print(f"        Delete {p['train_done']} or pass --force to retrain.")
        return _read_json(p["train_done"])

    metrics = run_training(args.config)

    sentinel = {"status": "done", "checkpoint": str(p["checkpoint"]), **metrics}
    _write_json(p["train_done"], sentinel)
    _write_json(p["train_metrics"], metrics)

    print(json.dumps(sentinel, indent=2))
    return sentinel


def cmd_validate(cfg: dict, p: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    _require_checkpoint(p)
    checkpoint = args.checkpoint or str(p["checkpoint"])
    metrics = run_validation(args.config, checkpoint_path=checkpoint)
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
    sub.add_argument("--config", default=_DEFAULT_CONFIG)


def _add_checkpoint_arg(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--checkpoint", default=None, help="Override checkpoint path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X-NERF++ local pipeline — run stages independently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p_bm = subs.add_parser("build-manifest", help="Build train/val/test manifests")
    _add_common_args(p_bm)
    p_bm.add_argument("--rebuild-manifests", action="store_true")

    p_tr = subs.add_parser("train", help="Train the model")
    _add_common_args(p_tr)
    p_tr.add_argument("--force", action="store_true", help="Retrain even if train_done.json exists")

    p_va = subs.add_parser("validate", help="Validate on the val split")
    _add_common_args(p_va)
    _add_checkpoint_arg(p_va)

    p_te = subs.add_parser("test", help="Evaluate on the test split")
    _add_common_args(p_te)
    _add_checkpoint_arg(p_te)

    p_zs = subs.add_parser("zero-shot", help="Build prototypes and evaluate zero-shot")
    _add_common_args(p_zs)
    _add_checkpoint_arg(p_zs)

    p_ex = subs.add_parser("export", help="Export local-inference checkpoint")
    _add_common_args(p_ex)
    _add_checkpoint_arg(p_ex)

    p_al = subs.add_parser("all", help="Run the full pipeline end-to-end")
    _add_common_args(p_al)
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
