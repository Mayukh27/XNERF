from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.evaluate import evaluate_predictions
from xnerf.evaluation.test_after_training import _safe_family_metrics, load_model, run_inference
from xnerf.training.train import run_training
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config


ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full X-NERF",
        "model": {},
    },
    "no_binary": {
        "label": "w/o Binary Encoder",
        "model": {"disabled_modalities": ["binary"]},
    },
    "no_api": {
        "label": "w/o API Encoder",
        "model": {"disabled_modalities": ["api"]},
    },
    "no_isr": {
        "label": "w/o ISR Encoder",
        "model": {"disabled_modalities": ["isr"]},
    },
    "no_cfg": {
        "label": "w/o CFG Encoder",
        "model": {"disabled_modalities": ["cfg"]},
    },
    "no_memory": {
        "label": "w/o Memory Encoder",
        "model": {"disabled_modalities": ["memory"]},
    },
    "no_network": {
        "label": "w/o Network Encoder",
        "model": {"disabled_modalities": ["network"]},
    },
    "no_sfs": {
        "label": "w/o SFS",
        "model": {"use_sfs": False},
    },
    "no_grl": {
        "label": "w/o GRL",
        "model": {"use_grl": False},
    },
    "no_mnef": {
        "label": "w/o MNEF (MLP head only)",
        "model": {"use_mnef": False},
    },
}


def _merge_disabled_modalities(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    base_disabled = list(base.get("disabled_modalities", []))
    override_disabled = list(override.get("disabled_modalities", []))
    if base_disabled or override_disabled:
        merged["disabled_modalities"] = sorted({*base_disabled, *override_disabled})
    for key, value in override.items():
        if key != "disabled_modalities":
            merged[key] = value
    return merged


def build_variant_config(
    base_cfg: dict[str, Any],
    variant_name: str,
    out_dir: Path,
    num_workers: int | None = 0,
    batch_size: int | None = 1,
    grad_accum: int | None = None,
    test_manifest: str | None = None,
) -> dict[str, Any]:
    if variant_name not in ABLATIONS:
        known = ", ".join(sorted(ABLATIONS))
        raise ValueError(f"unknown ablation variant {variant_name!r}; choose one of: {known}")

    cfg = copy.deepcopy(base_cfg)
    variant = ABLATIONS[variant_name]
    cfg.setdefault("model", {})
    cfg["model"] = _merge_disabled_modalities(cfg["model"], variant.get("model", {}))
    if test_manifest:
        cfg.setdefault("data", {})
        cfg["data"]["test_manifest"] = test_manifest

    cfg.setdefault("training", {})
    if variant_name != "full":
        cfg["training"]["checkpoint_dir"] = str(out_dir / variant_name / "checkpoints")
    if variant_name != "full" and num_workers is not None:
        cfg["training"]["num_workers"] = int(num_workers)
    if variant_name != "full" and batch_size is not None:
        old_batch_size = int(cfg["training"].get("batch_size", batch_size))
        old_grad_accum = int(cfg["training"].get("grad_accum", 1))
        cfg["training"]["batch_size"] = int(batch_size)
        if grad_accum is not None:
            cfg["training"]["grad_accum"] = int(grad_accum)
        elif int(batch_size) > 0:
            effective_batch = max(1, old_batch_size * old_grad_accum)
            cfg["training"]["grad_accum"] = max(1, round(effective_batch / int(batch_size)))
    cfg.setdefault("outputs", {})
    cfg["outputs"]["ablation_dir"] = str(out_dir / variant_name)
    return cfg


def write_variant_config(cfg: dict[str, Any], variant_dir: Path) -> Path:
    variant_dir.mkdir(parents=True, exist_ok=True)
    config_path = variant_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def _checkpoint_from_cfg(cfg: dict[str, Any], explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit)
    export_checkpoint = cfg.get("export", {}).get("checkpoint")
    if export_checkpoint:
        return Path(export_checkpoint)
    return Path(cfg["training"]["checkpoint_dir"]) / "best.pt"


@torch.no_grad()
def evaluate_ablation(config_path: str | Path, checkpoint_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    test_manifest = cfg["data"].get("test_manifest")
    if not test_manifest:
        raise ValueError("config data.test_manifest is required for ablation evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint_path), cfg, device)
    ds = MalwareManifestDataset(test_manifest, require_cache=True)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        collate_fn=collate_dicts,
    )

    malware_probs, malware_labels = [], []
    family_probs, family_labels = [], []
    embeddings, arch_labels, sample_ids = [], [], []
    for batch in tqdm(loader, desc="ablation test"):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        malware_probs.append(torch.softmax(outputs["malware_logits"], dim=-1).cpu().numpy())
        malware_labels.append(batch["label"].cpu().numpy())
        family_probs.append(torch.softmax(outputs["family_logits"], dim=-1).cpu().numpy())
        family_labels.append(batch["family_label"].cpu().numpy())
        embeddings.append(outputs["zero_shot_embedding"].cpu().numpy())
        arch_labels.append(batch["arch_id"].cpu().numpy())
        sample_ids.extend([str(x) for x in batch.get("sample_id", batch.get("sha256", []))])

    y_prob = np.concatenate(malware_probs)
    y_true = np.concatenate(malware_labels)
    family_y_prob = np.concatenate(family_probs)
    family_y_true = np.concatenate(family_labels)
    emb = np.concatenate(embeddings)
    arch_true = np.concatenate(arch_labels)

    metrics = evaluate_predictions(y_true, y_prob, arch_true=arch_true)
    metrics.update(_safe_family_metrics(family_y_true, family_y_prob))
    y_pred = y_prob.argmax(axis=1)
    diagnostics = {
        "checkpoint": str(checkpoint_path),
        "test_manifest": str(test_manifest),
        "num_test_samples": int(len(y_true)),
        "first_sample_ids": sample_ids[:10],
        "prediction_counts": {str(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_path / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_path / "predictions.npz",
        y_true=y_true,
        y_prob=y_prob,
        family_true=family_y_true,
        family_prob=family_y_prob,
        embeddings=emb,
        arch_true=arch_true,
        sample_ids=np.array(sample_ids),
    )
    return metrics


def run_ablation_suite(
    config_path: str | Path,
    out_dir: str | Path,
    variants: list[str],
    skip_train: bool = False,
    num_workers: int | None = 0,
    batch_size: int | None = 1,
    grad_accum: int | None = None,
    official_checkpoint: str | Path | None = None,
    test_manifest: str | None = None,
) -> list[dict[str, Any]]:
    base_cfg = load_config(config_path)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results = []

    for variant_name in variants:
        variant_dir = root / variant_name
        cfg = build_variant_config(
            base_cfg,
            variant_name,
            root,
            num_workers=num_workers,
            batch_size=batch_size,
            grad_accum=grad_accum,
            test_manifest=test_manifest,
        )
        variant_config = write_variant_config(cfg, variant_dir)
        checkpoint = _checkpoint_from_cfg(base_cfg, official_checkpoint) if variant_name == "full" else Path(cfg["training"]["checkpoint_dir"]) / "best.pt"
        if not skip_train and variant_name != "full":
            train_metrics = run_training(str(variant_config))
        else:
            train_metrics = {}
            if not checkpoint.exists():
                raise FileNotFoundError(f"--skip-train requested but checkpoint is missing: {checkpoint}")

        eval_dir = variant_dir / "test"
        if variant_name == "full":
            test_metrics = run_inference(
                cfg=cfg,
                checkpoint_path=checkpoint,
                manifest_path=cfg["data"]["test_manifest"],
                out_dir=eval_dir,
                predictions_name="predictions.npz",
                metrics_name="metrics.json",
                save_plots=False,
                desc="ablation full test",
            )
        else:
            test_metrics = evaluate_ablation(variant_config, checkpoint, eval_dir)
        row = {
            "variant": variant_name,
            "label": ABLATIONS[variant_name]["label"],
            "checkpoint": str(checkpoint),
            "train": train_metrics,
            "test": test_metrics,
        }
        results.append(row)
        (variant_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    write_summary(results, root)
    return results


def _pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}"


def _delta_pct(value: Any, baseline: Any) -> str:
    if value is None or baseline is None:
        return "N/A"
    return f"{(float(value) - float(baseline)) * 100:+.2f}"


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = out_dir / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "label",
                "detection_accuracy",
                "detection_precision",
                "detection_recall",
                "detection_f1",
                "roc_auc",
                "family_top1_accuracy",
                "family_top5_accuracy",
                "cross_architecture_accuracy",
                "detection_accuracy_pct",
                "detection_precision_pct",
                "detection_recall_pct",
                "detection_f1_pct",
                "roc_auc_pct",
                "family_top1_accuracy_pct",
                "family_top5_accuracy_pct",
                "cross_architecture_accuracy_pct",
            ],
        )
        writer.writeheader()
        for row in results:
            metrics = row["test"]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "label": row["label"],
                    "detection_accuracy": metrics.get("accuracy"),
                    "detection_precision": metrics.get("precision"),
                    "detection_recall": metrics.get("recall"),
                    "detection_f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc"),
                    "family_top1_accuracy": metrics.get("family_top1_accuracy"),
                    "family_top5_accuracy": metrics.get("family_top5_accuracy"),
                    "cross_architecture_accuracy": metrics.get("cross_architecture_accuracy"),
                    "detection_accuracy_pct": _pct(metrics.get("accuracy")),
                    "detection_precision_pct": _pct(metrics.get("precision")),
                    "detection_recall_pct": _pct(metrics.get("recall")),
                    "detection_f1_pct": _pct(metrics.get("f1")),
                    "roc_auc_pct": _pct(metrics.get("roc_auc")),
                    "family_top1_accuracy_pct": _pct(metrics.get("family_top1_accuracy")),
                    "family_top5_accuracy_pct": _pct(metrics.get("family_top5_accuracy")),
                    "cross_architecture_accuracy_pct": _pct(metrics.get("cross_architecture_accuracy")),
                }
            )

    baseline = next((row["test"] for row in results if row["variant"] == "full"), results[0]["test"] if results else {})
    lines = [
        "\\begin{table*}[t]",
        "\\caption{X-NERF Ablation Study. All values are measured on the held-out test split; higher is better.}",
        "\\label{tab:ablation}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{$\\Delta$F1} & \\textbf{AUC} & \\textbf{Fam@1} & \\textbf{Fam@5} & \\textbf{Cross-Arch.} \\\\",
        "\\midrule",
    ]
    for row in results:
        metrics = row["test"]
        label = row["label"].replace("&", "\\&")
        lines.append(
            f"{label} & "
            f"{_pct(metrics.get('accuracy'))} & "
            f"{_pct(metrics.get('precision'))} & "
            f"{_pct(metrics.get('recall'))} & "
            f"{_pct(metrics.get('f1'))} & "
            f"{_delta_pct(metrics.get('f1'), baseline.get('f1'))} & "
            f"{_pct(metrics.get('roc_auc'))} & "
            f"{_pct(metrics.get('family_top1_accuracy'))} & "
            f"{_pct(metrics.get('family_top5_accuracy'))} & "
            f"{_pct(metrics.get('cross_architecture_accuracy'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (out_dir / "ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _read_json_if_exists(*paths: Path) -> dict[str, Any]:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _npz_path(directory: Path, *names: str) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def compare_official_and_ablation_full(official_dir: str | Path, ablation_dir: str | Path) -> dict[str, Any]:
    official = Path(official_dir)
    ablation_full = Path(ablation_dir) / "full" / "test"
    official_metrics = _read_json_if_exists(official / "test_metrics.json", official / "metrics.json")
    ablation_metrics = _read_json_if_exists(ablation_full / "metrics.json", ablation_full / "test_metrics.json")
    official_diag = _read_json_if_exists(official / "diagnostics.json")
    ablation_diag = _read_json_if_exists(ablation_full / "diagnostics.json")

    official_npz_path = _npz_path(official, "test_predictions.npz", "predictions.npz")
    ablation_npz_path = _npz_path(ablation_full, "predictions.npz", "test_predictions.npz")
    comparison: dict[str, Any] = {
        "official_dir": str(official),
        "ablation_full_dir": str(ablation_full),
        "official_metrics": official_metrics,
        "ablation_full_metrics": ablation_metrics,
        "official_diagnostics": official_diag,
        "ablation_full_diagnostics": ablation_diag,
    }

    if official_npz_path and ablation_npz_path:
        official_npz = np.load(official_npz_path, allow_pickle=True)
        ablation_npz = np.load(ablation_npz_path, allow_pickle=True)
        official_y_prob = official_npz["y_prob"]
        ablation_y_prob = ablation_npz["y_prob"]
        official_y_true = official_npz["y_true"]
        ablation_y_true = ablation_npz["y_true"]
        official_pred = official_y_prob.argmax(axis=1)
        ablation_pred = ablation_y_prob.argmax(axis=1)
        sample_ids_equal = None
        if "sample_ids" in official_npz and "sample_ids" in ablation_npz:
            sample_ids_equal = bool(np.array_equal(official_npz["sample_ids"], ablation_npz["sample_ids"]))
        comparison.update(
            {
                "official_predictions": str(official_npz_path),
                "ablation_predictions": str(ablation_npz_path),
                "same_num_samples": int(len(official_y_true)) == int(len(ablation_y_true)),
                "same_labels": bool(np.array_equal(official_y_true, ablation_y_true)),
                "same_sample_ids": sample_ids_equal,
                "same_predicted_labels": bool(np.array_equal(official_pred, ablation_pred)),
                "max_probability_abs_diff": float(np.max(np.abs(official_y_prob - ablation_y_prob))) if official_y_prob.shape == ablation_y_prob.shape else None,
                "official_prediction_counts": {str(k): int(v) for k, v in zip(*np.unique(official_pred, return_counts=True))},
                "ablation_prediction_counts": {str(k): int(v) for k, v in zip(*np.unique(ablation_pred, return_counts=True))},
                "official_confusion_matrix": confusion_matrix(official_y_true, official_pred).tolist(),
                "ablation_confusion_matrix": confusion_matrix(ablation_y_true, ablation_pred).tolist(),
            }
        )

    Path(ablation_dir).mkdir(parents=True, exist_ok=True)
    out = Path(ablation_dir) / "full_vs_official_comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run X-NERF ablation studies")
    parser.add_argument("--config", default="config_publication_v2_50k.yaml")
    parser.add_argument("--out", default="runs/ablation")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "full",
            "no_binary",
            "no_api",
            "no_cfg",
            "no_memory",
            "no_network",
            "no_isr",
            "no_sfs",
            "no_grl",
            "no_mnef",
        ],
        choices=sorted(ABLATIONS),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Evaluate existing ablation checkpoints instead of training them.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for generated ablation configs. Keep 0 on Windows to avoid shared-memory error 1455.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size for generated ablation configs. Default 1 is intended for 4 GB GPUs.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation override. Defaults to preserving the config's effective batch size.",
    )
    parser.add_argument(
        "--official-checkpoint",
        default=None,
        help="Official full-model checkpoint. Defaults to config training.checkpoint_dir/best.pt.",
    )
    parser.add_argument(
        "--test-manifest",
        default=None,
        help="Override test manifest for every ablation, for example data/processed/test_manifest_clean.jsonl.",
    )
    parser.add_argument(
        "--official-out",
        default=None,
        help="Directory containing official test_metrics.json/test_predictions.npz. If set, compare it against runs\\ablation\\full\\test.",
    )
    args = parser.parse_args()
    results = run_ablation_suite(
        args.config,
        args.out,
        args.variants,
        skip_train=args.skip_train,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        official_checkpoint=args.official_checkpoint,
        test_manifest=args.test_manifest,
    )
    if args.official_out:
        compare_official_and_ablation_full(args.official_out, args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
