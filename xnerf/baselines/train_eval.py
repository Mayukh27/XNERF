from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from evaluation.metrics import classification_metrics
from xnerf.baselines.models import CNNMalware, GNNMalware, HYDRA, MalBERT
from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.preprocessing.ontology import ARCH_TO_ID
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.io import read_jsonl


BASELINE_CHOICES = ("ember-rf", "malconv", "malbert", "hydra", "gnn-malware")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_tensor_feature(path: str | Path, max_dim: int) -> np.ndarray:
    tensor = torch.load(path, map_location="cpu").float().flatten()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.zeros(max_dim, dtype=torch.float32)
    out[: min(max_dim, tensor.numel())] = tensor[:max_dim]
    return out.numpy()


def _numeric_manifest_features(row: dict[str, Any]) -> list[float]:
    keys = (
        "feature_dim",
        "api_call_count",
        "network_event_count",
        "row_index",
    )
    vals: list[float] = []
    for key in keys:
        try:
            vals.append(float(row.get(key, 0) or 0))
        except (TypeError, ValueError):
            vals.append(0.0)
    return vals


def _build_rf_matrix(manifest: str | Path, max_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rows = read_jsonl(manifest)
    x_rows: list[np.ndarray] = []
    labels: list[int] = []
    arch_ids: list[int] = []
    skipped_missing_features = 0
    used_feature_cache = 0

    for row in rows:
        feature_path = row.get("feature_path")
        if feature_path and Path(feature_path).exists():
            feature_vec = _load_tensor_feature(feature_path, max_features)
            used_feature_cache += 1
        else:
            numeric = np.asarray(_numeric_manifest_features(row), dtype=np.float32)
            if not np.any(numeric):
                skipped_missing_features += 1
                continue
            feature_vec = np.zeros(max_features, dtype=np.float32)
            feature_vec[: min(max_features, numeric.size)] = numeric[:max_features]

        x_rows.append(feature_vec)
        labels.append(int(row.get("label", 0)))
        arch = str(row.get("arch", "unknown")).strip().lower()
        arch_ids.append(ARCH_TO_ID.get(arch, ARCH_TO_ID["unknown"]))

    if not x_rows:
        raise RuntimeError(
            f"No usable tabular/static rows found in {manifest}. "
            "Generate feature caches first or choose a neural baseline."
        )

    stats = {
        "manifest": str(manifest),
        "rows_total": len(rows),
        "rows_used": len(x_rows),
        "rows_from_feature_cache": used_feature_cache,
        "rows_skipped_missing_features": skipped_missing_features,
        "max_features": max_features,
    }
    return np.vstack(x_rows), np.asarray(labels), np.asarray(arch_ids), stats


def train_ember_rf(args: argparse.Namespace) -> dict[str, Any]:
    x_train, y_train, _, train_stats = _build_rf_matrix(args.train_manifest, args.max_features)
    x_test, y_test, arch_test, test_stats = _build_rf_matrix(args.test_manifest, args.max_features)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)
    if y_score.shape[1] == 1:
        only_class = int(model.classes_[0])
        y_prob = np.zeros((len(y_test), 2), dtype=np.float32)
        y_prob[:, only_class] = 1.0
    else:
        y_prob = y_score

    metrics = classification_metrics(y_test, y_prob, arch_true=arch_test)
    payload = {
        "method": "EMBER RF",
        "baseline": "ember-rf",
        "seed": args.seed,
        "train_manifest": str(args.train_manifest),
        "test_manifest": str(args.test_manifest),
        "train_rows": train_stats,
        "test_rows": test_stats,
        **metrics,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "metrics.json", payload)
    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    return payload


def _row_has_modality(row: dict[str, Any], baseline: str) -> bool:
    if baseline == "malconv":
        return row.get("data_type") not in {"feature_csv", "feature_parquet", "api_sequence_csv", "api_sequence_txt"}
    if baseline == "malbert":
        return bool(row.get("api_ids"))
    if baseline == "hydra":
        return bool(row.get("api_ids")) and row.get("data_type") not in {"feature_csv", "feature_parquet"}
    if baseline == "gnn-malware":
        return str(row.get("path", "")).lower().endswith(".edgelist")
    return True


def _filtered_dataset(manifest: str | Path, baseline: str, require_cache: bool) -> tuple[Subset, dict[str, Any]]:
    rows = read_jsonl(manifest)
    indices = [idx for idx, row in enumerate(rows) if _row_has_modality(row, baseline)]
    if not indices:
        raise RuntimeError(f"No rows in {manifest} have the required modality for {baseline}.")
    dataset = MalwareManifestDataset(manifest, require_cache=require_cache)
    stats = {
        "manifest": str(manifest),
        "rows_total": len(rows),
        "rows_used": len(indices),
        "rows_skipped_modality": len(rows) - len(indices),
    }
    return Subset(dataset, indices), stats


def _make_model(baseline: str) -> torch.nn.Module:
    if baseline == "malconv":
        return CNNMalware(num_classes=2)
    if baseline == "malbert":
        return MalBERT(num_classes=2)
    if baseline == "hydra":
        return HYDRA(num_classes=2)
    if baseline == "gnn-malware":
        return GNNMalware(node_dim=4, num_classes=2)
    raise ValueError(f"unsupported neural baseline: {baseline}")


def _forward_baseline(model: torch.nn.Module, baseline: str, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if baseline == "malconv":
        return model(batch["binary_image"]), batch["label"], batch["arch_id"]
    if baseline == "malbert":
        return model(batch["api_ids"]), batch["label"], batch["arch_id"]
    if baseline == "hydra":
        return model(batch["binary_image"], batch["api_ids"]), batch["label"], batch["arch_id"]
    if baseline == "gnn-malware":
        logits = model(batch["graph_x"], batch["graph_edge_index"], batch["graph_batch"])
        sample_ids = batch["graph_sample_ids"].to(batch["label"].device)
        return logits, batch["label"].index_select(0, sample_ids), batch["arch_id"].index_select(0, sample_ids)
    raise ValueError(f"unsupported neural baseline: {baseline}")


def _evaluate_neural(
    model: torch.nn.Module,
    loader: DataLoader,
    baseline: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    arch_true: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{baseline} test"):
            batch = move_to_device(batch, device)
            logits, labels, arch_ids = _forward_baseline(model, baseline, batch)
            probs = torch.softmax(logits, dim=-1)
            y_true.append(labels.detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())
            arch_true.append(arch_ids.detach().cpu().numpy())
    return classification_metrics(np.concatenate(y_true), np.concatenate(y_prob), arch_true=np.concatenate(arch_true))


def train_neural(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds, train_stats = _filtered_dataset(args.train_manifest, args.baseline, args.require_cache)
    val_ds, val_stats = _filtered_dataset(args.val_manifest or args.test_manifest, args.baseline, args.require_cache)
    test_ds, test_stats = _filtered_dataset(args.test_manifest, args.baseline, args.require_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_dicts)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_dicts)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_dicts)

    model = _make_model(args.baseline).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"{args.baseline} epoch {epoch} train")):
            if args.debug_max_batches and batch_idx >= args.debug_max_batches:
                break
            batch = move_to_device(batch, device)
            logits, labels, _ = _forward_baseline(model, args.baseline, batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"{args.baseline} epoch {epoch} val")):
                if args.debug_max_batches and batch_idx >= args.debug_max_batches:
                    break
                batch = move_to_device(batch, device)
                logits, labels, _ = _forward_baseline(model, args.baseline, batch)
                val_loss += float(torch.nn.functional.cross_entropy(logits, labels).detach().cpu())
                val_batches += 1

        avg_train = train_loss / max(1, train_batches)
        avg_val = val_loss / max(1, val_batches)
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = _evaluate_neural(model, test_loader, args.baseline, device)
    payload = {
        "method": args.baseline,
        "baseline": args.baseline,
        "seed": args.seed,
        "device": str(device),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest or args.test_manifest),
        "test_manifest": str(args.test_manifest),
        "train_rows": train_stats,
        "val_rows": val_stats,
        "test_rows": test_stats,
        "best_val_loss": best_val_loss,
        "history": history,
        **metrics,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "metrics.json", payload)
    torch.save({"model": model.state_dict(), "baseline": args.baseline, "metrics": payload}, out_dir / "model.pt")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and test publication baselines on X-NERF++ manifests")
    parser.add_argument("--baseline", choices=BASELINE_CHOICES, required=True)
    parser.add_argument("--train-manifest", default="data/processed/train_publication_v2_50k.jsonl")
    parser.add_argument("--val-manifest", default="data/processed/val_publication_v2_50k.jsonl")
    parser.add_argument("--test-manifest", default="data/processed/test_publication_v2_50k.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--require-cache", action="store_true")

    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--debug-max-batches", type=int, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = str(Path("runs") / "baselines" / args.baseline / f"seed_{args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.baseline == "ember-rf":
        payload = train_ember_rf(args)
    else:
        payload = train_neural(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()

