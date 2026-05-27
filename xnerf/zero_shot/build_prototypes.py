"""build_prototypes.py — build a zero-shot prototype bank from trained embeddings.

Changes vs original:
  - Skips benign samples (label == 0); they have no family identity and
    pollute the bank with off-manifold embeddings.
  - Skips samples whose family field is 'unknown'.
  - L2-normalises every per-sample embedding before averaging so the mean
    stays on the unit hypersphere (matches cosine-similarity retrieval).
  - Reports per-family sample counts in metadata for debugging.
  - Warns when a family has very few support samples (<5).
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.test_after_training import load_model
from xnerf.utils.base import move_to_device
from xnerf.utils.config import load_config
from xnerf.zero_shot.prototypes import save_prototype_bank

_MIN_SUPPORT = 5  # warn if a family has fewer samples than this


@torch.no_grad()
def build_family_prototypes(
    config_path: str,
    checkpoint_path: str,
    manifest_path: str,
    output_path: str,
) -> Path:
    """Build and save a prototype bank.

    Only malware samples (label == 1) with a known family name are used.
    Each embedding is L2-normalised before accumulation so the mean prototype
    is meaningful under cosine similarity.

    Returns:
        Path to the saved prototype bank.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint_path), cfg, device)

    ds = MalwareManifestDataset(manifest_path)
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
    )

    # Accumulate L2-normalised embeddings per family.
    vectors: dict[str, list[torch.Tensor]] = defaultdict(list)
    skipped_benign = 0
    skipped_unknown = 0

    for batch in tqdm(loader, desc="build zero-shot prototypes"):
        families = [str(f) for f in batch.get("family", [])]
        labels   = batch["label"].tolist()

        # Determine which indices to keep (malware, known family).
        keep_idx = []
        for i, (fam, lbl) in enumerate(zip(families, labels)):
            if lbl == 0:
                skipped_benign += 1
            elif fam in ("unknown", "", "none", "None"):
                skipped_unknown += 1
            else:
                keep_idx.append(i)

        if not keep_idx:
            continue

        batch_gpu = move_to_device(batch, device)

        # Subset the batch tensors to keep_idx only.
        keep_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
        batch_sub = {
            k: v.index_select(0, keep_t) if torch.is_tensor(v) and v.shape[0] == len(families) else v
            for k, v in batch_gpu.items()
        }

        outputs = model(batch_sub)
        # L2-normalise each embedding before accumulating.
        emb = F.normalize(outputs["zero_shot_embedding"].detach().cpu().float(), dim=-1)

        for j, i in enumerate(keep_idx):
            vectors[families[i]].append(emb[j])

    if not vectors:
        raise RuntimeError(
            "No valid malware samples found in manifest. "
            "Check that label==1 samples exist and family fields are set."
        )

    # Log skipped counts.
    print(f"[build_prototypes] skipped {skipped_benign} benign, {skipped_unknown} unknown-family samples")
    print(f"[build_prototypes] building prototypes for {len(vectors)} families")

    # Compute mean prototype per family and warn on low support.
    labels_sorted = sorted(vectors)
    support_counts: dict[str, int] = {}
    prototype_list: list[torch.Tensor] = []

    for label in labels_sorted:
        vecs = vectors[label]
        support_counts[label] = len(vecs)
        if len(vecs) < _MIN_SUPPORT:
            warnings.warn(
                f"Family '{label}' has only {len(vecs)} support sample(s). "
                "Prototype quality may be poor.",
                stacklevel=2,
            )
        # Stack and mean — all vecs are already L2-normalised.
        proto = torch.stack(vecs).mean(dim=0)
        # Re-normalise the mean (it drifts off the sphere).
        proto = F.normalize(proto, dim=-1)
        prototype_list.append(proto)

    prototypes = torch.stack(prototype_list)  # [K, D]

    metadata = {
        "source_manifest": manifest_path,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "num_families": len(labels_sorted),
        "skipped_benign": skipped_benign,
        "skipped_unknown_family": skipped_unknown,
        "support_counts": support_counts,
    }

    path = save_prototype_bank(output_path, prototypes, labels_sorted, metadata=metadata)
    print(json.dumps({"prototype_bank": str(path), "families": len(labels_sorted)}, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build zero-shot prototype bank from trained X-NERF++ embeddings"
    )
    parser.add_argument("--config",     default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--checkpoint", default="/kaggle/working/checkpoints/best.pt")
    parser.add_argument("--manifest",   default="/kaggle/working/data/processed/train_manifest.jsonl")
    parser.add_argument("--output",     default="/kaggle/working/runs/zero_shot/prototypes.pt")
    args = parser.parse_args()
    build_family_prototypes(args.config, args.checkpoint, args.manifest, args.output)


if __name__ == "__main__":
    main()