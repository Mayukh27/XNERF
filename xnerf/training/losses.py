"""losses.py — training losses for X-NERF++.

Changes vs original:
  - family_label is now derived from the string "family" field via a per-batch
    label map, so family_ce always fires when the batch contains malware samples.
  - Added supervised contrastive loss (SupCon) on zero_shot_embedding so the
    embedding space is metric-trained and nearest-prototype retrieval works.
  - SupCon is only computed over malware samples (label==1); benign embeddings
    have no meaningful family identity and would pollute the metric space.
  - Temperature and loss weights are tunable via keyword args.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    family_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon loss (Khosla et al. 2020) over a single GPU batch.

    Args:
        embeddings: [B, D] float — already L2-normalised.
        family_ids: [B] long — integer family labels; samples with id == -1
                    are ignored (benign / unknown family).
        temperature: scalar, default 0.07.

    Returns:
        Scalar loss. Returns 0.0 if fewer than 2 distinct families are present.
    """
    device = embeddings.device

    # Only keep samples with a valid family label.
    valid = family_ids >= 0                              # [B] bool
    if valid.sum() < 2:
        return embeddings.new_tensor(0.0)

    emb = embeddings[valid]                              # [V, D]
    fam = family_ids[valid]                              # [V]

    # Similarity matrix.
    sim = emb @ emb.t() / temperature                   # [V, V]

    # Mask: same family, different sample.
    fam_eq = fam.unsqueeze(0) == fam.unsqueeze(1)       # [V, V]
    eye    = torch.eye(len(fam), dtype=torch.bool, device=device)
    pos_mask = fam_eq & ~eye                            # [V, V]

    # Skip if no positive pair exists.
    if pos_mask.sum() == 0:
        return embeddings.new_tensor(0.0)

    # Log-sum-exp over all negatives (excluding self).
    sim_masked = sim.masked_fill(eye, float("-inf"))     # remove diagonal
    log_denom  = torch.logsumexp(sim_masked, dim=1)     # [V]

    # Mean log-prob for every positive pair.
    pos_count = pos_mask.sum(dim=1).clamp_min(1)        # [V]
    log_prob  = sim.diagonal() - log_denom              # won't be used; compute per-pair
    # Recompute properly: for each anchor sum over positives.
    log_probs = sim - log_denom.unsqueeze(1)             # [V, V]
    loss_per_anchor = -(log_probs * pos_mask).sum(dim=1) / pos_count  # [V]

    # Only average over anchors that actually have a positive.
    has_pos = pos_mask.any(dim=1)
    return loss_per_anchor[has_pos].mean()


# ---------------------------------------------------------------------------
# Triplet margin loss (lightweight alternative / complement)
# ---------------------------------------------------------------------------

def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    family_ids: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Batch-hard triplet loss over valid (malware) samples."""
    valid = family_ids >= 0
    if valid.sum() < 2:
        return embeddings.new_tensor(0.0)

    emb = embeddings[valid]
    fam = family_ids[valid]

    dist = torch.cdist(emb, emb, p=2)                   # [V, V]
    fam_eq = fam.unsqueeze(0) == fam.unsqueeze(1)       # [V, V]
    eye    = torch.eye(len(fam), dtype=torch.bool, device=emb.device)

    # Hardest positive (furthest same-family sample).
    pos_dist = dist.masked_fill(~fam_eq | eye, 0.0).max(dim=1).values
    # Hardest negative (closest different-family sample).
    neg_dist = dist.masked_fill(fam_eq, float("inf")).min(dim=1).values

    has_pair = (fam_eq & ~eye).any(dim=1) & (~fam_eq).any(dim=1)
    if not has_pair.any():
        return embeddings.new_tensor(0.0)

    loss = F.relu(pos_dist[has_pair] - neg_dist[has_pair] + margin)
    return loss.mean()


# ---------------------------------------------------------------------------
# Family integer-label helper
# ---------------------------------------------------------------------------

def _family_ids_from_strings(
    families: list[str],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Convert string family names to per-batch integer ids.

    Benign samples (label == 0) or samples with family == 'unknown' get id -1
    so they are excluded from metric losses.

    Returns:
        LongTensor [B] with family integer ids (or -1).
    """
    unique = sorted({f for f, lbl in zip(families, labels.tolist()) if f != "unknown" and lbl == 1})
    name_to_id = {name: i for i, name in enumerate(unique)}
    ids = []
    for fam, lbl in zip(families, labels.tolist()):
        if lbl == 1 and fam != "unknown" and fam in name_to_id:
            ids.append(name_to_id[fam])
        else:
            ids.append(-1)
    return torch.tensor(ids, dtype=torch.long, device=labels.device)


# ---------------------------------------------------------------------------
# Main loss functions
# ---------------------------------------------------------------------------

def classification_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    family_weight: float = 0.5,
    arch_weight: float = 0.1,
    supcon_weight: float = 0.5,
    triplet_weight: float = 0.2,
    supcon_temperature: float = 0.07,
    triplet_margin: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Compute all training losses.

    Always-active losses:
        malware_ce   — binary malware classification.
        arch_adv     — gradient-reversed architecture adversary.
        field_smooth — MNEF field temporal smoothness.

    Metric losses (require ≥2 malware families in the batch):
        supcon       — supervised contrastive on zero_shot_embedding.
        triplet      — batch-hard triplet on zero_shot_embedding.

    Family CE (requires family_label in batch OR string family field):
        family_ce    — cross-entropy over known families.
    """
    losses: dict[str, torch.Tensor] = {
        "malware_ce": F.cross_entropy(outputs["malware_logits"], batch["label"]),
    }

    # --- Family CE ---
    # Prefer pre-computed integer labels; fall back to string-based mapping.
    if "family_label" in batch:
        losses["family_ce"] = (
            F.cross_entropy(outputs["family_logits"], batch["family_label"]) * family_weight
        )
    elif "family" in batch:
        family_ids = _family_ids_from_strings(batch["family"], batch["label"])
        valid = family_ids >= 0
        if valid.sum() > 0:
            # Only compute CE for samples that have a known family.
            logits_v = outputs["family_logits"][valid]
            # Re-index ids to 0…K-1 for valid samples.
            ids_v = family_ids[valid]
            _, ids_reindexed = torch.unique(ids_v, return_inverse=True)
            if logits_v.shape[1] >= ids_reindexed.max().item() + 1:
                losses["family_ce"] = (
                    F.cross_entropy(logits_v, ids_reindexed) * family_weight
                )

    # --- Architecture adversary ---
    losses["arch_adv"] = (
        F.cross_entropy(outputs["arch_logits"], batch["arch_id"]) * arch_weight
    )

    # --- Field smoothness ---
    if outputs["field"].shape[1] > 1:
        losses["field_smooth"] = (
            (outputs["field"][:, 1:] - outputs["field"][:, :-1]).pow(2).mean() * 0.01
        )

    # --- Metric losses on zero_shot_embedding ---
    if "family" in batch:
        family_ids = _family_ids_from_strings(batch["family"], batch["label"])
        emb_norm = F.normalize(outputs["zero_shot_embedding"].float(), dim=-1)

        sc = supervised_contrastive_loss(emb_norm, family_ids, temperature=supcon_temperature)
        if sc > 0:
            losses["supcon"] = sc * supcon_weight

        tr = batch_hard_triplet_loss(emb_norm, family_ids, margin=triplet_margin)
        if tr > 0:
            losses["triplet"] = tr * triplet_weight

    return losses


def total_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(losses.values())