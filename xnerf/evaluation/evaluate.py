from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from xnerf.preprocessing.ontology import ARCH_TO_ID


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for saving evaluation plots") from exc
    return plt


def _require_tsne():
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn TSNE support is required for embedding plots") from exc
    return TSNE


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, arch_true: np.ndarray | None = None, arch_pred: np.ndarray | None = None, zero_shot_mask: np.ndarray | None = None) -> dict:
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob.shape[1] == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics["roc_auc"] = None
    if zero_shot_mask is not None and zero_shot_mask.any():
        metrics["zero_shot_accuracy"] = accuracy_score(y_true[zero_shot_mask], y_pred[zero_shot_mask])
    if arch_true is not None:
        arch_true_arr = np.asarray(arch_true)
        mask = arch_true_arr != ARCH_TO_ID["unknown"]
        if mask.any():
            arch_malware_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            metrics["architecture_malware_accuracy"] = arch_malware_accuracy
            metrics["cross_architecture_accuracy"] = arch_malware_accuracy
            id_to_arch = {idx: name for name, idx in ARCH_TO_ID.items()}
            per_architecture_accuracy = {}
            for arch_id in sorted(set(arch_true_arr[mask].tolist())):
                arch_mask = arch_true_arr == arch_id
                per_architecture_accuracy[id_to_arch.get(int(arch_id), f"arch_{int(arch_id)}")] = accuracy_score(y_true[arch_mask], y_pred[arch_mask])
            metrics["per_architecture_accuracy"] = per_architecture_accuracy
        else:
            metrics["architecture_malware_accuracy"] = None
            metrics["cross_architecture_accuracy"] = None
            metrics["per_architecture_accuracy"] = {}
    return metrics


def evaluate_family_predictions(family_true: np.ndarray, family_prob: np.ndarray) -> dict:
    mask = np.asarray(family_true) >= 0
    if not mask.any():
        return {"family_top1_accuracy": None, "family_top5_accuracy": None}
    y_true = np.asarray(family_true)[mask]
    y_prob = np.asarray(family_prob)[mask]
    y_pred = y_prob.argmax(axis=1)
    metrics = {"family_top1_accuracy": accuracy_score(y_true, y_pred)}
    labels = np.arange(y_prob.shape[1])
    k = min(5, y_prob.shape[1])
    try:
        from sklearn.metrics import top_k_accuracy_score

        metrics["family_top5_accuracy"] = top_k_accuracy_score(y_true, y_prob, k=k, labels=labels)
    except ValueError:
        metrics["family_top5_accuracy"] = None
    return metrics


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    plt = _require_matplotlib()
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5.6, 4.8), dpi=300)

    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    classes = ["Benign", "Malware"]

    plt.xticks([0, 1], classes, fontsize=11)
    plt.yticks([0, 1], classes, fontsize=11)

    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold")

    total = cm.sum()
    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = 100 * count / total
            plt.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if count > thresh else "black",
            )

    print("\nConfusion Matrix:")
    print(cm)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    
def save_tsne(embeddings: np.ndarray, labels: np.ndarray, out: Path) -> None:
    plt = _require_matplotlib()
    TSNE = _require_tsne()
    coords = TSNE(n_components=2, perplexity=min(30, max(2, len(labels) // 3)), init="pca", learning_rate="auto").fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def save_umap(embeddings: np.ndarray, labels: np.ndarray, out: Path) -> None:
    plt = _require_matplotlib()
    TSNE = _require_tsne()
    try:
        import umap
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    except Exception:
        coords = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(embeddings)
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help=".npz with y_true, y_prob, embeddings")
    parser.add_argument("--out", default="runs/eval", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = np.load(args.predictions)
    arch_true = data["arch_true"] if "arch_true" in data else None
    metrics = evaluate_predictions(data["y_true"], data["y_prob"], arch_true=arch_true)
    if "family_true" in data and "family_prob" in data:
        metrics.update(evaluate_family_predictions(data["family_true"], data["family_prob"]))
    (args.out / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(data["y_true"], data["y_prob"].argmax(axis=1), args.out / "confusion_matrix.png")
    if "embeddings" in data:
        save_tsne(data["embeddings"], data["y_true"], args.out / "tsne.png")
        save_umap(data["embeddings"], data["y_true"], args.out / "umap.png")


if __name__ == "__main__":
    main()
