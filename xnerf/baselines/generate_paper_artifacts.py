from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "family_top1_accuracy", "family_top5_accuracy", "cross_architecture_accuracy")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "family_top1_accuracy": "Family Top-1",
    "family_top5_accuracy": "Family Top-5",
    "cross_architecture_accuracy": "Cross-Arch Accuracy",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_name(metrics: dict[str, Any]) -> str:
    return str(metrics.get("method") or metrics.get("baseline") or "unknown")


def _collect_runs(root: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    metric_files = sorted(root.glob("*/seed_*/metrics.json"))
    if not metric_files:
        metric_files = sorted(root.glob("**/metrics.json"))
    if not metric_files:
        raise RuntimeError(f"No metrics.json files found under {root}")

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for metrics_path in metric_files:
        metrics = _read_json(metrics_path)
        grouped.setdefault(_method_name(metrics), []).append((metrics_path, metrics))
    return grouped


def _metric_stats(runs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = [float(run[key]) for run in runs if run.get(key) is not None]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _save_metric_barplots(grouped: dict[str, list[tuple[Path, dict[str, Any]]]], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(grouped)
    run_groups = [[metrics for _, metrics in grouped[method]] for method in methods]
    saved: dict[str, str] = {}

    for metric in METRICS:
        means = []
        errors = []
        for runs in run_groups:
            avg, err = _metric_stats(runs, metric)
            means.append(avg * 100.0)
            errors.append(err * 100.0)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=errors, capsize=4, color="#2F6B8A", edgecolor="#1D3340", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
        ax.set_title(f"Baseline Comparison: {METRIC_LABELS[metric]}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(max(0.0, min(means) - 5.0), min(100.0, max(means) + 5.0) if means else 100.0)
        fig.tight_layout()

        png_path = out_dir / f"baseline_{metric}.png"
        pdf_path = out_dir / f"baseline_{metric}.pdf"
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        saved[f"{metric}_png"] = str(png_path)
        saved[f"{metric}_pdf"] = str(pdf_path)

    return saved


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value) * 100.0:.2f}"
    except (TypeError, ValueError):
        return "--"


def _write_rows_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_family_table(grouped: dict[str, list[tuple[Path, dict[str, Any]]]], out_dir: Path) -> dict[str, str]:
    wanted = {"gnn-malware": "GNN Malware", "malbert": "MalBERT", "hydra": "HYDRA"}
    rows: list[dict[str, Any]] = []
    for method, runs in sorted(grouped.items()):
        key = method.lower()
        if key not in wanted:
            continue
        run_metrics = [metrics for _, metrics in runs]
        top1, _ = _metric_stats(run_metrics, "family_top1_accuracy")
        top5, _ = _metric_stats(run_metrics, "family_top5_accuracy")
        rows.append({"method": wanted[key], "family_top1_accuracy_pct": _fmt_pct(top1), "family_top5_accuracy_pct": _fmt_pct(top5)})

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "family_topk_table.csv"
    tex_path = out_dir / "family_topk_table.tex"
    _write_rows_csv(csv_path, ("method", "family_top1_accuracy_pct", "family_top5_accuracy_pct"), rows)
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Top-1 Acc. (\%) & Top-5 Acc. (\%) \\",
        r"\midrule",
    ]
    lines.extend(f"{row['method']} & {row['family_top1_accuracy_pct']} & {row['family_top5_accuracy_pct']} \\\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "tex": str(tex_path)}


def _append_optional_metrics(
    rows: list[dict[str, Any]],
    label: str,
    metrics_path: Path | None,
) -> None:
    if metrics_path and metrics_path.exists():
        metrics = _read_json(metrics_path)
        rows.append({"method": label, "cross_architecture_accuracy_pct": _fmt_pct(metrics.get("cross_architecture_accuracy"))})


def _save_cross_arch_table(
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]],
    out_dir: Path,
    xnerf_metrics: Path | None = None,
    no_alignment_metrics: Path | None = None,
) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    _append_optional_metrics(rows, "Without Alignment", no_alignment_metrics)
    if "safe" in {method.lower() for method in grouped}:
        for method, runs in sorted(grouped.items()):
            if method.lower() == "safe":
                avg, _ = _metric_stats([metrics for _, metrics in runs], "cross_architecture_accuracy")
                rows.append({"method": "SAFE", "cross_architecture_accuracy_pct": _fmt_pct(avg)})
                break
    _append_optional_metrics(rows, "X-NERF", xnerf_metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_architecture_table.csv"
    tex_path = out_dir / "cross_architecture_table.tex"
    _write_rows_csv(csv_path, ("method", "cross_architecture_accuracy_pct"), rows)
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Method & Cross-Architecture Acc. (\%) \\",
        r"\midrule",
    ]
    lines.extend(f"{row['method']} & {row['cross_architecture_accuracy_pct']} \\\\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "tex": str(tex_path)}


def _load_prediction_arrays(run_paths: list[Path]) -> tuple[np.ndarray, np.ndarray] | None:
    y_true_parts = []
    y_pred_parts = []
    for metrics_path in run_paths:
        pred_path = metrics_path.with_name("predictions.npz")
        if not pred_path.exists():
            continue
        data = np.load(pred_path)
        y_true_parts.append(np.asarray(data["y_true"]))
        if "y_pred" in data:
            y_pred_parts.append(np.asarray(data["y_pred"]))
        else:
            y_pred_parts.append(np.asarray(data["y_prob"]).argmax(axis=1))

    if not y_true_parts:
        return None
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def _save_confusion_matrices(grouped: dict[str, list[tuple[Path, dict[str, Any]]]], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for method, runs in sorted(grouped.items()):
        arrays = _load_prediction_arrays([path for path, _ in runs])
        if arrays is None:
            continue
        y_true, y_pred = arrays
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        display = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malware"])
        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{method} Confusion Matrix")
        fig.tight_layout()

        safe_method = method.lower().replace(" ", "_").replace("/", "_")
        png_path = out_dir / f"{safe_method}_confusion_matrix.png"
        pdf_path = out_dir / f"{safe_method}_confusion_matrix.pdf"
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        saved[f"{method}_png"] = str(png_path)
        saved[f"{method}_pdf"] = str(pdf_path)

    return saved


def generate(
    root: str | Path,
    out_dir: str | Path,
    xnerf_metrics: str | Path | None = None,
    no_alignment_metrics: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = Path(out_dir)
    grouped = _collect_runs(root)
    metric_plots = _save_metric_barplots(grouped, out_dir / "metric_plots")
    confusion_matrices = _save_confusion_matrices(grouped, out_dir / "confusion_matrices")
    family_table = _save_family_table(grouped, out_dir / "tables")
    cross_arch_table = _save_cross_arch_table(
        grouped,
        out_dir / "tables",
        Path(xnerf_metrics) if xnerf_metrics else None,
        Path(no_alignment_metrics) if no_alignment_metrics else None,
    )

    payload = {
        "root": str(root),
        "out_dir": str(out_dir),
        "methods": sorted(grouped),
        "metric_plots": metric_plots,
        "confusion_matrices": confusion_matrices,
        "family_table": family_table,
        "cross_architecture_table": cross_arch_table,
        "note": "Confusion matrices require predictions.npz files generated by xnerf.baselines.train_eval.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_artifacts.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate paper plots and confusion matrices from baseline results")
    parser.add_argument("--root", default="runs/baselines")
    parser.add_argument("--out-dir", default="reports/baseline_artifacts")
    parser.add_argument("--xnerf-metrics", default="runs/publication_test/test_metrics.json")
    parser.add_argument("--no-alignment-metrics", default="runs/ablation/no_grl/test/metrics.json")
    args = parser.parse_args(argv)
    payload = generate(args.root, args.out_dir, xnerf_metrics=args.xnerf_metrics, no_alignment_metrics=args.no_alignment_metrics)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
