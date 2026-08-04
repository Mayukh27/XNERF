from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
METHOD_LABELS = {
    "ember-rf": "EMBER RF",
    "ember rf": "EMBER RF",
    "malconv": "Byte-CNN",
    "byte-cnn": "Byte-CNN",
    "byte_cnn": "Byte-CNN",
    "cnn-malware": "Byte-CNN",
    "cnn malware": "Byte-CNN",
    "t_api": "Transformer-API",
    "transformer-api": "Transformer-API",
    "transformer_api": "Transformer-API",
    "malbert": "Transformer-API",
    "latefusion": "LateFusion",
    "hydra": "LateFusion",
    "cfg_gnn": "CFG-GNN",
    "cfg-gnn": "CFG-GNN",
    "gnn-malware": "CFG-GNN",
    "gnn malware": "CFG-GNN",
    "safe": "SAFE",
}


def _read_metrics(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise RuntimeError("No metrics.json files found.")
    return rows


def _fmt(values: list[float]) -> str:
    scaled = [100.0 * value for value in values]
    if len(scaled) == 1:
        return f"{scaled[0]:.2f}"
    return f"{mean(scaled):.2f} $\\pm$ {stdev(scaled):.2f}"


def summarize(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    metric_files = sorted(root.glob("*/seed_*/metrics.json"))
    if not metric_files:
        metric_files = sorted(root.glob("**/metrics.json"))
    rows = _read_metrics(metric_files)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        raw_name = str(row.get("method") or row.get("baseline") or "unknown")
        name = METHOD_LABELS.get(raw_name.strip().lower(), raw_name)
        grouped.setdefault(name, []).append(row)

    summary: dict[str, Any] = {}
    latex_lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{AUC} \\\\",
        "\\midrule",
    ]
    for method, method_rows in sorted(grouped.items()):
        summary[method] = {
            key: [row.get(key) for row in method_rows if row.get(key) is not None]
            for key in METRIC_KEYS
        }
        cells = []
        for key in METRIC_KEYS:
            vals = [float(value) for value in summary[method][key]]
            cells.append(_fmt(vals) if vals else "--")
        latex_lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    return {
        "root": str(root),
        "metric_files": [str(path) for path in metric_files],
        "summary": summary,
        "latex": "\n".join(latex_lines),
    }


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Summarize baseline metrics into a paper table")
    parser.add_argument("--root", default="runs/baselines")
    parser.add_argument("--out", default="runs/baselines/summary.json")
    args = parser.parse_args(argv)
    payload = summarize(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(payload["latex"])
    return payload


if __name__ == "__main__":
    main()
