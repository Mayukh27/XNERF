from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline
from xnerf.sandbox.cape_parser import parse_cape_report
from xnerf.utils.io import sha256_file, write_jsonl
from xnerf.utils.tokenization import tokens_to_ids


def infer_arch(path: Path) -> str:
    text = path.name.lower()
    for arch in ("arm64", "arm", "mips", "riscv", "x64", "x86"):
        if arch in text:
            return arch
    return "x86"


def infer_label(path: Path) -> int:
    text = str(path).lower()
    if any(token in text for token in ("benign", "goodware", "clean")):
        return 0
    return 1


def parse_label_value(value: str | int | float | None, default: int = 1) -> int:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"0", "benign", "goodware", "clean", "false", "normal"}:
        return 0
    if text in {"1", "malware", "malicious", "true", "infected"}:
        return 1
    return default


def enrich_dynamic_report(record: dict, path: Path) -> dict:
    if path.suffix.lower() != ".json":
        return record
    parts = {part.lower() for part in path.parts}
    if not ({"cape", "avast", "andmal2020", "cicmaldroid2020"} & parts):
        return record
    try:
        parsed = parse_cape_report(path)
    except Exception as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["api_ids"] = tokens_to_ids(parsed["api_calls"], vocab_size=8192, max_len=256, prefix="api")
    record["network_ids"] = tokens_to_ids(parsed["network_events"], vocab_size=4096, max_len=256, prefix="net")
    record["memory_event_count"] = len(parsed.get("memory_events", []))
    record["process_event_count"] = len(parsed.get("process_events", []))
    record["api_call_count"] = len(parsed.get("api_calls", []))
    record["network_event_count"] = len(parsed.get("network_events", []))
    score = parsed.get("summary", {}).get("score")
    if score is not None:
        record["sandbox_score"] = score
    return record


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_header(row: list[str]) -> bool:
    if not row:
        return True
    numeric = sum(_safe_float(cell) is not None for cell in row[1:])
    return numeric < max(1, len(row[1:]) // 2)


def norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


ID_COLUMNS = {
    "sha",
    "sha1",
    "sha256",
    "md5",
    "hash",
    "sample",
    "sample_id",
    "sampleid",
    "file",
    "filename",
    "apk",
    "apk_name",
    "id",
}

LABEL_COLUMNS = {"label", "class", "category", "verdict", "is_malware", "malware", "type"}
FAMILY_COLUMNS = {"family", "malware_family", "class_name", "category_name"}


def column_index(headers: list[str], candidates: set[str]) -> int | None:
    normalized = [norm_col(h) for h in headers]
    for i, name in enumerate(normalized):
        if name in candidates:
            return i
    return None


def is_label_map_csv(path: Path, first_row: list[str]) -> bool:
    name = path.name.lower()
    if "public_labels" in name and _looks_like_header(first_row):
        return True
    headers = [norm_col(x) for x in first_row]
    header_set = set(headers)
    metadata = ID_COLUMNS | LABEL_COLUMNS | FAMILY_COLUMNS
    non_metadata = [h for h in headers if h not in metadata]
    return bool(header_set & ID_COLUMNS) and bool(header_set & (LABEL_COLUMNS | FAMILY_COLUMNS)) and len(non_metadata) <= 1


def load_label_maps(raw: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in raw.rglob("*.csv"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f)
                first = next(reader, [])
                if not is_label_map_csv(path, first):
                    continue
                id_idx = column_index(first, ID_COLUMNS)
                label_idx = column_index(first, LABEL_COLUMNS)
                family_idx = column_index(first, FAMILY_COLUMNS)
                if id_idx is None:
                    continue
                for row in reader:
                    if id_idx >= len(row):
                        continue
                    sample_id = row[id_idx].strip()
                    if not sample_id:
                        continue
                    item = labels.setdefault(sample_id, {})
                    if label_idx is not None and label_idx < len(row):
                        item["label"] = parse_label_value(row[label_idx], default=item.get("label", infer_label(path)))
                    if family_idx is not None and family_idx < len(row) and row[family_idx].strip():
                        item["family"] = row[family_idx].strip()
        except OSError:
            continue
    return labels


def feature_family_from_path(path: Path) -> str:
    stem = path.stem
    if stem.lower() in {"csv", "features", "feature_vectors_static", "feature_vectors"}:
        return path.parent.name
    return stem


def build_memory_trace(features: list[float], rows: int = 512, cols: int = 8):
    import torch

    out = torch.zeros(rows * cols, dtype=torch.float32)
    if features:
        values = torch.tensor(features[: rows * cols], dtype=torch.float32)
        if values.numel() > 1:
            values = (values - values.mean()) / values.std().clamp_min(1e-6)
        out[: values.numel()] = values
    return out.view(rows, cols)


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (cleaned or fallback)[:80]


def row_features(row: list[str], numeric_indexes: list[int]) -> list[float]:
    features = []
    for i in numeric_indexes:
        if i < len(row):
            value = _safe_float(row[i])
            if value is not None:
                features.append(value)
    return features


def process_feature_csv(
    path: Path,
    raw: Path,
    cache: Path,
    label_maps: dict[str, dict] | None = None,
    max_rows_per_csv: int | None = None,
) -> list[dict]:
    """Convert headerless or headered numeric feature-vector CSV rows.

    Headerless expected row shape:
        sample_id, f1, f2, f3, ...

    Headered tables use known id/label/family columns when present and all
    numeric non-label columns as features. Label-map CSVs such as
    public_labels.csv return no training samples.

    Outputs one dataset sample per CSV row. Numeric features are cached as a
    [512,8] tensor consumed by MalwareManifestDataset.memory_trace.
    """

    rows = []
    feature_cache = cache / "features"
    feature_cache.mkdir(parents=True, exist_ok=True)
    dataset = path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown"
    default_label = infer_label(path)
    default_family = feature_family_from_path(path)
    label_maps = label_maps or {}
    file_key = sha256_file(path)[:12]
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, [])
        if not first:
            return rows
        has_header = _looks_like_header(first)
        if has_header and is_label_map_csv(path, first):
            print(f"Loaded label-map CSV metadata, not training samples: {path}")
            return rows
        id_idx = 0
        label_idx = None
        family_idx = None
        numeric_indexes = list(range(1, len(first)))
        pending_rows = []
        if has_header:
            headers = first
            id_idx = column_index(headers, ID_COLUMNS)
            label_idx = column_index(headers, LABEL_COLUMNS)
            family_idx = column_index(headers, FAMILY_COLUMNS)
            normalized = [norm_col(h) for h in headers]
            skip_cols = {idx for idx in (id_idx, label_idx, family_idx) if idx is not None}
            numeric_indexes = [
                i
                for i, name in enumerate(normalized)
                if i not in skip_cols and name not in LABEL_COLUMNS and name not in FAMILY_COLUMNS
            ]
            id_idx = 0 if id_idx is None else id_idx
        else:
            pending_rows.append(first)

        for idx, row in enumerate(reader):
            pending_rows.append(row)

        for idx, row in enumerate(pending_rows):
            if max_rows_per_csv is not None and len(rows) >= max_rows_per_csv:
                break
            if not row:
                continue
            if has_header and idx == 0 and row == first:
                continue
            sample_id = row[id_idx].strip() if id_idx < len(row) else f"{path.stem}_{idx}"
            sample_id = sample_id or f"{path.stem}_{idx}"
            features = row_features(row, numeric_indexes)
            if not features:
                continue
            import torch

            feature_path = feature_cache / f"{file_key}_{safe_name(path.stem, 'csv')}_{idx}_{safe_name(sample_id, 'sample')[:16]}.pt"
            torch.save(build_memory_trace(features), feature_path)
            mapped = label_maps.get(sample_id, {})
            label = mapped.get("label", default_label)
            if label_idx is not None and label_idx < len(row):
                label = parse_label_value(row[label_idx], default=label)
            family = mapped.get("family", default_family)
            if family_idx is not None and family_idx < len(row) and row[family_idx].strip():
                family = row[family_idx].strip()
            rows.append(
                {
                    "path": str(path),
                    "row_index": idx,
                    "sample_id": sample_id,
                    "sha256": sample_id if len(sample_id) >= 16 else f"{sha256_file(path)}:{idx}",
                    "dataset": dataset,
                    "data_type": "feature_csv",
                    "feature_path": str(feature_path),
                    "feature_dim": len(features),
                    "label": label,
                    "family": family,
                    "arch": infer_arch(path),
                }
            )
    return rows


def split_rows(
    rows: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic train/val/test split.

    Stratifies coarsely by label and family where available. If a bucket is too
    small, it is still assigned deterministically so every sample appears in
    exactly one split.
    """

    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("label", 0), row.get("family", "unknown"))
        buckets.setdefault(key, []).append(row)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        if n < 3:
            train.extend(bucket)
            continue
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    if not test and len(train) > 2:
        test.append(train.pop())
    if not val and len(train) > 2:
        val.append(train.pop())
    return train, val, test


def write_splits(out: Path, rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, int]:
    train, val, test = split_rows(rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    split_dir = out.parent
    write_jsonl(split_dir / "train_manifest.jsonl", train)
    write_jsonl(split_dir / "val_manifest.jsonl", val)
    write_jsonl(split_dir / "test_manifest.jsonl", test)
    return {"train": len(train), "val": len(val), "test": len(test)}


def build_manifest(
    root: Path,
    out: Path,
    max_binary_bytes: int = 2_000_000,
    make_splits: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> None:
    raw = root / "raw"
    cache = root / "cache" / "isr"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    if not raw.exists():
        raise FileNotFoundError(f"raw dataset folder not found: {raw}")
    label_maps = load_label_maps(raw)
    if label_maps:
        print(f"Loaded {len(label_maps)} CSV label-map entries")
    normalizers: dict[str, ArchitectureNormalizationPipeline] = {}
    for path in raw.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".zip", ".7z", ".rar"}:
            continue
        if path.suffix.lower() == ".csv":
            csv_rows = process_feature_csv(path, raw=raw, cache=cache, label_maps=label_maps)
            rows.extend(csv_rows)
            print(f"Parsed {len(csv_rows)} feature rows from {path}")
            continue
        record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "dataset": path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown",
            "label": infer_label(path),
            "family": path.parent.name,
            "arch": infer_arch(path),
        }
        record = enrich_dynamic_report(record, path)
        if path.stat().st_size <= max_binary_bytes and path.suffix.lower() in {".bin", ".exe", ".dll", ".so", ".elf", ""}:
            arch = record["arch"]
            normalizers.setdefault(arch, ArchitectureNormalizationPipeline(arch=arch))
            blob = path.read_bytes()
            isr = normalizers[arch].process({"bytes": blob, "arch": arch})
            isr_path = cache / f"{record['sha256']}.pt"
            import torch

            torch.save(isr, isr_path)
            record["isr_path"] = str(isr_path)
        rows.append(record)
    if not rows:
        raise RuntimeError(f"No dataset files found under {raw}. Missing datasets are allowed, but at least one usable dataset is required.")
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} manifest rows to {out}")
    if make_splits:
        counts = write_splits(out, rows, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        print(f"Wrote split manifests to {out.parent}: {counts}")


def validate_manifest(path: Path) -> None:
    count = 0
    missing = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            count += 1
            if not Path(row["path"]).exists():
                missing.append(row["path"])
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest paths are missing, first={missing[0]}")
    if count == 0:
        raise RuntimeError(f"Manifest is empty: {path}")
    print(f"Validated {count} manifest rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build X-NERF++ unified manifest")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/manifest.jsonl"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.validate:
        validate_manifest(args.out)
    else:
        build_manifest(
            args.root,
            args.out,
            make_splits=not args.no_split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
