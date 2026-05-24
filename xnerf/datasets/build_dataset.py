from __future__ import annotations

import argparse
import json
from pathlib import Path

from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline
from xnerf.utils.io import sha256_file, write_jsonl


def infer_arch(path: Path) -> str:
    text = path.name.lower()
    for arch in ("arm64", "arm", "mips", "riscv", "x64", "x86"):
        if arch in text:
            return arch
    return "x86"


def build_manifest(root: Path, out: Path, max_binary_bytes: int = 2_000_000) -> None:
    raw = root / "raw"
    cache = root / "cache" / "isr"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    if not raw.exists():
        raise FileNotFoundError(f"raw dataset folder not found: {raw}")
    normalizers: dict[str, ArchitectureNormalizationPipeline] = {}
    for path in raw.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".zip", ".7z", ".rar"}:
            continue
        record = {
            "path": str(path),
            "sha256": sha256_file(path),
            "dataset": path.relative_to(raw).parts[0] if len(path.relative_to(raw).parts) else "unknown",
            "label": 1,
            "family": path.parent.name,
            "arch": infer_arch(path),
        }
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
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.validate:
        validate_manifest(args.out)
    else:
        build_manifest(args.root, args.out)


if __name__ == "__main__":
    main()
