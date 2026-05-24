from __future__ import annotations

from pathlib import Path

from xnerf.datasets.build_dataset import build_manifest
from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.utils.io import read_jsonl


def test_headerless_feature_csv_becomes_per_row_samples(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "AndMal2020" / "static" / "benign"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "CCCS Ben0.csv"
    csv_path.write_text(
        "shaaaa111,1,2,3,4,5\n"
        "shabbb222,10,20,30,40,50\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["data_type"] == "feature_csv"
    assert rows[0]["label"] == 0
    assert rows[0]["feature_dim"] == 5

    ds = MalwareManifestDataset(out)
    item = ds[0]
    assert item["memory_trace"].shape == (512, 8)
    assert item["binary_image"].sum().item() == 0


def test_headered_feature_csv_uses_label_and_numeric_columns(tmp_path: Path):
    csv_dir = tmp_path / "raw" / "AndMal2020" / "dynamic"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "dynamic.csv"
    csv_path.write_text(
        "sha256,label,family,api_count,net_count,text_col\n"
        "abc123,benign,cleanfam,1,2,ignore\n"
        "def456,malicious,badfam,5,7,ignore\n",
        encoding="utf-8",
    )
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 2
    assert rows[0]["label"] == 0
    assert rows[0]["family"] == "cleanfam"
    assert rows[0]["feature_dim"] == 2
    assert rows[1]["label"] == 1


def test_public_labels_csv_is_metadata_not_sample(tmp_path: Path):
    raw = tmp_path / "raw" / "cape"
    raw.mkdir(parents=True)
    (raw / "public_labels.csv").write_text(
        "sha256,label,family\n"
        "abc123,malicious,trojan\n",
        encoding="utf-8",
    )
    feature_dir = tmp_path / "raw" / "AndMal2020" / "static"
    feature_dir.mkdir(parents=True)
    (feature_dir / "features.csv").write_text("abc123,1,2,3\n", encoding="utf-8")
    out = tmp_path / "processed" / "manifest.jsonl"

    build_manifest(tmp_path, out, make_splits=False)

    rows = read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["family"] == "trojan"
