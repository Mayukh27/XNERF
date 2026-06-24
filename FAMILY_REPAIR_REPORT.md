# FAMILY_REPAIR_REPORT.md
## Family Classification Repair Report

## Issue

Rows with family labels such as `train_ember_2018_v2_features`, `public_small_reports`, or `dynamic_api_call_sequence_per_malware_100_0_306` are dataset/source names, not malware families. Training family CE on those labels is scientifically invalid.

## Fix

Implemented invalid-family masking:

- `xnerf/datasets/loaders.py:240`: malware rows whose raw family matches placeholder rules are marked invalid.
- `xnerf/datasets/loaders.py:282`: invalid family labels become `-1`.
- `xnerf/training/losses.py:12`: family CE uses `ignore_index=-1`.
- All-ignored batches get a zero family loss that remains connected to the computation graph.

## Counts

Valid/ignored counts require an actual manifest. No manifest is committed in this repository, so exact counts are not fabricated here.

To compute after manifest generation:

```powershell
python - <<'PY'
from collections import Counter
from xnerf.datasets.loaders import MalwareManifestDataset
ds = MalwareManifestDataset("data/processed/train_manifest.jsonl")
c = Counter(int(ds[i]["family_label"]) for i in range(len(ds)))
print({"ignored": c[-1], "valid": sum(v for k,v in c.items() if k >= 0)})
PY
```

