# DATASET_TABLE.md
## X-NERF++ Dataset Inventory

This table reflects what the **ingestion code can parse** (per
`xnerf/datasets/build_dataset.py`, `extract_archives.py`, README), not what
data is physically present in the repository. The `data/archives/` tree at
audit time contains almost no real data — see "Present in Repo" column.
`data/raw/`, `data/cache/`, `data/processed/` are entirely git-ignored, so
no manifest, cache, or split file exists in version control at all.

| Dataset | Modality (Static/Dynamic) | Architecture(s) | Families / Labels | Ingestion Path | Present in Repo at Audit Time | Status |
|---|---|---|---|---|---|---|
| **MalNet-Tiny** | Static (byte-images) + Static (call graphs) | inferred from filename (`infer_arch`), defaults x86 | benign/malware via `infer_label` (path keyword match), family = parent folder name | `process` generic binary path → `_binary_image`; graph via `.edgelist` → `nx.read_edgelist` | `data/archives/malnet_tiny/{images,graphs}/.gitkeep` only — **no data** | IMPLEMENTED (parser) / NO DATA |
| **AndMal2020** (static) | Static numeric feature CSV | n/a (feature vectors) | label/family from CSV columns or filename (`CCCS Ben*` → benign) | `process_feature_csv` (headerless/headered) | `.gitkeep` only — **no data** | IMPLEMENTED (parser) / NO DATA |
| **AndMal2020** (dynamic) | Dynamic numeric feature CSV (before/after reboot) | n/a | label/family from CSV header columns | `process_feature_csv` | `.gitkeep` only — **no data** | IMPLEMENTED (parser) / NO DATA |
| **CICMalDroid2020** | Static + dynamic feature CSV (`CSV.zip`) | n/a | label/family from CSV | `process_feature_csv`; explicitly excluded from "first working run" per README | `.gitkeep` only — **no data**; `scripts/inspect_cicmaldroid.py` exists to inspect a local copy outside the repo | IMPLEMENTED (parser) / NO DATA |
| **Drebin** | Static Android feature vectors | n/a | label/family from CSV (if present) | `process_feature_csv` | **`drebin.zip` IS present** (binary archive, not extracted/inspected during this audit) | IMPLEMENTED (parser) / DATA PRESENT BUT UNVERIFIED |
| **EMBER** | Static PE feature vectors (Parquet or CSV) | n/a (PE/Windows) | label/family from Parquet/CSV columns; placeholder-pattern filter explicitly recognizes `train_ember_2018_v2_features` as a non-family name | `process_feature_parquet` / `process_feature_csv` | `.gitkeep` only — **no data** | IMPLEMENTED (parser) / NO DATA |
| **VirusShare** | Raw binaries (authorized only) | inferred from filename | label=1 default (malware), family=unknown unless mapped | generic binary path; requires `VIRUSSHARE_API_KEY` and pre-approved hash list in `download.py` | not present (download-gated, requires explicit authorization) | IMPLEMENTED (download helper) / NO DATA — explicitly gated for ethical/legal reasons |
| **CAPE / Avast sandbox reports** | Dynamic (API calls, network events, process/memory events) | n/a | label from path/keyword default + `public_labels.csv` mapping; family from mapping CSV | `enrich_dynamic_report` → `parse_cape_report` | `public_labels.csv` + `ReadMe.md` present; **no actual `.json`/`.zip` report files** | IMPLEMENTED (parser, unit-tested) / NO REPORT DATA |
| **MalBehavD-V1** | Dynamic API-call-sequence CSV | n/a | `labels` column (0/1) | `process_api_sequence_csv` (auto-detected via `t_\d+`/`api_\d+` column pattern) | **`MalBehavD-V1-dataset.csv` IS present** | IMPLEMENTED (parser) / DATA PRESENT BUT UNVERIFIED |
| **MalAPI-2019** (implied by test suite) | Dynamic API-call-sequence text (`all_analysis_data.txt` + sibling `labels.csv`) | n/a | family = line-aligned label from `labels.csv` | `process_api_sequence_txt` | not present in `data/archives/`; logic only exercised by `tests/test_feature_csv.py::test_malapi_text_sequences_use_line_labels` | IMPLEMENTED (parser, unit-tested) / NO DATA |
| **MalwareAnalysisDatasetsAPICallSequences** (implied by test suite) | Dynamic numeric API-call-sequence CSV (`t_0,t_1,...,malware`) | n/a | `malware` column | `process_api_sequence_csv` | not present; only exercised by `tests/test_feature_csv.py::test_numeric_api_sequence_csv_becomes_dynamic_rows` | IMPLEMENTED (parser, unit-tested) / NO DATA |
| **CIC-YNU IoTMal** (special-cased) | n/a (inferred from `dataset == "CIC-YNU_IoTMal"` string match in parquet path) | IoT (implied) | family-conditioned **stochastic downsampling** (`IOTMAL_FAMILY_SAMPLE_PROBS`: Mirai 0.2%, Benign 1%, DarkNexus 10%, Unknown 10%, Gafgyt 20%, Generic 40%, all others kept 100%) | `process_feature_parquet` special-case branch | not present anywhere in `data/archives/` | IMPLEMENTED (special-case logic) / NO DATA — **undocumented in README**, only discoverable by reading source |
| **"new_dataset" generic extension point** | Any of the above, by file type | n/a | README explicitly states no extractor code change is needed; file-type dispatch is fully generic | any matching dispatch branch | n/a (documentation feature) | IMPLEMENTED — genuinely extensible via folder convention `data/archives/<name>/<modality>/*` |

### Cross-Cutting Notes

- **Architecture inference is filename-substring matching, not binary
  analysis.** `infer_arch()` (`build_dataset.py`) and the duplicate
  `infer_arch` import path checks tokens like `"arm64"`, `"mipsel"`,
  `"x64"` against the lower-cased file path and **defaults to `x86`** if
  none match. For non-PE/ELF feature-CSV/Parquet datasets (the overwhelming
  majority of supported sources), there is no real architecture signal in
  the data at all, so the `arch` field is in practice **almost always the
  `x86` default** for static-feature datasets. `xnerf/datasets/audit.py`
  (`architecture_audit_report`, `detect_single_architecture_dataset`) is a
  built-in self-check specifically designed to detect and flag this
  single-architecture degeneracy — its existence is itself evidence that
  architecture diversity is a known, unresolved data-quality risk for this
  project, not a solved problem.
- **Label normalization** (`parse_label_value`) accepts an unusually wide
  set of synonyms (`0/1, b/s, benign/malware, true/false, infected,
  goodware/clean/normal`), defaulting to `infer_label()`'s keyword search
  over the file path when no label column is present.
- **Family normalization** (`xnerf/datasets/family_cleaning.py`) maps raw,
  highly heterogeneous family strings (e.g., `"Spy"`, `"worms"`, `"PUA"`)
  to a canonical capitalized vocabulary via a hand-built alias table plus
  prefix-matching fallback (`startswith("trojan")` → `"Trojan"`, etc.), and
  explicitly recognizes and discards 14 **dataset-name-as-family
  placeholder patterns** (e.g. a regex flags `train_ember_2018_v2_features`,
  `cicmaldroid2020`, `malnet_tiny`, `andmal2020`, `<unknown>`,
  `no_category` as *not real family names* and remaps them to
  `"unknown"`). This placeholder-filtering logic is unit-tested
  (`tests/test_family_normalization.py`) and is one of the more
  defensible/non-trivial pieces of data engineering in the repo, since
  several of the underlying public datasets genuinely use the dataset's own
  name as a junk "family" value in their distributed CSVs.
- **No dataset in this table has end-to-end recorded statistics** (sample
  counts, class balance, family distribution) anywhere in the repository —
  no `DATASET_TABLE.md`, EDA notebook, or summary JSON exists upstream of
  this audit. Any per-dataset sample counts appearing in a future paper
  must be generated by actually running `build_manifest()` against
  acquired data and are **not currently available**.

### Cross-Platform / Cross-Architecture Coverage Claim (README) vs Evidence

README states support for "x86, x64, ARM, ARM64, MIPS, and RISC-V using
Capstone." This is true **only for the ISR/disassembly path**
(`DisassemblerProcessor`, `preprocessing/ontology.py::ARCH_TO_ID`), which
itself is — per `ARCHITECTURE_REPORT.md` §2.5 — **not consumed by the
model's forward pass**. None of the seven dataset sources actually
ingested by `build_dataset.py` provide verified multi-architecture binary
samples in the repository; the six-way architecture coverage is therefore a
**capability of the disassembly module**, not a demonstrated property of
any assembled training set.
