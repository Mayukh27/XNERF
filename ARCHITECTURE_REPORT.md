# ARCHITECTURE_REPORT.md
## X-NERF++ (XNERF) — Full Architecture Reconstruction

Repository: `github.com/Mayukh27/XNERF` (branch: `main`, HEAD `17e2fa6`)
Audited by static source-code reverse engineering. 38 commits, single primary
author, no committed checkpoints/metrics (`.gitignore` excludes
`data/raw`, `data/cache`, `data/processed`, `runs/`, `checkpoints/`,
`models/*.pt`). All claims below are evidence-tagged to file paths.

---

## 1. High-Level Pipeline Map

```
                         ┌────────────────────────────┐
                         │   data/archives/<dataset>/  │   (user-supplied ZIP/TAR)
                         └──────────────┬─────────────┘
                                         │ extract_archives.py
                                         ▼
                         ┌────────────────────────────┐
                         │   data/raw/<dataset>/...   │   (flat extracted files)
                         └──────────────┬─────────────┘
                                         │ build_dataset.py (build_manifest)
                                         ▼
            ┌─────────────────────────────────────────────────────┐
            │  data/processed/manifest.jsonl                      │
            │  + train_manifest.jsonl / val_manifest.jsonl /       │
            │    test_manifest.jsonl  + family_vocab.json          │
            └───────────────────────────┬───────────────────────--┘
                                         │ generate_cache_from_manifest (optional)
                                         ▼
            ┌─────────────────────────────────────────────────────┐
            │ data/cache/isr/  (per-sample ISR .pt + feature .pt)  │
            └───────────────────────────┬───────────────────────--┘
                                         │ MalwareManifestDataset (loaders.py)
                                         ▼
            ┌─────────────────────────────────────────────────────┐
            │  Multimodal batch dict (binary_image, graph_x/ei,    │
            │  api_ids, network_ids, memory_trace, isr, arch_id,   │
            │  label, family_label)                                │
            └───────────────────────────┬───────────────────────--┘
                                         │ XNERFPlusPlus.forward (model.py)
                                         ▼
            ┌─────────────────────────────────────────────────────┐
            │ Encoders → SFS → MNEF Field → Aligner →              │
            │ Trajectory Decoder → {malware_logits, family_logits, │
            │ zero_shot_embedding, arch_logits, field, stage/trans} │
            └───────────────────────────┬───────────────────────--┘
                          ┌──────────────┴───────────────┐
                          ▼                               ▼
              XNerfTrainer (trainer.py)         Explainability / API / Sandbox
              classification_losses             ReportGenerator, FastAPI app,
              (malware_ce, family_ce,            terminal sandbox CLI
               arch_adv, field_smooth)
                          │
                          ▼
              checkpoints/best.pt, last.pt
                          │
            ┌─────────────┼──────────────────┬───────────────┐
            ▼              ▼                  ▼               ▼
        test_after_   zero_shot/         deployment/      xnerf_output/
        training.py   build_prototypes,  export_checkpoint  summary.json
        (test split)  evaluate_zero_shot  → local-inference  + copied
                       (cosine prototype    .pt for API/CLI    artifacts
                        bank)
```

This entire chain is orchestrated end-to-end by two near-duplicate
orchestrator modules: `xnerf/pipeline/kaggle_run.py` (Kaggle paths,
`/kaggle/...`) and `xnerf/pipeline/local_run.py` (local paths, `data/...`,
`runs/...`). Both expose the same six sub-commands
(`build-manifest`, `train`, `validate`, `test`, `zero-shot`, `export`) plus a
monolithic `pipeline` command that chains them with idempotency markers
(`train_done.json`).

**Evidence:** `xnerf/pipeline/kaggle_run.py`, `xnerf/pipeline/local_run.py`,
`README.md` lines 147–211.

---

## 2. The 10 Requested Sub-Pipelines

### 2.1 Data Ingestion Pipeline
```
data/archives/<dataset>/<modality>/*.zip|*.tar|*.tar.gz
        │  extract_dataset_archives()  [xnerf/datasets/extract_archives.py]
        │  - rglob over archive_root
        │  - shutil.unpack_archive() per archive, idempotent via
        │    ".extracted_<stem>" marker files
        │  - also copies loose .csv/.json/.jsonl/.txt/.parquet files
        │  - recursively re-extracts nested archives inside data/raw/
        ▼
data/raw/<dataset>/<modality>/... (flat files)
```
**Inputs:** archive root path (default `/kaggle/input/.../archives` or
`data/archives`). **Outputs:** mirrored directory tree under `data/raw`.
**Status:** IMPLEMENTED — pure filesystem operation, no ML dependency.

### 2.2 Dataset Extraction / Parsing Pipeline (per-file type dispatch)
```
data/raw/**/*  ──▶  build_manifest() dispatch in build_dataset.py
   ├─ *.csv  → is_api_sequence_csv()? ─yes→ process_api_sequence_csv()
   │                                   └no→ process_feature_csv()
   ├─ all_analysis_data.txt → process_api_sequence_txt()
   ├─ *.parquet → process_feature_parquet()
   └─ other (binary/json/etc.) → generic record + enrich_dynamic_report()
                                   (CAPE/Avast JSON via cape_parser.py)
```
File-type detection is heuristic (`_looks_like_header`, regex-based column
matchers `ID_COLUMNS` / `LABEL_COLUMNS` / `FAMILY_COLUMNS`, and the
`apicallsequence`/`t_\d+` column-name pattern for behaviour CSVs).
**Status:** IMPLEMENTED for: headerless/headered feature CSV, numeric/text
API-sequence CSV, `all_analysis_data.txt` (MalAPI-2019 style), Parquet
(EMBER style), CAPE/Avast JSON (zipped or loose). Verified by
`tests/test_build_dataset.py`, `tests/test_feature_csv.py`,
`tests/test_cape_parser.py`.

### 2.3 Manifest Generation Pipeline
```
rows[] ──▶ write_jsonl(manifest.jsonl)
        ├─ write_family_vocab() → family_vocab.json
        └─ split_rows() / split_train_val() [stratified by (label,family)]
              or _hash_split() [deterministic SHA-256 hash mode, supports
              --resume via per-file progress markers in
              data/cache/manifest_progress/]
            ──▶ train_manifest.jsonl / val_manifest.jsonl / test_manifest.jsonl
```
Two split strategies are implemented: `stratified` (in-memory bucket-and-cut,
default) and `hash` (streaming, resumable, deterministic via
`sha256(seed:key)`). **Status:** IMPLEMENTED
(`xnerf/datasets/build_dataset.py:710-1088`).

### 2.4 Cache Generation Pipeline (tensor materialization)
```
manifest row ──▶ generate_cache_from_manifest() [build_dataset.py:1184]
   ├─ data_type == feature_csv/feature_parquet
   │     → _feature_csv/_parquet_values_for_row() → build_memory_trace()
   │       (z-score normalize, pad/truncate to 512×8) → .pt under
   │       cache/isr/features/<file_key[:2]>/<shard>/<name>_<row>_<key>.pt
   └─ binary candidate (.bin/.exe/.dll/.so/.elf/no-ext, ≤2MB)
         → ArchitectureNormalizationPipeline(arch).process(bytes)
           → ISR tensor [1024,4] → cache/isr/<sha256>.pt
```
Cache paths are **deterministic** from `(file_key, row_index, sample_id)` so
that train/val/test split manifests (subsets of the same rows) can reuse a
single feature-cache shard without recomputation
(`tests/test_feature_csv.py::test_split_manifest_reuses_parent_generated_cache`).
**Status:** IMPLEMENTED. `--manifest-only` flag allows manifest construction
*without* materializing any cache (used for fast iteration on column-mapping
logic).

### 2.5 Feature Extraction Pipeline (architecture normalization / ISR)
```
raw bytes ──▶ ArchitectureNormalizationPipeline [preprocessing/pipeline.py]
   1. DisassemblerProcessor(arch) — Capstone disasm → Instruction(addr,
      mnemonic, op_str, size); arch∈{x86,x64,arm,arm64,mips,riscv}
   2. SemanticMapperProcessor — MNEMONIC_TO_SEMANTIC dict lookup →
      16-class ontology (PAD, UNKNOWN, DATA_TRANSFER, ARITHMETIC, LOGIC,
      CONTROL_FLOW, CALL, RETURN, STACK, MEMORY_LOAD, MEMORY_STORE,
      CRYPTO_HINT, SYSTEM, PRIVILEGE, NETWORK, ANTI_ANALYSIS)
   3. ISRBuilderProcessor — pack into LongTensor [max_len=1024, 4]
      columns = (semantic_id, arch_id, address_delta_bucket≤255, size)
   ▼
ISR tensor [1024,4]  (the cross-architecture "common representation")
```
**Status:** IMPLEMENTED for the 6 listed architectures, dependent on
`capstone`. **Important limitation:** the ontology covers ~45 hand-listed
mnemonics; any mnemonic not in `MNEMONIC_TO_SEMANTIC` (the large majority of
each ISA's instruction set, e.g. SIMD/AVX, NEON, VFP, most ARM64 and RISC-V
opcodes) silently maps to `UNKNOWN` (semantic_id=1). `PRIVILEGE` (id 13) is
declared in `SEMANTIC_CLASSES` but **no mnemonic is ever mapped to it** —
dead ontology class. **Evidence:** `xnerf/preprocessing/ontology.py`.
The ISR tensor is produced and cached but **is never consumed by the model
forward pass** — `XNerfBatch` declares an `isr` field and the dataset loader
populates it, but `XNERFPlusPlus.forward()` never reads `batch["isr"]`
(confirmed: no reference to `"isr"` inside `xnerf/model.py`). This is a
**PARTIALLY_IMPLEMENTED** feature: computed end-to-end, plumbed into the
batch dict, but architecturally disconnected from the model that is supposed
to consume it.

### 2.6 Training Pipeline
```
MalwareManifestDataset(train) ──▶ DataLoader(collate_dicts)
        │
        ▼
XNerfTrainer.fit() [trainer.py]
  for epoch in range(start_epoch, epochs+1):
     for batch in train_loader:
        outputs = model(batch)                  # forward, AMP autocast
        losses  = classification_losses(...)     # see §5
        loss    = sum(losses.values())
        scaler.scale(loss/grad_accum).backward()
        every grad_accum steps:
           unscale → finite-check grads → clip_grad_norm_ → optimizer.step()
           finite-check params/optimizer-state → zero_grad
     val_loss = self.validate() (if val set)
     if val_loss < best: save checkpoints/best.pt
     else bad_epochs += 1; early-stop at patience
     always save checkpoints/last.pt
```
Extensive **NaN/Inf guarding** is implemented at every stage (model outputs,
loss terms, gradients, post-step parameters, optimizer state) — see
`XNerfTrainer._check_finite_tensor/_check_gradients/_check_parameters/
_check_optimizer_state`. This is genuine defensive engineering, not
boilerplate. **Status:** IMPLEMENTED. Mixed precision (`torch.cuda.amp`),
gradient accumulation, `DataParallel` for >1 GPU, resumable checkpoints
(optimizer + scaler + epoch + best_val_loss + bad_epochs state), atomic
checkpoint writes (`tmp.replace(path)` with `PermissionError` fallback) are
all present (`xnerf/training/trainer.py`).
A Ray Tune launcher (`xnerf/training/ray_train.py`) wraps `run_training` for
hyperparameter search but only varies `config_path`, i.e. **no actual search
space is defined** — PARTIALLY_IMPLEMENTED / scaffold only.

### 2.7 Evaluation Pipeline
```
checkpoints/best.pt ──▶ test_after_training.run_test()
   - loads model, runs test_manifest through model in eval mode
   - collects y_prob, y_true, zero_shot_embedding, arch_pred/true
   - evaluate.evaluate_predictions(): accuracy/precision/recall/F1/
     ROC-AUC(binary)/cross_architecture_accuracy
   - writes test_metrics.json, test_predictions.npz,
     confusion_matrix.png, tsne.png, umap.png (≥3 samples)
```
**Duplication note:** there are **two parallel evaluation stacks**:
`xnerf/evaluation/{evaluate.py,test_after_training.py}` (used by the
pipeline orchestrators) and a second, structurally similar but independently
written `evaluation/{evaluate.py,metrics.py,reports.py}` at repo root (used
by nothing in `pipeline/kaggle_run.py` or `pipeline/local_run.py` — no
`import` of top-level `evaluation.*` exists outside `evaluation/evaluate.py`
itself). The root-level `evaluation/` package is **UNUSED** dead code
duplicating ~70% of `xnerf/evaluation/`'s functionality with minor
differences (it supports `family_accuracy`, writes `metrics.csv`, and has a
manifest+checkpoint CLI mode the other does not).
**Status:** `xnerf/evaluation/*` = IMPLEMENTED; `evaluation/*` (root) =
IMPLEMENTED-BUT-UNUSED/orphaned.

### 2.8 Inference Pipeline
Three independent inference entry points exist, all loading
`XNERFPlusPlus` and an exported checkpoint, but with non-identical
batch-construction logic:
1. **FastAPI service** — `xnerf/api/app.py` (`/upload`, `/analyze`,
   `/result/{id}`, `/health`). Builds a batch via a throwaway one-row
   `MalwareManifestDataset` manifest written to disk, then manually
   overwrites `batch["isr"]` after the fact (since the dataset's own `isr`
   logic only triggers for binary file suffixes).
2. **Local CLI** — `xnerf/deployment/local_analyze.py`
   (`python -m xnerf.deployment.local_analyze --checkpoint ... --sample ...`).
   Same throwaway-manifest pattern, also produces a PDF report and prints a
   JSON summary.
3. **Terminal sandbox** — `sandbox/sandbox.py` (entry via top-level
   `sandbox.py` shim or `python sandbox/sandbox.py <file>`). Independent,
   simpler `feature_extractor.py` that constructs `binary_image`,
   `memory_trace` (from raw byte values, not Capstone-derived features),
   zero-filled `api_ids`/`network_ids`, and the real ISR via
   `ArchitectureNormalizationPipeline`. Has its own checkpoint-resolution
   fallback chain (`sandbox/config.py::_resolve_checkpoint_path`) trying 7
   hardcoded candidate paths, and its own family-name resolution
   (`sandbox/inference.py::family_name`) trying 9 hardcoded sidecar JSON
   paths if the checkpoint lacks embedded `family_names` metadata.
**Status:** IMPLEMENTED but **triplicated** — three independent, partially
inconsistent re-implementations of "load checkpoint → build one-sample batch
→ forward → summarize" instead of one shared inference module.

### 2.9 Deployment Pipeline
```
Kaggle training ──▶ checkpoints/best.pt
        │ export_checkpoint() [xnerf/deployment/export_checkpoint.py]
        │  - strips "module." prefix (DataParallel)
        │  - resolves family_names from checkpoint metadata or train
        │    manifest fallback, validates count via
        │    validate_checkpoint_family_metadata()
        ▼
models/xnerf_local_inference.pt  (format tag "xnerf-local-inference-v1")
        │
        ├─ docker compose up --build  (xnerf/deployment/Dockerfile,
        │    python:3.11-slim, pip install -r requirements.txt,
        │    runs `uvicorn xnerf.api.app:app`)
        └─ local venv + `uvicorn xnerf.api.app:app --reload`
```
**Status:** IMPLEMENTED end-to-end at the code level. **Not independently
verified**: no exported checkpoint, no Docker build log, and no recorded
container run exist in the repository or its history.

### 2.10 Sandbox Pipeline (dynamic-report ingestion, *not* sample execution)
```
CAPE/Avast JSON report (.json or .zip of .json)
   ──▶ parse_cape_report() [xnerf/sandbox/cape_parser.py]
        - behavior.processes[].calls[].api          → api_calls[]
        - behavior.apistats[pid][api]=count          → api_calls[] (capped
                                                          at 5 repeats/api)
        - behavior.summary.resolved_apis              → api_calls[]
        - network.dns/http/tcp/udp/icmp/smtp/irc      → network_events[]
        - memory/procmemory/dropped/signatures/
          summary.{keys,files,mutexes,services}       → memory_events[]
        - behavior.processes[] (pid,name)             → process_events[]
   ──▶ tokens_to_ids() [hash-based stable tokenization, NOT a learned
        vocabulary] → api_ids[≤256], network_ids[≤256]
```
The README explicitly states: *"The framework treats raw binaries as
analysis inputs and does not execute samples."* — confirmed: there is no
sandbox execution engine, hypervisor integration, or dynamic monitoring
code anywhere in the repository. "Sandbox" here means **parsing
third-party CAPE/Avast sandbox report JSON**, not running a sandbox.
**Status:** IMPLEMENTED (the parser, well-tested in
`tests/test_cape_parser.py`) but the feature name is misleading relative to
what most readers would assume "sandbox pipeline" means.

---

## 3. Module-by-Module Contract Table

| Module | File | Purpose | Input | Output | Key Deps |
|---|---|---|---|---|---|
| `BinaryImageEncoder` | `encoders/binary_image.py` | Byte-image CNN encoder | `[B,1/3,H,W]` in [0,1] | `[B,512]` | torchvision ResNet18 (`weights=None`, conv1 modified to accept 3ch, fc→Identity) |
| `APIEncoder` | `encoders/api.py` | API-call-sequence transformer | `api_ids:[B,T]` LongTensor | `[B,512]` | `nn.TransformerEncoder` (4 layers, 8 heads, hidden 256), masked mean-pool, pad-row guard for all-pad rows |
| `NetworkEncoder` | `encoders/network.py` | Network-event transformer | `network_ids:[B,T]` | `[B,512]` | identical pattern to APIEncoder, 3 layers |
| `MemoryEncoder` | `encoders/memory.py` | Dilated TCN over memory/feature trace | `[B,T,8]` | `[B,512]` | 3×`Conv1d` dilations (1,2,4), `AdaptiveAvgPool1d` |
| `CFGEncoder` | `encoders/cfg.py` | Control/call-flow-graph encoder | `x:[N,4]`, `edge_index:[2,E]`, `batch:[N]` | `[B,512]` | `torch_geometric.GATConv` ×2 (4 heads), `global_mean_pool` |
| `SemanticFieldSynchronizer` (SFS) | `synchronization/sfs.py` | Cross-modal fusion + temporal expansion | dict of present `{512}`-dim modal embeddings | `[B,T,2048]` | per-modality `Linear` proj + learned type-embedding, `MultiheadAttention` (8 heads), bidirectional `GRU` |
| `MNEF` | `fields/mnef.py` | Continuous "execution field" `F(x,t,s,m,a)` | `x,t:[B,T,1]`, `s:[B,T,2048]`, `m:[B,T,512]`, `a:[B,T,64]` | `field:[B,T,1024]`, `behavior_logits:[B,T,5]` | sinusoidal `PositionalEncoding` (NeRF-style, 8 frequency bands), 3-layer MLP |
| `CrossArchitectureAligner` | `alignment/adversarial.py` | Gradient-reversal domain-adversarial arch alignment | `features:[B,2048]` | `aligned:[B,2048]`, `arch_logits:[B,6]` | custom `GradientReverse` autograd Function |
| `TrajectoryDecoder` | `renderer/trajectory_decoder.py` | Decode field into 5-stage attack trajectory + DiGraph | `field:[B,T,1024]` | `stage_logits:[B,T,5]`, `transition_logits:[B,T-1,5,5]` | `networkx.DiGraph` reconstruction |
| `XNERFPlusPlus` | `model.py` | End-to-end orchestrator of all of the above | multimodal batch dict | `malware_logits, family_logits, zero_shot_embedding, arch_logits, field, behavior_logits, stage_logits, transition_logits` | all modules above |
| `MalwareManifestDataset` | `datasets/loaders.py` | JSONL-manifest-backed multimodal dataset | manifest row dict | batch dict (see §2.6 inputs) | `family_cleaning`, `ontology`, `utils/io` |
| `XNerfTrainer` | `training/trainer.py` | Production train/val loop w/ NaN guarding, AMP, resume | model + datasets | checkpoints + metrics dict | `torch.cuda.amp`, `validation.validate_family_batch` |
| `ZeroShotPrototypeClassifier` | `zero_shot/prototypes.py` | Cosine-similarity prototype classifier | `embeddings:[B,2048]`, `prototypes:[K,2048]` | `logits/probabilities/prediction` | none beyond torch |
| `ReportGenerator` | `explainability/report_generator.py` | Analyst PDF/JSON report | model outputs + optional `nx.DiGraph` | dict summary + PDF | `reportlab` |

---

## 4. Full Model Graph (`XNERFPlusPlus.forward`, annotated)

```
batch{api_ids, memory_trace, network_ids, binary_image?, graph_x/edge_index/batch?, arch_id}
   │
   ├─ api_ids ───────────▶ APIEncoder ───────────▶ e_api    [B,512]
   ├─ memory_trace ──────▶ MemoryEncoder ─────────▶ e_mem    [B,512]
   ├─ network_ids ───────▶ NetworkEncoder ────────▶ e_net    [B,512]
   ├─ binary_image (opt) ▶ BinaryImageEncoder ─────▶ e_bin    [B,512]
   └─ graph_*    (opt)   ▶ CFGEncoder ─────────────▶ e_graph  [B,512]
                              │
        {e_api,e_mem,e_net,e_bin?,e_graph?} ──▶ SemanticFieldSynchronizer
                              │   (per-modality proj + type-embed → stack
                              │    → self-attention → mean → +time-embed
                              │    → FFN → bi-GRU)
                              ▼
                      semantic  [B, field_time=16, 2048]
                              │
              ┌───────────────┴────────────────────────┐
              │ pooled = semantic.mean(dim=1)  [B,2048] │
              ▼                                          ▼
   CrossArchitectureAligner(pooled)            coords = linspace(0,1,T)
   ├─ aligned        [B,2048]  (GELU+LN dense)  arch_emb = arch_embed(arch_id)
   └─ arch_logits    [B,6]   (via GRL+MLP)      mem_ctx  = memory_context(e_mem)
              │                                          │
   malware_head(aligned) → malware_logits [B,2]          │
   family_head(aligned)  → family_logits  [B,32]         │
   zero_shot_embedding = aligned          [B,2048]       │
                                                           ▼
                                    MNEF(coords, coords, semantic, mem_ctx, arch_emb)
                                       ├─ field           [B,16,1024]
                                       └─ behavior_logits [B,16,5]  (UNSUPERVISED — see §6)
                                                           │
                                          TrajectoryDecoder(field)
                                            ├─ stage_logits      [B,16,5]
                                            └─ transition_logits [B,15,5,5]
```
**Critical structural finding:** `aligned` (the domain-adversarially-aligned
features used for `malware_logits`/`family_logits`/`zero_shot_embedding`)
and `semantic`/`field` (the pathway feeding `MNEF` and `TrajectoryDecoder`)
are **two separate branches off the same `semantic` tensor** — the
adversarial alignment transform is *not* applied before the field/trajectory
branch. The "cross-architecture alignment" therefore only directly affects
classification and zero-shot embeddings, not the explainability/trajectory
output, despite the README implying a single unified pipeline.

---

## 5. Loss Function Wiring (actually executed, vs. defined-but-dead)

**Actually computed every training step** (`xnerf/training/losses.py
classification_losses`):
```
L_total = L_malware_ce
        + 0.1 · L_family_ce            (only if family_label in batch)
        + 0.1 · L_arch_adv             (CE(arch_logits, arch_id), through GRL)
        + 0.01 · L_field_smooth        (mean squared diff of consecutive
                                          field timesteps; unsupervised)
```
**Defined in code but never invoked by the training loop** (verified via
repo-wide grep, see CLAIM_VALIDATION.md for evidence):
- `MNEF.field_losses` (`behavior_ce` against `behavior_targets`) — no
  `behavior_targets` are ever produced by the dataset or passed to this
  function anywhere in the codebase.
- `CrossArchitectureAligner.losses` (`Ladv`, and especially `Lcrossarch` =
  `1 - cosine_similarity(paired_a, paired_b)`) — requires paired
  same-malware/different-architecture embeddings; **no pairing mechanism
  exists** in `MalwareManifestDataset` or `build_dataset.py` to construct
  such pairs.
- `SemanticFieldSynchronizer.contrastive_loss` — a NT-Xent-style InfoNCE
  loss between two modality embeddings; never called by `trainer.py` or any
  training script.

This means the model's **only object-level supervision** is malware/family
classification (+ a tiny weighted architecture-discrimination term and an
unsupervised smoothness regularizer). All the more "research-flavored"
losses (cross-architecture contrastive alignment, multi-modal contrastive
synchronization, behavior-stage supervision) are implemented as callable
functions but are **not wired into the training graph that actually runs**.

---

## 6. Baselines (declared, never executed)

`xnerf/baselines/models.py` defines five baseline architectures
(`CNNMalware`, `MalBERT`, `HYDRA`, `CrossArchitectureSiamese`,
`GNNMalware`) intended for comparison against `XNERFPlusPlus`. Repo-wide
search confirms **no training script, no evaluation script, and no config
file ever imports or instantiates any of these classes** outside their own
module and `__init__.py`. They are inert scaffolding for a future
comparison study, not executed baselines with results.

---

## 7. Configuration Surface

| File | Purpose | num_families | Notes |
|---|---|---|---|
| `config.yaml` (repo root) | Local quick-run | 32 | `debug_max_batches: 50`, `use_amp: false` |
| `config_balanced_92k.yaml` | Larger local run on a "balanced 92k" manifest variant | **221** | Filenames (`*_balanced_92k.jsonl`) imply a specific pre-built manifest not present in the repo (data is gitignored) |
| `xnerf/configs/default.yaml` | Library default | 32 | minimal |
| `xnerf/configs/kaggle.yaml` | Kaggle GPU run | 32 | hardcodes a specific Kaggle username path (`/kaggle/input/datasets/mayukh02/...`) — not portable to other Kaggle accounts without edits |
| `xnerf/configs/local_inference.yaml` | API/CLI inference | 32 | device defaults to CPU |
| `xnerf/configs/datasets.yaml` | Human-readable dataset registry | n/a | documentation only, not parsed by any loader |

The `num_families` mismatch (32 vs 221) across configs is evidence that the
family vocabulary is **dataset/run-dependent**, not a fixed architectural
constant — the model's `family_head` output width must match whichever
manifest's `family_vocab.json` was used to train it, and checkpoints embed
their own `family_names` list specifically to guard against this drift
(`xnerf/datasets/validation.py::validate_checkpoint_family_metadata`).
