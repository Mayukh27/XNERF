# PAPER_IDEAS.md
## X-NERF++ — Ranked Paper-Worthy Ideas Discovered in the Codebase

Each idea lists what already exists in code (so reviewers/co-authors can
verify it isn't vaporware) vs. what experimental work remains. Publication
Readiness (%) reflects: code exists (25%) + wired into live training (25%)
+ real dataset assembled (25%) + at least one completed baseline-compared
experiment (25%). Since no experiment has been run anywhere in this
repository (see CLAIM_VALIDATION.md), **no idea below currently exceeds
50%** Publication Readiness; all percentages reflect *codebase* readiness,
not paper-writing readiness.

---

### Idea #1 — A Unified, Format-Agnostic Ingestion and Reproducible-Manifest Framework for Heterogeneous Public Malware Datasets

- **Track fit:** Tooling/Resource/Reproducibility track (IEEE S&P
  workshops, ACM TOPS artifacts, USENIX Security artifact evaluation, or
  a "Resource Paper" track at a security/ML venue).
- **Novelty:** Low-to-medium as a method; **medium-high as a resource**.
- **What already exists:** A tested ingestion engine handling ≥6 distinct
  dataset schema families (feature CSV headerless/headered, API-sequence
  CSV/TXT, Parquet, CAPE/Avast JSON), deterministic stratified/hash
  splitting, sharded reusable tensor caching, and a documented
  zero-code-change extension convention for new datasets.
  (`xnerf/datasets/build_dataset.py`, 8 passing unit tests.)
- **Required baselines:** Comparison against manual per-dataset
  preprocessing scripts (time-to-integrate a new dataset, lines of code
  per dataset, schema-detection accuracy).
- **Required experiments:**
  1. Integrate ≥5 real public datasets (AndMal2020, CICMalDroid2020,
     Drebin, EMBER, MalBehavD-V1) end-to-end and report ingestion success
     rate, parse-time, and resulting manifest statistics.
  2. Inter-annotator-style validation of the family-normalization
     placeholder-filtering logic against a hand-labeled sample of raw
     family strings.
  3. Ablation: hash-mode vs. stratified-mode split stability across reruns.
- **Publication Readiness: 45%.** Code is the most mature artifact in the
  repo; missing entirely is (a) actually acquiring and running the ingestion
  over the real datasets at scale, (b) any quantitative report of the
  resulting manifest statistics.

---

### Idea #2 — Malware Neural Execution Fields: Continuous, NeRF-Inspired Representations for Cross-Architecture Behavioral Malware Analysis

- **Track fit:** Main research track, ML/security venue (e.g., IEEE
  TDSC, ACM CCS, NDSS workshops, or an IEEE conference such as IEEE
  TrustCom / IEEE Security & Privacy workshops; Springer LNCS proceedings
  for a regional security/AI conference are also a realistic fit given the
  proof-of-concept maturity level).
- **Novelty:** High (Category D per NOVELTY_ANALYSIS.md #1) — applying
  continuous neural-field / NeRF-style positional encoding to malware
  execution representation is, to the best of this audit's literature
  comparison, an unusual and underexplored framing.
- **What already exists:** `MNEF` module (positionally-encoded
  `x,t,s,m,a` → continuous field + behavior logits), `TrajectoryDecoder`
  (field → discrete stage/transition graph), full forward-pass integration
  in `XNERFPlusPlus`.
- **Required baselines:** (a) Direct sequence classifier (Transformer over
  API/network tokens, no field formulation) — `APIEncoder`/`NetworkEncoder`
  + linear head, already present as raw components; (b) `xnerf/baselines/
  models.py::MalBERT`/`HYDRA`; (c) non-continuous (discrete per-timestep
  MLP, no positional encoding) ablation of MNEF itself.
- **Required experiments:**
  1. **Construct a labeled attack-stage dataset** by rule-matching CAPE/
     Avast `signatures[]`, `summary.{started_services,write_keys,...}`
     fields (already extracted by `cape_parser.py`) against a MITRE
     ATT&CK tactic taxonomy, to finally supervise `behavior_ce`.
  2. Wire `MNEF.field_losses`'s `behavior_ce` into `classification_losses`
     and retrain.
  3. Ablate: field-based (MNEF) vs. flat-MLP vs. no-field behavior
     prediction, holding encoders/SFS fixed.
  4. Report malware/family classification accuracy, behavior-stage
     macro-F1, and qualitative trajectory-graph case studies.
- **Required ground-truth/labels gap:** This is the single largest blocker
  — no attack-stage labels currently exist anywhere in the pipeline.
- **Publication Readiness: 30%.** Architecture exists and is forward-
  correct; training integration and labeled data are both entirely
  missing.

---

### Idea #3 — Domain-Adversarial Architecture-Invariant Malware Representations: A Cross-ISA Transfer Study

- **Track fit:** Main research track, security/ML venue (e.g., IEEE
  TIFS, RAID, DIMVA, or an IEEE/Springer conference focused on IoT/embedded
  security given the natural MIPS/ARM IoT-malware angle).
- **Novelty:** Medium (GRL mechanism is Category A; the cross-ISA
  malware application is the novel part, Category C/D depending on
  execution).
- **What already exists:** `CrossArchitectureAligner` (GRL +
  discriminator, live in training), `Lcrossarch` paired-cosine loss
  (defined, dead), 6-architecture Capstone ISR pipeline
  (`ArchitectureNormalizationPipeline`), `xnerf/datasets/audit.py`'s
  single-architecture-dominance detector (a useful diagnostic already
  built for exactly this study).
- **Required baselines:** (a) No-adversarial-alignment ablation (train
  with `arch_adv` loss weight = 0); (b) `CrossArchitectureSiamese`
  baseline already stubbed in `xnerf/baselines/models.py`; (c) naive
  feature concatenation with architecture one-hot (no alignment at all).
- **Required experiments:**
  1. **Solve the paired-sample construction problem**: identify or build a
     corpus of the *same* malware family compiled/observed across ≥2
     architectures (realistic candidates: Mirai/Gafgyt IoT botnet variants
     spanning x86/ARM/MIPS, commonly available in IoT-malware research
     datasets) to finally exercise `Lcrossarch`.
  2. Train with vs. without `Lcrossarch` and report cross-architecture
     family-transfer accuracy (train on arch A, test on arch B family
     classification).
  3. t-SNE/UMAP visualization of aligned vs. unaligned embeddings colored
     by architecture (infrastructure for this — `save_tsne`/`save_umap` —
     already exists in `xnerf/evaluation/evaluate.py`).
- **Publication Readiness: 25%.** The easy half (single-feature domain-
  adversarial training) is wired and trainable today; the scientifically
  interesting half (paired cross-arch alignment) requires new data-
  collection work, not just code.

---

### Idea #4 — Zero-Shot and Few-Shot Generalization to Unseen Malware Families via Prototype-Based Metric Learning

- **Track fit:** Main research track or workshop (e.g., a malware-
  detection workshop co-located with an IEEE/ACM security conference);
  this is the most "standard ML methodology, applied carefully" idea of
  the set, suiting a solid applied-ML venue or a Springer LNCS workshop.
- **Novelty:** Low-medium mechanism (Category A, prototypical networks);
  medium value from the *evaluation* (which families transfer, which
  don't, why).
- **What already exists:** `ZeroShotPrototypeClassifier`,
  `build_family_prototypes`, two generations of the zero-shot evaluator
  (the `-2` variant adds top-5 accuracy, per-family breakdown, benign-
  exclusion, and L2-normalization fixes that materially improve evaluation
  correctness).
- **Required baselines:** (a) Random/majority-class baseline; (b) k-NN
  over raw (non-aligned) features; (c) standard linear-probe classifier
  retrained per family (upper-bound "non-zero-shot" comparison).
- **Required experiments:**
  1. Consolidate on the `-2` evaluator's corrected methodology, wire it
     into the pipeline orchestrators (currently the non-`-2` is used).
  2. Held-out-family protocol: remove K families entirely from training,
     build prototypes from remaining-family embeddings, evaluate
     zero-shot accuracy on the held-out K.
  3. Per-family breakdown analysis correlated with family sample count
     and modality availability (the `-2` evaluator already outputs this
     breakdown).
- **Publication Readiness: 35%.** Mechanism and (most) evaluation
  machinery are implemented; missing is real data, a proper held-out-
  family protocol (current code does not appear to enforce "prototype
  families are disjoint from any family seen during representation
  training," which would need explicit experimental design), and any
  completed run.

---

### Idea #5 — A Cross-Modal Attention Fusion Study for Static + Dynamic Multimodal Malware Representations

- **Track fit:** Workshop or short/applied paper at an ML-for-security
  venue.
- **Novelty:** Low (Category B, engineering synthesis of known fusion
  techniques) — positioned as an ablation/empirical study rather than a
  new method.
- **What already exists:** `SemanticFieldSynchronizer` (attention fusion,
  live in training); `HYDRA` baseline class (simple concatenation fusion,
  unused); dead `contrastive_loss` method that could be revived as an
  additional fusion-training objective.
- **Required baselines:** Late-fusion voting; simple concatenation
  (`HYDRA`); attention fusion without temporal GRU; attention fusion with
  the (currently dead) contrastive loss enabled.
- **Required experiments:** Controlled ablation across fusion strategies
  holding encoders fixed, on a manifest with genuinely overlapping
  multimodal samples (a prerequisite not currently met — see
  CLAIM_VALIDATION.md item 4).
- **Publication Readiness: 20%.** Lowest readiness of the set because it
  requires the hardest *data* precondition (real per-sample multimodal
  overlap across ≥3 modalities), which nothing in the current dataset
  inventory (DATASET_TABLE.md) actually provides.

---

## Ranked Summary

| Rank | Idea | Novelty | Readiness | Primary Blocker |
|---|---|---|---|---|
| 1 | Multi-Format Dataset Ingestion Resource Paper | Low/Med | 45% | Run at scale on real data; report stats |
| 2 | Zero-Shot Prototype Generalization Study | Low/Med | 35% | Held-out-family protocol + real run |
| 3 | Malware Neural Execution Fields (MNEF) | High | 30% | Attack-stage labels + loss wiring |
| 4 | Domain-Adversarial Cross-ISA Alignment | Med | 25% | Paired multi-arch sample corpus |
| 5 | Cross-Modal Fusion Ablation Study | Low | 20% | Genuine multimodal-overlap dataset |

**Recommendation for fastest credible publication:** pursue **Idea #1**
first (lowest data risk, highest current maturity, suitable for a
resource/tooling track), using it as the foundation/artifact description
for a subsequent **Idea #3 or #4** main-track submission once real data and
at least one completed training run exist.
