# CLAIM_VALIDATION.md
## X-NERF++ — Strict Claim Validation

Methodology: every claim is checked against source code, test coverage, and
repository artifacts (commit history, `.gitignore`, presence/absence of
checkpoints, manifests, metrics files). Where a `runs/`, `checkpoints/`, or
`models/*.pt` file would be needed as proof and is absent, the claim is
placed in **DO NOT CLAIM** regardless of how complete the surrounding code
looks, per the audit's "strict" instruction.

**Ground-truth check performed:** `git log` shows 38 commits, all from a
single contributor, with debugging-style messages ("Fixed zero shot",
"Fixed build_dataset", "validate fix", "fixed pipeline"). `.gitignore`
excludes `data/raw/`, `data/cache/`, `data/processed/`, `runs/`,
`checkpoints/`, `models/*.pt`, `models/*.pth`. **No trained checkpoint,
metrics JSON, confusion matrix, prototype bank, or exported
`xnerf_local_inference.pt` exists anywhere in the repository or its
history.** Therefore **no experimental number in this audit's other
documents is sourced from an actual run** — all such numbers are explicitly
marked as placeholders.

---

## SAFE TO CLAIM

These are implemented, exercised by passing unit tests, and verifiable by
inspection/by anyone who clones the repo and runs `pytest`, independent of
whether a full training run has ever occurred:

1. The repository implements a configurable, multi-format dataset-ingestion
   pipeline that converts heterogeneous malware-research data formats
   (headerless/headered numeric feature CSV, API-call-sequence CSV/TXT,
   Parquet, CAPE/Avast JSON sandbox reports) into one unified JSONL
   manifest schema, with deterministic train/val/test splitting (stratified
   or hash-based, resumable). — Evidence: `xnerf/datasets/build_dataset.py`,
   `tests/test_build_dataset.py`, `tests/test_feature_csv.py`.
2. A CAPE/Avast-style sandbox JSON report parser extracts API calls,
   network events, memory/registry/file events, and process metadata into a
   normalized schema, including from JSON files nested inside ZIP archives.
   — Evidence: `xnerf/sandbox/cape_parser.py`, `tests/test_cape_parser.py`
   (both pass and are deterministic, no GPU/network required).
3. A configurable family-name normalization layer canonicalizes
   heterogeneous malware-family strings (aliases, prefix matching, benign
   marker detection, dataset-name-as-family placeholder filtering) into a
   stable, sorted vocabulary with `benign` and `unknown` pinned to
   fixed indices. — Evidence: `xnerf/datasets/family_cleaning.py`,
   `tests/test_family_normalization.py`.
4. A multimodal neural architecture (`XNERFPlusPlus`) is implemented and is
   constructible/forward-callable on synthetic tensors of the documented
   shapes, combining: a ResNet18-based byte-image encoder, a Transformer
   API-call encoder, a Transformer network-event encoder, a dilated-Conv1d
   memory-trace encoder, a Graph-Attention-Network CFG encoder, an 8-head
   cross-modal attention + bidirectional-GRU temporal synchronizer, a
   NeRF-style positional-encoding continuous "execution field" MLP, a
   gradient-reversal-layer domain-adversarial architecture aligner, and a
   stage/transition decoder that reconstructs `networkx.DiGraph` objects.
   — Evidence: `xnerf/model.py` + each cited submodule file.
5. The training loop implements mixed-precision training, gradient
   accumulation, gradient clipping, early stopping, resumable checkpoints
   (optimizer + AMP scaler + epoch + best-loss state), and explicit
   non-finite (NaN/Inf) detection at the model-output, loss-term, gradient,
   post-optimizer-step parameter, and optimizer-state levels, raising a
   descriptive `RuntimeError` with batch diagnostics on failure. — Evidence:
   `xnerf/training/trainer.py` (read directly; this is unusually thorough
   defensive engineering relative to typical research code and is a
   legitimate **engineering** contribution regardless of whether the model
   has been trained to convergence).
6. A deterministic cache-path scheme allows feature/ISR tensors to be
   computed once and reused across multiple manifests (e.g., a balanced
   subsample manifest reusing a parent manifest's cache) without
   recomputation, verified by a unit test that asserts cache-hit counts.
   — Evidence: `xnerf/datasets/build_dataset.py::generate_cache_from_manifest`,
   `tests/test_feature_csv.py::test_split_manifest_reuses_parent_generated_cache`.
7. An architecture-disassembly normalization pipeline (Capstone-based)
   converts raw bytes for six instruction set architectures (x86, x64, ARM,
   ARM64, MIPS, RISC-V) into a common 4-column "Intermediate Semantic
   Representation" tensor via a hand-built ~45-mnemonic semantic ontology.
   — Evidence: `xnerf/preprocessing/{disassembler,semantic_mapper,
   isr_builder,ontology,pipeline}.py`. (Claim is scoped to *the ISR
   computation itself*; see DO NOT CLAIM #3 for what this does *not* imply
   about the trained model.)
8. A zero-shot classification mechanism exists: family-averaged prototype
   vectors are built from embeddings and unseen samples are classified by
   cosine similarity against the prototype bank, with top-1/top-5 accuracy
   and per-family breakdown reporting implemented in the more recent
   evaluator variant. — Evidence: `xnerf/zero_shot/prototypes.py`,
   `xnerf/zero_shot/build_prototypes.py`,
   `xnerf/zero_shot/evaluate_zero_shot-2.py`. (Claim is scoped to *the
   mechanism being implemented and internally consistent*, not to any
   reported accuracy number — none exists.)
9. A FastAPI inference service, a local CLI analyzer, and a terminal
   "sandbox" CLI all independently implement load-checkpoint →
   build-one-sample-batch → forward → summarize-and-PDF-report, and a
   Docker Compose + Dockerfile definition exists to containerize the API
   service. — Evidence: `xnerf/api/app.py`,
   `xnerf/deployment/local_analyze.py`, `sandbox/sandbox.py`,
   `xnerf/deployment/Dockerfile`, `docker-compose.yml`.
10. The codebase enforces consistent module contracts (`BaseModule`,
    `Processor`, `Trainer`, `DatasetLoader` base classes) across all neural,
    preprocessing, training, and dataset components, with documented
    input/output tensor shapes in every class docstring. — Evidence:
    `xnerf/utils/base.py` and consistent inheritance throughout.

---

## DO NOT CLAIM

These appear in the README, code comments, or model architecture, but are
**not** scientifically or experimentally supportable from this repository as
it stands. A paper draft must not state or imply any of the following
without first generating the missing evidence.

1. **Any accuracy, F1, ROC-AUC, zero-shot accuracy, or
   cross-architecture-accuracy number.** No checkpoint, metrics JSON, or
   prediction `.npz` exists in the repository. Any number quoted in
   `PROJECT_CONTEXT.md`/`PAPER_IDEAS.md` is explicitly labeled
   `[PLACEHOLDER — NOT YET MEASURED]` and must not be treated as real until
   training is actually run and the resulting `runs/test/test_metrics.json`
   is generated and inspected.
2. **"Adversarial cross-architecture alignment" as a trained, validated
   mechanism.** The gradient-reversal discriminator (`arch_logits`) *is*
   wired into the live training loss (`arch_adv`, weight 0.1), so the model
   does receive *some* domain-adversarial signal — that part is real. But
   the more scientifically meaningful **paired cross-architecture
   consistency loss** (`CrossArchitectureAligner.losses`'s `Lcrossarch`,
   `1 − cosine_similarity(paired_a, paired_b)`) is defined but **never
   invoked anywhere in the training code**, and no mechanism exists in
   `MalwareManifestDataset` or `build_dataset.py` to construct the
   "same malware sample on two different architectures" pairs this loss
   would require. Do not claim the system performs *paired* cross-arch
   representation alignment — only single-feature domain-adversarial
   regularization is actually trained.
3. **"Neural trajectory renderer for Environment Check, Privilege
   Escalation, Persistence, Credential Access, and Exfiltration" as a
   working, supervised behavior-stage classifier.** `TrajectoryDecoder`'s
   `stage_logits`/`transition_logits` and `MNEF`'s `behavior_logits` are
   computed every forward pass and consumed by the explainability report,
   the API, and the local CLI to produce stage labels and a behavior graph
   — but **no loss term in `classification_losses` (the function actually
   called by the trainer) ever supervises these heads.**
   `MNEF.field_losses`'s `behavior_ce` term is defined but never called,
   and no dataset in the ingestion pipeline produces `behavior_targets`
   (ground-truth attack-stage labels) at all. Practically, these heads are
   trained *only indirectly*, through shared upstream encoder weights and a
   0.01-weighted unsupervised temporal-smoothness term — there is no
   mechanism by which `stage_head`'s five output classes would come to mean
   "Environment Check" vs. "Exfiltration" specifically. **Do not claim the
   five-stage trajectory output is behaviorally meaningful or validated**;
   at best it is an untrained/weakly-regularized structural placeholder.
4. **"Multimodal encoders for binary images, CFGs, API traces, memory
   traces, and network events" as jointly demonstrated on real,
   verified multimodal samples.** The five encoders are each independently
   implemented and unit-testable on synthetic tensors, but no dataset
   currently present in `data/archives/` (or referenced by a populated
   manifest) provides more than one or two of these modalities
   *simultaneously* for the same sample — most ingestion paths
   (`process_feature_csv`, `process_feature_parquet`,
   `process_api_sequence_csv/txt`) populate `memory_trace` or `api_ids`
   alone, with `binary_image` left at its default zero tensor
   (`MalwareManifestDataset.__getitem__`: `binary_image` is explicitly
   zeroed when `data_type in {"feature_csv","feature_parquet"}`) and
   `graph_x`/`graph_edge_index` populated only for `.edgelist` paths. Do not
   claim a demonstrated five-modality fusion result; claim only that the
   *architecture* supports up to five modalities per sample when present.
5. **The ISR tensor (the actual Capstone-based, six-architecture
   disassembly/semantic-ontology representation) contributing to model
   predictions.** It is computed, cached, and present in the batch
   dictionary (`batch["isr"]`), but `XNERFPlusPlus.forward()` never reads
   that key. The "architecture normalization" headline feature of the
   README is therefore currently **disconnected from the trained model's
   decision-making**, even though the disassembly/ontology code itself
   works in isolation.
6. **Six-architecture, cross-platform generalization as a property of the
   trained model.** Architecture is inferred from filename substrings
   (defaulting to `x86`) for the vast majority of supported (feature-vector)
   datasets, which carry no genuine multi-ISA binary content. Until a
   dataset with verified non-x86 binaries is ingested and a held-out
   cross-architecture test is run and reported, "cross-architecture"
   results are unverified.
7. **Sandbox dynamic execution / dynamic analysis as performed by this
   framework.** "Sandbox" here means *parsing existing third-party CAPE/
   Avast report JSON*, not executing or monitoring samples. The README
   itself states the framework does not execute samples — do not let a
   paper draft imply otherwise via the word "sandbox" alone.
8. **Baseline comparisons (CNN, MalBERT, HYDRA, Siamese, GNN) as run or
   reported.** These five classes exist in `xnerf/baselines/models.py` but
   are never instantiated, trained, or evaluated anywhere in the codebase.
   Any "X-NERF++ outperforms CNN/Transformer/XGBoost baselines" claim is
   presently **unfounded** — the comparison has not been executed.
9. **The contrastive cross-modal synchronization loss
   (`SemanticFieldSynchronizer.contrastive_loss`) as part of training.**
   Defined, never called. Do not claim a contrastive multimodal
   pretraining objective is in effect.
10. **Hyperparameter search / Ray Tune results.** `ray_train.py` wraps
    training in a `tune.Tuner` but defines no actual search space (only the
    config path is varied) — there is no tuning study to report.
11. **Docker/API deployment having been run successfully.** The Dockerfile
    and compose file are syntactically plausible but no build log, container
    run, or integration test exists confirming the image builds or the
    `/analyze` endpoint functions against a real checkpoint.
12. **Any specific dataset's sample counts, class balance, or family
    distribution.** `data/archives/` contains almost no real data at audit
    time (two non-empty files: `drebin.zip`, `MalBehavD-V1-dataset.csv`,
    neither extracted/profiled in this audit) and no manifest exists in the
    repository. Any number describing dataset size/composition is
    presently invented and must be replaced once a real manifest is built.

---

## Borderline / Requires Explicit Caveat If Mentioned

- The **IoT-specific stochastic downsampling logic** for
  `CIC-YNU_IoTMal` (`IOTMAL_FAMILY_SAMPLE_PROBS`) is real, implemented, and
  affects manifest construction — but it is **not documented anywhere**
  (README, docstring, or config), is keyed to a single hardcoded dataset
  name string match, and has a fixed random seed (`IOTMAL_RANDOM_SEED =
  1337`) set as a **module-level side effect on import** of
  `build_dataset.py`, which is poor practice (it silently reseeds Python's
  global `random` state for any other code that imports this module). If
  discussed in a paper, this must be described as an ad hoc class-balancing
  heuristic for one specific dataset, not a general sampling strategy.
- The **duplicate evaluation stacks** (`evaluation/` at repo root vs.
  `xnerf/evaluation/`) and **duplicate zero-shot evaluators**
  (`evaluate_zero_shot.py` vs. `evaluate_zero_shot-2.py`, the latter adding
  benign-skipping, L2-normalization, and per-family/top-5 metrics) suggest
  in-progress, not-yet-consolidated iteration. Only the variant actually
  invoked by `xnerf/pipeline/{kaggle_run,local_run}.py` —
  `xnerf/evaluation/*` and `evaluate_zero_shot.py` (without the `-2`
  suffix) — should be treated as the "production" path in a paper; the
  others should be described as superseded drafts, not parallel validated
  alternatives.
