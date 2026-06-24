# AUDIT_FIX_REPORT.md
## X-NERF++ Repository Audit Fix Report

Date: 2026-06-24

## 1. Model Architecture Audit

| Issue | File | Class/function | Severity | Fix applied |
|---|---|---|---|---|
| ISR tensors were present in batches but not consumed by the model. | `xnerf/model.py` | `XNERFPlusPlus.forward` | Critical | Added `ISREncoder` and wired `batch["isr"]` into fusion. |
| Graph embeddings used key `graph`, but SFS accepted `cfg`; graphs were effectively dropped. | `xnerf/model.py` | `XNERFPlusPlus.forward` | Critical | Changed graph embedding key to `cfg`. |
| SFS did not list ISR as a valid modality. | `xnerf/synchronization/sfs.py` | `SemanticFieldSynchronizer` | High | Added `isr` to `self.modalities`. |

## 2. Dataset Audit

| Issue | File | Class/function | Severity | Fix applied |
|---|---|---|---|---|
| Malware-family placeholders could become family-class targets. | `xnerf/datasets/loaders.py` | `MalwareManifestDataset.__getitem__` | Critical | Placeholder malware families now emit `family_label=-1`. |
| `num_families` is manifest-dependent. | configs | n/a | High | Documented in README/project context; no hard-coded universal value claimed. |

## 3. Training Pipeline Audit

The trainer still calls `classification_losses` from `xnerf/training/losses.py`. Family CE now ignores invalid labels. Runtime training smoke tests were not completed in this environment because `pytest` and `torch` are unavailable in the usable Python runtime.

## 4. Loss Function Audit

Live loss:

```text
malware_ce + 0.1 * masked_family_ce + 0.1 * arch_adv + 0.01 * field_smooth
```

Fixed: invalid family labels are ignored with `ignore_index=-1`.

Not fixed: behavior CE, stage CE, transition CE, contrastive SFS loss, and paired `Lcrossarch` are still not active because required targets/pairs are absent.

## 5. Family Classification Audit

Implemented practical `VALID_FAMILY_FILTER` behavior:

- Benign samples can keep `benign`.
- Malware samples whose family value is a dataset/source placeholder receive `family_label=-1`.
- Loss ignores those samples for family CE while still using them for malware CE.

## 6. Cross-Architecture Audit

The adversarial architecture discriminator remains live through `arch_adv`. Paired cross-architecture alignment remains not implemented because same-malware cross-ISA pairs are not produced by the dataset.

## 7. ISR Pipeline Audit

Fixed. Current path:

```text
isr_path -> MalwareManifestDataset["isr"] -> collate_dicts -> batch["isr"]
-> ISREncoder -> SFS modality "isr" -> semantic -> malware/family heads
```

## 8. Graph Pipeline Audit

Fixed key mismatch. Current path:

```text
.edgelist -> graph_x/graph_edge_index -> collate_dicts -> CFGEncoder
-> SFS modality "cfg" -> semantic -> malware/family heads
```

## 9. MNEF Audit

MNEF remains connected and produces `field` and `behavior_logits`. It is not behavior-supervised. The trajectory outputs must be treated as architectural outputs, not validated attack-stage predictions.

## 10. Dead Code Audit

Still open:

- Root-level `evaluation/` package is separate from `xnerf/evaluation/`.
- `evaluate_zero_shot-2.py` is not the wired evaluator.
- Baseline model classes are defined but not trained by the pipeline.

## Verification

Passed:

- AST syntax parse for edited Python files.

Blocked:

- `pytest`: command not available in `.venv`; bundled Python lacks `pytest`.
- Synthetic forward/backward: bundled Python lacks `torch`.

