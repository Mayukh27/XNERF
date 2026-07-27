# Baseline Training and Testing Guide

This folder documents how to collect baseline numbers for the X-NERF++ paper.
Baseline results must be produced on the same train/validation/test manifests
as X-NERF++ before they are used in a comparison table.

## 1. Use the Same Split

The publication config currently points to these manifests:

```text
data/processed/train_publication_v2_50k.jsonl
data/processed/val_publication_v2_50k.jsonl
data/processed/test_publication_v2_50k.jsonl
```

Every baseline command below uses those files by default. Do not compare
against numbers from a different split unless the paper clearly labels them as
external reported results.

## 2. Train and Test EMBER RF

This is the first baseline to run because it is fast and gives a strong static
feature reference.

```bash
python -m xnerf.baselines.train_eval \
  --baseline ember-rf \
  --seed 42 \
  --out-dir runs/baselines/ember-rf/seed_42
```

Output:

```text
runs/baselines/ember-rf/seed_42/model.pkl
runs/baselines/ember-rf/seed_42/metrics.json
```

The `metrics.json` file contains Accuracy, Precision, Recall, F1, ROC-AUC,
cross-architecture accuracy, and per-architecture accuracy.

## 3. Train and Test MalConv

MalConv is the raw-byte CNN baseline. It uses rows whose manifest entries point
to binary-like files rather than CSV/API-only rows.

```bash
python -m xnerf.baselines.train_eval \
  --baseline malconv \
  --seed 42 \
  --batch-size 16 \
  --epochs 5 \
  --out-dir runs/baselines/malconv/seed_42
```

## 4. Train and Test MalBERT

MalBERT is the API-sequence Transformer baseline. It uses only rows containing
`api_ids`.

```bash
python -m xnerf.baselines.train_eval \
  --baseline malbert \
  --seed 42 \
  --batch-size 16 \
  --epochs 5 \
  --out-dir runs/baselines/malbert/seed_42
```

## 5. Train and Test HYDRA-Style Fusion

This baseline combines binary image and API-sequence inputs. If this is not an
exact reproduction of a published HYDRA implementation, report it in the paper
as `HYDRA-style static-dynamic fusion`.

```bash
python -m xnerf.baselines.train_eval \
  --baseline hydra \
  --seed 42 \
  --batch-size 8 \
  --epochs 5 \
  --out-dir runs/baselines/hydra/seed_42
```

## 6. Train and Test GNN Malware

This baseline uses graph-only rows whose paths end in `.edgelist`.

```bash
python -m xnerf.baselines.train_eval \
  --baseline gnn-malware \
  --seed 42 \
  --batch-size 8 \
  --epochs 5 \
  --out-dir runs/baselines/gnn-malware/seed_42
```

## 7. Run Three Seeds for Publication

For the paper, run each baseline with at least three seeds:

```bash
python -m xnerf.baselines.train_eval --baseline ember-rf --seed 42
python -m xnerf.baselines.train_eval --baseline ember-rf --seed 123
python -m xnerf.baselines.train_eval --baseline ember-rf --seed 2026
```

Repeat for each baseline. Report the mean and standard deviation in the paper:

```latex
EMBER RF & 93.5 $\pm$ 0.3 & 92.0 $\pm$ 0.4 & ...
```

## 8. Generate the Paper Table

After the baseline runs finish, generate a LaTeX table from the saved JSON
files:

```bash
python -m xnerf.baselines.summarize_results \
  --root runs/baselines \
  --out runs/baselines/summary.json
```

This prints a LaTeX table and stores the same output in
`runs/baselines/summary.json`.

## 9. Fairness Rules for the Paper

- Use the same train/validation/test manifests for X-NERF++ and all baselines.
- Use the same metric function for every model.
- Save the raw JSON result file for each run.
- Report whether a baseline used all rows or only rows with the required
  modality. The `metrics.json` file records `rows_total`, `rows_used`, and
  skipped rows.
- Do not present copied numbers from other papers as direct comparisons. Mark
  them as external reported results if you include them.

## 10. Paper Method Text

Use this wording in the experiment section:

```latex
For fair comparison, all baselines were trained and evaluated on the same
deterministic train/validation/test split as X-NERF++. Static baselines use
raw bytes or cached tabular PE/dynamic features, dynamic baselines use API-call
sequences, graph baselines use CFG-derived graph representations, and hybrid
baselines combine static and dynamic modalities. All methods are evaluated on
the same held-out test set using Accuracy, Precision, Recall, F1, ROC-AUC,
cross-architecture accuracy, and per-architecture accuracy.
```
