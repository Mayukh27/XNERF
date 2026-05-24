# X-NERF++

Cross-Architecture Neural Execution Rendering Framework for defensive malware intelligence.

## What Is Included

- Architecture normalization for x86, x64, ARM, ARM64, MIPS, and RISC-V using Capstone.
- Multimodal encoders for binary images, CFGs, API traces, memory traces, and network events.
- Semantic Field Synchronizer producing `[batch,time,2048]`.
- Malware Neural Execution Field `F(x,t,s,m,a)`.
- Adversarial cross-architecture alignment.
- Neural trajectory renderer for Environment Check, Privilege Escalation, Persistence, Credential Access, and Exfiltration.
- Explainability PDF generation, baselines, training, evaluation, FastAPI, Docker, and Kaggle setup.

## Intended Workflow

Your PC does not need to train X-NERF++. Use it to prepare code, stage dataset archives, and run local inference/API after Kaggle training.

1. Put dataset archives in [data/archives](/c/Users/Mayukh/OneDrive/Documents/IEDC/data/archives).
2. Upload `data/archives` to Kaggle as a Dataset named `xnerf-malware-archives`.
3. Train on Kaggle with [xnerf/configs/kaggle.yaml](/c/Users/Mayukh/OneDrive/Documents/IEDC/xnerf/configs/kaggle.yaml).
4. Export `/kaggle/working/export/xnerf_local_inference.pt`.
5. Download that file to [models/xnerf_local_inference.pt](/c/Users/Mayukh/OneDrive/Documents/IEDC/models/README.md).
6. Run the local API or local CLI on your PC.

## Local Inference Quick Start

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
$env:XNERF_CHECKPOINT="models/xnerf_local_inference.pt"
uvicorn xnerf.api.app:app --reload
```

For one file without the API:

```powershell
python -m xnerf.deployment.local_analyze --checkpoint models/xnerf_local_inference.pt --sample path\to\sample.bin --arch x86
```

## Dataset Archive Locations

Put ZIP/TAR archives in these exact local folders before uploading them to Kaggle:

```text
data/
  archives/
    malnet_tiny/
      images/             MalNet Tiny image archives
      graphs/             MalNet Tiny graph archives
    AndMal2020/
      static/             AndMal2020 static CSV/features archives
      dynamic/            AndMal2020 dynamic before/after reboot archives
    cicmaldroid2020/      CICMalDroid2020 archives
    drebin/               Drebin feature archives
    ember/                EMBER feature archives
    virusshare/           authorized raw sample archives only
    cape/                 CAPE JSON report archives
      reports/            Avast/CAPE report ZIP/TAR files with .json reports
```

After upload to Kaggle, the expected paths are:

```text
/kaggle/input/xnerf-malware-archives/archives/malnet_tiny/
/kaggle/input/xnerf-malware-archives/archives/AndMal2020/
/kaggle/input/xnerf-malware-archives/archives/cicmaldroid2020/
/kaggle/input/xnerf-malware-archives/archives/drebin/
/kaggle/input/xnerf-malware-archives/archives/ember/
/kaggle/input/xnerf-malware-archives/archives/virusshare/
/kaggle/input/xnerf-malware-archives/archives/cape/
```

All dataset folders are optional. The code skips missing folders. Training only fails if the final extracted `raw/` directory contains zero usable files.

For a first working Kaggle run, you can use only:

```text
data/archives/malnet_tiny/images/
data/archives/malnet_tiny/graphs/
```

CICMalDroid2020 is not required for the first model. Add it later when you want to train the API/dynamic behavior branch with real Android traces.

AndMal2020 static and dynamic archives should be placed here:

```text
data/archives/AndMal2020/static/
data/archives/AndMal2020/dynamic/
```

Future datasets do not require extractor code changes. Add them as:

```text
data/archives/<dataset_name>/<modality_or_split>/*.zip
```

For the Avast/CAPE report dataset, use:

```text
data/archives/cape/reports/avast_reports.zip
```

The parser accepts CAPE-style JSON fields such as `behavior.processes[].calls`, `behavior.apistats`, and `network.dns/http/tcp/udp`. During manifest building, those reports become `api_ids` and `network_ids` for the dynamic encoders.

AndMal2020/CIC numeric CSVs are also supported. Headerless rows shaped like:

```text
sample_id, 0.12, 4, 8.5, ...
```

are converted into one training sample per row. The numeric feature vector is cached as a tensor and used as `memory_trace [512,8]`.

Headered CSVs are also supported. Known `id/hash/sha256`, `label/class/verdict`, and `family/category` columns are used as metadata; all other numeric columns become features. `public_labels.csv` is treated as metadata, not as a training sample file.

The recursive extractor preserves the relative path:

```text
data/archives/new_dataset/static/a.zip
  -> /kaggle/working/data/raw/new_dataset/static/

data/archives/new_dataset/dynamic/b.tar.gz
  -> /kaggle/working/data/raw/new_dataset/dynamic/
```

On Kaggle, extraction writes to:

```text
/kaggle/working/data/
  raw/
    malnet_tiny/
    AndMal2020/
    cicmaldroid2020/
    drebin/
    ember/
    virusshare/
    cape/
  cache/
    isr/
  processed/
    manifest.jsonl
```

VirusShare handling requires explicit authorization and `VIRUSSHARE_API_KEY`. The framework treats raw binaries as analysis inputs and does not execute samples.

## Kaggle Training Commands

Use the cells in [xnerf/notebooks/kaggle_setup.py](/c/Users/Mayukh/OneDrive/Documents/IEDC/xnerf/notebooks/kaggle_setup.py), or run:

```bash
pip install -q torch-geometric transformers fastapi uvicorn capstone networkx ray umap-learn reportlab
python -m xnerf.pipeline.kaggle_run --config xnerf/configs/kaggle.yaml
```

That single command performs extraction, train/val/test split creation, training, held-out testing, zero-shot prototype build/evaluation, and local checkpoint export.

The build step writes:

```text
/kaggle/working/data/processed/manifest.jsonl
/kaggle/working/data/processed/train_manifest.jsonl
/kaggle/working/data/processed/val_manifest.jsonl
/kaggle/working/data/processed/test_manifest.jsonl
```

After training/testing, Kaggle outputs:

```text
/kaggle/working/checkpoints/best.pt
/kaggle/working/runs/metrics.json
/kaggle/working/runs/test/test_metrics.json
/kaggle/working/runs/test/test_predictions.npz
/kaggle/working/runs/test/confusion_matrix.png
/kaggle/working/runs/test/tsne.png
/kaggle/working/runs/test/umap.png
/kaggle/working/runs/zero_shot/prototypes.pt
/kaggle/working/runs/zero_shot/zero_shot_metrics.json
/kaggle/working/runs/zero_shot/zero_shot_predictions.npz
/kaggle/working/export/xnerf_local_inference.pt
```

The one-command pipeline also copies the important files to one final output folder:

```text
/kaggle/working/xnerf_output/summary.json
/kaggle/working/xnerf_output/xnerf_local_inference.pt
/kaggle/working/xnerf_output/test_metrics.json
/kaggle/working/xnerf_output/zero_shot_metrics.json
/kaggle/working/xnerf_output/prototypes.pt
/kaggle/working/xnerf_output/confusion_matrix.png
/kaggle/working/xnerf_output/test_predictions.npz
/kaggle/working/xnerf_output/zero_shot_predictions.npz
```

## Zero-Shot Implementation

X-NERF++ exposes `zero_shot_embedding` from the trained model. The zero-shot path builds a prototype bank by averaging embeddings per family or behavior label, then classifies unseen samples by cosine similarity.

The one-command Kaggle pipeline runs this automatically. To run only zero-shot manually:

```bash
python -m xnerf.zero_shot.build_prototypes --config xnerf/configs/kaggle.yaml --checkpoint /kaggle/working/checkpoints/best.pt --manifest /kaggle/working/data/processed/train_manifest.jsonl --output /kaggle/working/runs/zero_shot/prototypes.pt
python -m xnerf.zero_shot.evaluate_zero_shot --config xnerf/configs/kaggle.yaml --checkpoint /kaggle/working/checkpoints/best.pt --manifest /kaggle/working/data/processed/test_manifest.jsonl --prototypes /kaggle/working/runs/zero_shot/prototypes.pt --out /kaggle/working/runs/zero_shot
```

This writes `zero_shot_accuracy` to:

```text
/kaggle/working/runs/zero_shot/zero_shot_metrics.json
```

## Module Contracts

All neural modules inherit `BaseModule`. All preprocessing classes inherit `Processor`. All training loops inherit `Trainer`. All datasets inherit `DatasetLoader`.

Each major module documents inputs, outputs, tensor dimensions, forward signature, and a usage example in its class docstring.

## API

```bash
docker compose up --build
curl -F "file=@sample.bin" http://localhost:8000/upload
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d "{\"upload_id\":\"...\",\"arch\":\"x86\"}"
curl http://localhost:8000/result/<id>
```

## Free Training Target

Use `xnerf/notebooks/kaggle_setup.py` as notebook cells on a free Kaggle T4 GPU. Keep batch size small and use gradient accumulation.

## Tests

Parser and dataset tests:

```bash
pytest tests/test_cape_parser.py tests/test_build_dataset.py
pytest tests/test_feature_csv.py
```
