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
python -m xnerf.datasets.extract_archives --archive-root /kaggle/input/xnerf-malware-archives/archives --data-root /kaggle/working/data
python -m xnerf.datasets.build_dataset --root /kaggle/working/data --out /kaggle/working/data/processed/manifest.jsonl
python -m xnerf.training.train --config xnerf/configs/kaggle.yaml
python -m xnerf.deployment.export_checkpoint --input /kaggle/working/checkpoints/best.pt --config xnerf/configs/kaggle.yaml --output /kaggle/working/export/xnerf_local_inference.pt
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
