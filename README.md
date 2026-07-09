<div align="center">

# X-NERF++

**Cross-Architecture Neural Execution Rendering Framework**

*Unified multi-modal malware representation learning across CPU architectures*

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)

</div>

---

## Overview

X-NERF++ is a research framework for malware analysis that learns a single, **architecture-invariant** representation of program behavior from six heterogeneous input modalities — binary images, API call sequences, control-flow graphs, memory traces, network events, and an intermediate semantic representation (ISR).

These modalities are fused through a **Semantic Field Synchronizer**, aligned across instruction-set architectures with an adversarial **Cross-Architecture Aligner**, and projected into a continuous **Malware Neural Execution Field (MNEF)** that supports detection, family attribution, cross-architecture recognition, zero-shot recognition, and similarity retrieval from one shared latent space.

## Architecture

<div align="center">
<img src="xnerf_architecture_4k.png" alt="X-NERF++ overall architecture diagram" width="100%">
</div>

The framework is organized into six stages, top to bottom:

| Stage | Component | Role |
|---|---|---|
| 1 | **Input modalities** | Six parallel input streams: binary images, API call sequences, CFG graphs, memory traces, network events, ISR |
| 2 | **Modality-specific encoders** | CNN, transformer, GNN, temporal MLP, temporal transformer, and embedding network per modality |
| 3 | **Semantic Field Synchronizer (SFS)** | Cross-modal attention and temporal fusion into a shared latent representation `[B × T × 2048]` |
| 4 | **Cross-architecture aligner** | Gradient reversal layer + architecture discriminator for architecture-invariant features (x86, x64, ARM, ARM64, MIPS, RISC-V) |
| 5 | **Malware Neural Execution Field (MNEF)** | Continuous field `F(x, t, s, m, a)` over execution position, temporal state, semantic embedding, memory context, and architecture embedding |
| 6 | **Task heads** | Detection, family attribution (221 families), cross-architecture recognition, zero-shot recognition, retrieval |

A preprocessing pipeline (dataset loading through ISR generation) feeds Stage 1, and a five-term training objective — classification, family, adversarial, field-smoothness, and prototype-contrastive losses — combines into the total loss used during optimization.

## Features

- **Six-modality fusion** — combines static, dynamic, structural, and semantic views of a binary in one model
- **Architecture invariance** — adversarially trained to generalize across x86, x64, ARM, ARM64, MIPS, and RISC-V
- **Continuous execution field** — MNEF models malware behavior as a continuous function rather than a fixed-length vector
- **Five task heads from one backbone** — detection, attribution, cross-architecture matching, zero-shot recognition, retrieval
- **Deployment-ready** — export path to ONNX/TorchScript, a FastAPI inference service, Docker packaging, and a CLI analyzer

## Installation

```bash
git clone https://github.com/Mayukh27/xnerf.git
cd xnerf

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick start

**Training**

```bash
python train.py \
  --config configs/xnerf_pp.yaml \
  --data-dir /path/to/dataset \
  --output-dir checkpoints/
```

**Inference (CLI)**

```bash
python cli_analyzer.py --input sample.bin --checkpoint checkpoints/xnerf_pp.pt
```

**Serve as an API**

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

**Docker**

```bash
docker build -t xnerf-plus-plus .
docker run -p 8000:8000 xnerf-plus-plus
```

## Repository structure

```
x-nerf-plus-plus/
├── configs/                 # training and model configs
├── data/                    # preprocessing pipeline (parsers, extractors, ISR generator)
├── models/
│   ├── encoders/             # per-modality encoders
│   ├── sfs.py                 # Semantic Field Synchronizer
│   ├── aligner.py             # Cross-Architecture Aligner (GRL + discriminator)
│   ├── mnef.py                 # Malware Neural Execution Field
│   └── heads/                 # task heads
├── service/                 # FastAPI inference service
├── cli_analyzer.py          # command-line analyzer
├── train.py
└── xnerf_architecture.svg   # architecture diagram (vector)
```

## Citation

If you use X-NERF++ in your research, please cite:

```bibtex
@article{xnerfpp2026,
  title   = {X-NERF++: Cross-Architecture Neural Execution Rendering Framework},
  author  = {<authors>},
  journal = {<venue>},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
