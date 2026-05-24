# Kaggle notebook cells for X-NERF++.
#
# Cell 1: install dependencies
# !pip install -q torch-geometric transformers fastapi uvicorn capstone networkx ray umap-learn reportlab
#
# Cell 2: clone or upload project
# %cd /kaggle/working
# !git clone <your-repo-url> xnerf-project || true
# %cd /kaggle/working/xnerf-project
#
# Cell 3: dataset archive locations
# # Upload this local folder as a Kaggle Dataset named: xnerf-malware-archives
# # local: data/archives/<dataset_name>/*.zip|*.tar|*.tar.gz
# # kaggle: /kaggle/input/xnerf-malware-archives/archives/<dataset_name>/*.zip|*.tar|*.tar.gz
# !find /kaggle/input/xnerf-malware-archives/archives -maxdepth 4 -type f | head -50
#
# Cell 4: extract archives into Kaggle working disk
# !python -m xnerf.datasets.extract_archives \
#   --archive-root /kaggle/input/xnerf-malware-archives/archives \
#   --data-root /kaggle/working/data
#
# Cell 5: build manifest and ISR cache
# !python -m xnerf.datasets.build_dataset \
#   --root /kaggle/working/data \
#   --out /kaggle/working/data/processed/manifest.jsonl
#
# Cell 6: GPU check
# import torch
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
#
# Cell 7: train on Kaggle T4
# !python -m xnerf.training.train --config xnerf/configs/kaggle.yaml
#
# Cell 8: export compact local inference checkpoint
# !python -m xnerf.deployment.export_checkpoint \
#   --input /kaggle/working/checkpoints/best.pt \
#   --config xnerf/configs/kaggle.yaml \
#   --output /kaggle/working/export/xnerf_local_inference.pt
#
# Cell 9: download artifact from Kaggle output panel
# !ls -lh /kaggle/working/export/xnerf_local_inference.pt
