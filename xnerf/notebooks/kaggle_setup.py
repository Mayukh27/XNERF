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
# # Zip each dataset subfolder under data/archives, then upload the whole
# # data/archives folder as one Kaggle Dataset named: xnerf-malware-archives
# # local: data/archives/<dataset_name>/*.zip|*.tar|*.tar.gz
# # kaggle: /kaggle/input/xnerf-malware-archives/archives/<dataset_name>/*.zip|*.tar|*.tar.gz
# !find /kaggle/input/xnerf-malware-archives/archives -maxdepth 4 -type f | head -50
#
# Cell 4: GPU check
# import torch
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
#
# Cell 5: one real-world run: extract, split, train, test, zero-shot, export
# !python -m xnerf.pipeline.kaggle_run --config xnerf/configs/kaggle.yaml
#
# Cell 6: download artifacts from Kaggle output panel
# !find /kaggle/working/xnerf_output -maxdepth 1 -type f -print
# !cat /kaggle/working/xnerf_output/summary.json




#python -m xnerf.pipeline.kaggle_run build-manifest --config xnerf/configs/kaggle.yaml
#python -m xnerf.pipeline.kaggle_run train           --config xnerf/configs/kaggle.yaml
# -- save outputs, start a new session, load checkpoint as dataset --
#python -m xnerf.pipeline.kaggle_run test            --config xnerf/configs/kaggle.yaml
#python -m xnerf.pipeline.kaggle_run zero-shot       --config xnerf/configs/kaggle.yaml
#python -m xnerf.pipeline.kaggle_run export          --config xnerf/configs/kaggle.yaml


#python -m xnerf.pipeline.local_run build-manifest
#python -m xnerf.pipeline.local_run train
#python -m xnerf.pipeline.local_run test
# point at a specific checkpoint without retraining:
#python -m xnerf.pipeline.local_run test --checkpoint checkpoints/epoch3.pt