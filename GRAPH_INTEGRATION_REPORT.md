# GRAPH_INTEGRATION_REPORT.md
## Graph Integration Report

## Status

Graph fusion key mismatch is fixed.

## Forward Path

```text
.edgelist
-> MalwareManifestDataset._load_graph
-> graph_x [N,4], graph_edge_index [2,E]
-> collate_dicts adds graph_batch [N]
-> CFGEncoder
-> embeddings["cfg"] [B,512]
-> SemanticFieldSynchronizer modality "cfg"
-> semantic [B,field_time,2048]
-> malware_logits and family_logits
```

## Evidence

- `xnerf/datasets/loaders.py`: `.edgelist` loading creates graph tensors.
- `xnerf/utils/base.py`: `collate_dicts` batches graph tensors.
- `xnerf/model.py:63`: graph encoder output is stored as `embeddings["cfg"]`.
- `xnerf/synchronization/sfs.py:28`: `cfg` is an accepted SFS modality.

## Remaining Caveat

The branch only activates for samples with non-empty `.edgelist` data. Feature-only and API-only samples still train without graph signal, as expected.

