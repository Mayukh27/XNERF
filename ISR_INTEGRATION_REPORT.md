# ISR_INTEGRATION_REPORT.md
## ISR Integration Report

## Status

ISR is now connected to model predictions.

## Forward Path

```text
data/cache/isr/<sha>.pt
-> MalwareManifestDataset.__getitem__ returns isr [1024,4]
-> collate_dicts stacks to batch["isr"] [B,1024,4]
-> XNERFPlusPlus.forward
-> ISREncoder
-> embeddings["isr"] [B,512]
-> SemanticFieldSynchronizer modality "isr"
-> semantic [B,field_time,2048]
-> pooled/aligned [B,2048]
-> malware_logits, family_logits, zero_shot_embedding
```

## Tensor Dimensions

- Input ISR: `[B,T,4]`
- ISR columns: semantic id, architecture id, address-delta bucket, instruction size
- ISR embedding: `[B,512]`
- SFS output: `[B,field_time,2048]`
- Malware logits: `[B,2]` by default
- Family logits: `[B,num_families]`

## Evidence

- `xnerf/encoders/isr.py:10`: `ISREncoder`
- `xnerf/model.py:43`: model owns `self.isr`
- `xnerf/model.py:68`: `embeddings["isr"] = self.isr(batch["isr"])`
- `xnerf/synchronization/sfs.py:28`: `isr` is an accepted SFS modality

## Gradients

Expected gradient path:

```text
malware/family loss -> aligned -> semantic -> SFS attention/projection -> ISR encoder
```

Runtime gradient verification could not be executed in this environment because the available Python runtime lacks `torch`.

