# CROSS_ARCH_AUDIT.md
## Cross-Architecture Audit

## Search Result

The repository contains a `CrossArchitectureAligner` with a gradient-reversal architecture discriminator. The live training path uses `arch_logits` through `classification_losses`.

## Live Path

```text
semantic.mean(dim=1)
-> CrossArchitectureAligner
-> arch_logits
-> F.cross_entropy(arch_logits, arch_id) * 0.1
```

## Status

- Architecture adversarial loss: implemented and called.
- Paired same-malware cross-architecture loss: not live.
- Architecture prototype/class-conditional fallback: not implemented in this pass.

## Why Not Fully Fixed

The current manifest/dataloader does not produce paired same-malware samples across architectures or class-conditional per-architecture batches. Adding a prototype fallback without a verified multi-architecture distribution would risk creating a decorative loss term rather than a defensible research mechanism.

## Required Next Step

Add an explicit batch sampler or memory bank that groups embeddings by family/class and architecture, then document whether the loss is pairwise, prototype-based, or class-conditional. Do not claim paired cross-architecture alignment until this exists and receives gradients in a smoke test.

