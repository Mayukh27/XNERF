# LOSS_GRAPH.md
## Loss Graph

Live training loss in `xnerf/training/losses.py`:

```text
final_loss =
  malware_ce
  + 0.1 * family_ce(ignore_index=-1)
  + 0.1 * arch_adv
  + 0.01 * field_smooth
```

## Terms

| Term | Source | Target | Weight | Status |
|---|---|---|---:|---|
| `malware_ce` | `malware_logits` | `label` | 1.0 | Live |
| `family_ce` | `family_logits` | `family_label`, ignoring `-1` | 0.1 | Live, masked |
| `arch_adv` | `arch_logits` | `arch_id` | 0.1 | Live |
| `field_smooth` | `field[:,1:] - field[:,:-1]` | none | 0.01 | Live regularizer |
| `behavior_ce` | `behavior_logits` | missing | n/a | Not live |
| `stage_ce` | `stage_logits` | missing | n/a | Not live |
| `cross_arch_loss` | paired embeddings | missing pairs | n/a | Not live |
| `contrastive_loss` | modality embeddings | missing call | n/a | Not live |

## Gradient Notes

ISR and CFG now feed `semantic`, so malware/family losses can backpropagate through those branches when the corresponding tensors are present. Runtime gradient verification is still pending because `torch` is unavailable in the usable local Python runtime.

