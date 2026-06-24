# TRAINING_READINESS_REPORT.md
## Training Readiness Report

## Summary

The repository is closer to training-ready after this pass, but a full training readiness sign-off requires a working Python environment with `torch`, `torch_geometric`, and `pytest`.

## Fixed Before Training

- ISR branch connected to fusion and downstream heads.
- CFG graph branch key fixed.
- Invalid family placeholders masked from family CE.
- README/project context updated to avoid unsupported claims.

## Checks Attempted

| Check | Result |
|---|---|
| Python AST syntax parse for edited files | Passed |
| Focused pytest suite | Blocked: `pytest` unavailable |
| Synthetic forward/backward | Blocked: `torch` unavailable |
| One mini-batch smoke test | Blocked: runtime/dependency issue |
| One epoch smoke test | Not run |
| Checkpoint save | Not run |
| Validation loop | Not run |

## Required Command Once Environment Is Repaired

```powershell
python -m pytest tests
python -m xnerf.pipeline.local_run train --config config.yaml
```

For the balanced manifest:

```powershell
python -m xnerf.pipeline.local_run pipeline --config config_balanced_90k.yaml
```

## Readiness Score

Current engineering readiness: 70/100.

Publication readiness: 25/100 until real training, metrics, ablations, and claim validation artifacts exist.

