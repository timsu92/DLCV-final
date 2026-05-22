# patch: knshnb gradient accumulation

Recorded at: 2026-05-21T17:43:24

Reason: allow smaller 16 GB micro-batches while preserving approximate effective batch size.

Behavior impact: changes optimizer step frequency when `accumulate_grad_batches` is greater than 1. Model architecture is unchanged.

Changed files:
- `3rd-party/kaggle-happywhale-1st-place/src/train.py (foreach=False)`
