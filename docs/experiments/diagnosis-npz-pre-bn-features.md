# diagnosis: NPZ embed_features stored pre-BN backbone features (not neck output)

Recorded at: 2026-05-21 (initial: 2026-05-21; resolved: 2026-05-21)

## Status: RESOLVED — score 0.83464 after fix

## Symptoms

- B6 training MAP@5 ≈ 0.95, B7 ≈ 0.81, but Kaggle submission score was 0.07149
- `embed_features1` in the NPZ: all-positive values (mean=4.34, std=2.64), pairwise cosine similarity 0.9485
- KNN completely unable to distinguish individuals

## Root Cause

`src/train.py` runs inference through Lightning Trainer immediately after training:

```python
trainer = Trainer(precision=16, sync_batchnorm=True, ...)
trainer.fit(model, ...)
trainer.test(model, ...)   # ← saves NPZ
```

Under `precision=16 + sync_batchnorm=True`, `trainer.test()` does not correctly execute
`neck` (a `BatchNorm1d`). The NPZ received the raw backbone concatenation `h` (post-ReLU,
all-positive) instead of `neck(h)` (BatchNorm output, zero-centered, mixed-sign).

### What `get_feat()` is supposed to return

```python
def get_feat(self, x):
    ms = self.backbone(x)
    h = torch.cat([global_pool(m) for global_pool, m in zip(self.global_pools, ms)], dim=1)
    return self.neck(h)   # ← BatchNorm1d output; zero-centered, mixed-sign
```

Both the KNN features (`embed_features1/2`) and the ArcFace logits (`pred_logit`) in the NPZ
depend on `get_feat()` output, so both were broken.

### Verification numbers

| Comparison | Cosine similarity |
|---|---|
| NPZ feat vs current backbone `h` | **0.84** |
| NPZ feat vs current `neck(h)` | **0.07** |
| `last.ckpt` training top-1 accuracy (loaded correctly) | **100%** |

The checkpoint itself was good; only the NPZ was wrong.

## Failed Workaround (no longer in codebase)

`src/ensemble.py` was patched with mean-centering (subtract per-model training-set feature
mean from both train and test features before KNN). The intra/inter cosine gap improved 18×
(0.028 → 0.508), lifting submission score from 0.07149 → 0.13169. This did not fix the
ArcFace logit side, and the gap to the expected score remained large.
**Mean-centering has been removed from ensemble.py.**

## Fix

Added `src/rerun_inference.py` — a standalone script that loads the checkpoint with
`model.eval()` in float32, bypassing Lightning Trainer entirely:

```python
model = SphereClassifier.load_from_checkpoint(ckpt)
model.eval()   # float32, no AMP, no sync_batchnorm
# manual DataLoader → save NPZ
```

Multi-GPU inference is supported via `torchrun` (shard-based, no DataParallel):

```bash
# single GPU
python -m src.rerun_inference --ckpt last.ckpt --out_dir ... --config ... --batch_size 8

# dual GPU
torchrun --nproc_per_node=2 -m src.rerun_inference --ckpt last.ckpt --out_dir ... --config ... --batch_size 8
```

**Do not use `trainer.test()` in `src/train.py` for inference.** This bug is latent in the
original repo and will recur if `do_inference=True` is passed to `train()`.

## Resolution

After re-running inference with `rerun_inference.py` for both B6 and B7:

- Correct post-BN features: mean ≈ 0.02, std ≈ 0.53–0.61 (zero-centered, mixed-sign)
- B6 + B7 ensemble (`knn_ratio=0.5`, `new_ratio=0.165`): **Kaggle score 0.83464**

Submission: `results/2026-05-21-knshnb-b6-b7-ensemble-corrected/submissions/b6b7-corrected-0.165-0.46197943296283484.csv`
