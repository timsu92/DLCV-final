# patch: knshnb LR scheduler dict wrapping

Recorded at: 2026-05-17

## Reason

`pytorch_lightning==1.5.10` (pinned by knshnb requirements) validates the lr scheduler returned from `configure_optimizers` with `isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)`. In `torch>=2.0` (we have 2.11.0+cu128) the base class was renamed to `LRScheduler` and `_LRScheduler` no longer appears in the LambdaLR MRO, so the isinstance check fails with:

```
ValueError: The provided lr scheduler "<torch.optim.lr_scheduler.LambdaLR object at 0x...>" is invalid
```

## Fix

`src/train.py:configure_optimizers` previously returned:

```python
return [optimizer], [scheduler]
```

Now returns:

```python
return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
```

PL 1.5's `_configure_schedulers` checks `isinstance(scheduler, dict)` *before* the broken isinstance check, so the dict branch is taken and the validation is satisfied with no behavior change.

## Behavior impact

None. `interval: "epoch"` matches PL's default and matches the original code path (the `WarmupCosineLambda` is parameterized in epochs via `max_epochs` and `warmup_steps_ratio`).

## Changed files

- `3rd-party/kaggle-happywhale-1st-place/src/train.py` (configure_optimizers return)
