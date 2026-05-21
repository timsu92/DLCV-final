# Kaggle Happy Whale and Dolphin — 1st-Place Reproduction

Reproduction of the [Preferred Dolphin 1st-place solution](https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/preferred-dolphin-1st-place-solution) for the Kaggle [Happy Whale and Dolphin](https://www.kaggle.com/competitions/happy-whale-and-dolphin) competition.

This repository wraps the knshnb pipeline (`3rd-party/kaggle-happywhale-1st-place`) with
scripts for reproducible run management, data preparation, and ensemble generation.

<!-- **Reproduced score: 0.83464** (B6 + B7 ensemble, knn_ratio=0.5) -->

## Key Bug Fix

The original `src/train.py` inference path (`trainer.test()`) silently stores
pre-BatchNorm backbone features instead of the correct post-BN neck output due to a
Lightning Trainer `precision=16 + sync_batchnorm=True` interaction.
This caused submission scores of 0.07–0.13 despite 0.95/0.81 training MAP.

**Always use `src/rerun_inference.py` for inference. Do not use `trainer.test()`.**

See [`docs/experiments/diagnosis-npz-pre-bn-features.md`](docs/experiments/diagnosis-npz-pre-bn-features.md)
for the full diagnosis, verification numbers, and fix rationale.

## Repository Layout

```
3rd-party/kaggle-happywhale-1st-place/   knshnb pipeline (submodule)
3rd-party/kaggle-happywhale-1st-place-solution-charmq/  charmq pipeline (submodule)
data/                                    competition data (downloaded separately)
docs/experiments/                        experiment records and diagnoses
results/                                 training runs (gitignored)
scripts/                                 project-level helper scripts
  prepare_data.py    symlink competition data into repo input/ dirs
  run_manager.py     create timestamped result directories
  run_knshnb.py      invoke knshnb training
  run_ensemble.py    invoke knshnb ensemble
  record_experiment.py  append notes to docs/experiments/
```

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules <this-repo>
cd <this-repo>
```

### 2. Create the knshnb Python environment

```bash
uv venv .venv-knshnb --python 3.10
uv pip install -r 3rd-party/kaggle-happywhale-1st-place/requirements.txt \
    --python .venv-knshnb
```

### 3. Download competition data

Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and a configured `~/.kaggle/kaggle.json`.

```bash
mkdir -p data
kaggle competitions download -c happy-whale-and-dolphin -p data/
cd data && unzip happy-whale-and-dolphin.zip && cd ..
```

### 4. Link competition data into the knshnb pipeline

```bash
python scripts/prepare_data.py \
    --repo knshnb \
    --data-dir data \
    --target-dir 3rd-party/kaggle-happywhale-1st-place/input
```

This creates symlinks for `train.csv`, `sample_submission.csv`, `train_images/`, and
`test_images/`. The `fullbody_charm` CSVs and label arrays are already present in the knshnb
submodule itself.

### 5. Link auxiliary bbox CSVs from the charmq submodule

The fullbody, backfin, and Detic bbox files must be symlinked manually from the charmq
submodule. Run from the **project root**:

```bash
KNSHNB=3rd-party/kaggle-happywhale-1st-place
CHARMQ=3rd-party/kaggle-happywhale-1st-place-solution-charmq

ln -s "$(pwd)/$CHARMQ/happywhale_data/fullbody_train.csv" "$KNSHNB/input/fullbody_train.csv"
ln -s "$(pwd)/$CHARMQ/happywhale_data/fullbody_test.csv"  "$KNSHNB/input/fullbody_test.csv"
ln -s "$(pwd)/$CHARMQ/happywhale_data/train_backfin.csv"  "$KNSHNB/input/train_backfin.csv"
ln -s "$(pwd)/$CHARMQ/happywhale_data/test_backfin.csv"   "$KNSHNB/input/test_backfin.csv"
ln -s "$(pwd)/$CHARMQ/happywhale_data/train2.csv"         "$KNSHNB/input/train2.csv"
ln -s "$(pwd)/$CHARMQ/happywhale_data/test2.csv"          "$KNSHNB/input/test2.csv"
```

> **Why from charmq?** The original Kaggle phalanx Detic-bbox dataset
> (`kaggle.com/datasets/phalanx/whale2-cropped-dataset`) returns HTTP 404 as of 2026-05-17.
> The jpbremer fullbody/backfin datasets are also mirrored here.
> See [`data/source_manifest.yaml`](data/source_manifest.yaml) for full provenance.

## Training

Create a run directory, then train. Commands are run from the **project root**.

```bash
# Create run directory (prints the path)
RUN_DIR=$(python scripts/run_manager.py \
    --run-name knshnb-b6 \
    --notes "EfficientNet-B6 B6 reproduction")
# e.g. results/2026-05-21-123456-knshnb-b6

# Train (checkpoint saved to <RUN_DIR>/predictions/b6-batch3-accum2/-1/last.ckpt)
python scripts/run_knshnb.py \
    --config-path config/efficientnet_b6.yaml \
    --exp-name b6-batch3-accum2 \
    --out-base-dir "$(pwd)/$RUN_DIR/predictions" \
    --save-checkpoint
```

## Inference

**Use `rerun_inference.py`, not `trainer.test()`** (see Key Bug Fix above).

Run from inside the knshnb repo directory:

```bash
cd 3rd-party/kaggle-happywhale-1st-place

# Single GPU
.venv-knshnb/bin/python -m src.rerun_inference \
    --ckpt /path/to/results/.../predictions/b6-batch3-accum2/-1/last.ckpt \
    --out_dir /path/to/results/.../predictions/b6-batch3-accum2/-1 \
    --config config/efficientnet_b6.yaml \
    --batch_size 8

# Dual GPU (each GPU gets batch_size=8)
.venv-knshnb/bin/torchrun --nproc_per_node=2 -m src.rerun_inference \
    --ckpt /path/to/results/.../predictions/b6-batch3-accum2/-1/last.ckpt \
    --out_dir /path/to/results/.../predictions/b6-batch3-accum2/-1 \
    --config config/efficientnet_b6.yaml \
    --batch_size 8

cd ../..
```

Inference generates `train_fullbody_results.npz`, `test_fullbody_results.npz`,
`train_fullbody_charm_results.npz`, and `test_fullbody_charm_results.npz` inside `--out_dir`.

Healthy post-BN features look like: `mean ≈ 0.02, std ≈ 0.53–0.61` (zero-centered, mixed-sign).
If you see `mean ≈ 4, std ≈ 2.6` with all-positive values, the wrong inference path was used.

## Ensemble

```bash
python scripts/run_ensemble.py \
    --model-dir /path/to/b6-results/predictions/b6-batch3-accum2/-1 \
    --model-dir /path/to/b7-results/predictions/b7-batch2-accum2/-1 \
    --out-prefix b6b7
```

The submission CSV and pseudo-label CSV are written to
`3rd-party/kaggle-happywhale-1st-place/submission/`. Move them to `results/` for archiving.

## Reproduction Results

| Run | Score |
|---|---|
| B6 + B7 ensemble (broken NPZ, pre-BN features) | 0.07149 |
| B6 + B7 ensemble + mean-centering workaround | 0.13169 |
| B6 + B7 ensemble (fixed NPZ via rerun_inference.py) | **0.83464** |
