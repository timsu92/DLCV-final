# Happywhale First-Place Reproduction Design

Date: 2026-05-17

## Goal

Build a score-first reproduction workflow for the Kaggle Happywhale - Whale and Dolphin Identification competition. The immediate target is to reproduce a first-place-level submission using the public Preferred Dolphin solution code and artifacts, then use any remaining time to improve or ensemble additional models.

This project also needs enough provenance for a later report with introduction, related works, method / approach, experimental results, conclusion, and references. Source tracking is therefore part of the design, but Kaggle score remains the primary objective.

## Current Environment

The local official competition data is already available under `data/`.

- Train images: 51,033
- Test images: 27,956
- Train identities: 15,587
- Singleton identities: 9,258
- GPUs: 2 x NVIDIA GeForce RTX 5080, about 16 GB VRAM each
- System RAM: about 125 GiB
- Free project disk space at exploration time: about 264 GB
- PyTorch visible CUDA devices: 2

The machine is strong enough for one or more 16 GB-friendly single-model training runs, embedding extraction, and prediction-level ensemble. It is not expected to reproduce the original team's large multi-model, multi-GPU training setup unchanged.

## External Sources

Use public sources directly when useful. Preserve source information before relying on any artifact.

Primary sources:

- Kaggle competition: `https://www.kaggle.com/competitions/happy-whale-and-dolphin`
- Kaggle first-place writeup: `https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/preferred-dolphin-1st-place-solution`
- Local summary: `docs/happywhale-1st-place-solution.zh-TW.md`
- knshnb repo: `3rd-party/kaggle-happywhale-1st-place`
  - Commit observed: `69690142177ecb69e8a4d1720837ec02c25071d6`
  - URL: `https://github.com/knshnb/kaggle-happywhale-1st-place`
- charmq repo: `3rd-party/kaggle-happywhale-1st-place-solution-charmq`
  - Commit observed: `c4ca5e2fe63cf5952fc8afaf0e85d7613a29d00b`
  - URL: `https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq`
- Preferred Networks technical article:
  - `https://tech.preferred.jp/ja/blog/kaggle-happywhale-1st-10th-solution/`

Artifact sources named by the public solution:

- Jan Bre fullbody annotations dataset
- Jan Bre backfin YOLOv5 notebook outputs
- phalanx whale2 cropped dataset / Detic boxes
- Awsaf YOLOv5 cropped dataset notebook outputs
- Preferred Dolphin pseudo labels, especially `round2.csv` if available

## Directory Layout

Use the user-approved project layout:

- `data/source_manifest.yaml`
  - Records every external source, artifact, model weight, pseudo label, bbox CSV, repo URL, commit, retrieval date, checksum, and intended use.
- `data/external/`
  - Stores downloaded external bbox files, pseudo labels, pretrained weights, and other Kaggle artifacts.
- `results/<timestamp>-<run_name>/`
  - Stores each run's logs, configs, intermediate prediction artifacts, metrics, and submissions.
- `docs/experiments/`
  - Stores report-friendly experiment notes, patch notes, OOM notes, score logs, and decision records.

For each run, create a run folder such as:

```text
results/2026-05-17-143000-knshnb-b6/
  run_manifest.yaml
  logs/
  configs/
  predictions/
  submissions/
  metrics/
```

Submissions are stored inside the corresponding run folder rather than in a separate top-level submissions directory.

## Source Manifest

`data/source_manifest.yaml` is the canonical provenance file. It must be updated during the workflow, not only at the end. Whenever a new external source, document, dataset, bbox file, pseudo label, model weight, generated derivative, or copied artifact is first used, add or update its manifest entry before relying on it in an experiment.

This applies to data and documentation sources. If a source is useful for the later report, it belongs in the manifest even if it is not consumed by training code.

Each entry should include:

- Source name
- Source type: competition data, git submodule, Kaggle dataset, notebook output, pseudo label, pretrained weight, generated artifact
- URL or local origin
- Git commit, dataset version, or notebook URL when available
- Retrieval date
- Local path
- SHA256 checksum for downloaded or generated files when practical
- License or Kaggle usage note when available
- Purpose in the pipeline

The official Kaggle competition data should be recorded as already downloaded, not re-downloaded unless a missing file is detected. The manifest should be copied or referenced by every timestamped run so each result can be traced to the source state that produced it.

## Main Approach

Use the public first-place repositories as executable reference pipelines. Do not rewrite and merge both repositories into a new training framework at the start.

Recommended execution order:

1. Use `knshnb/kaggle-happywhale-1st-place` as the primary control pipeline.
2. Make local data paths match its expected `input/` layout through symlinks or minimal copies.
3. Run knshnb EfficientNet B6 and B7 first.
4. Use knshnb `src/ensemble.py` to combine model outputs with KNN + logit postprocessing.
5. Add charmq outputs only after the knshnb B6/B7 path is stable.

This matches the public solution structure: each teammate kept an independent pipeline, and ensemble happened at the prediction artifact level. The charmq README and knshnb README describe compatible prediction formats, so the integration point should be model directories / prediction files, not merged model code.

## Model Strategy

Primary models:

- `tf_efficientnet_b6_ns`, 1024 x 1024, knshnb config
- `tf_efficientnet_b7_ns`, 1024 x 1024, knshnb config

The knshnb README states that a single B7 could rank around third on the final leaderboard post submission, and B6 + B7 ensemble could reach first-place-level performance. Therefore B6 + B7 is the first target.

Secondary models, only after B6 + B7 works:

- charmq EfficientNet B7 compatible output
- EfficientNetV2-M or V2-L if VRAM/time allow
- Extra B5/B6/B7 seed or image-size variants

Do not prioritize custom attention modules at the beginning. The first-place solution already uses GeM pooling, sub-center ArcFace, dynamic margins, flip TTA, bbox mix, fullbody / fullbody_charm inference, KNN, logits, and pseudo labels. Adding a new attention block would increase risk before the known high-value pieces are reproduced.

ConvNeXt is not a first-priority model because the collected first-place notes list it among methods that did not help.

## Data and Crop Strategy

Use the existing public bbox/crop artifacts rather than training a detector from scratch first.

Training crop mix should follow the first-place design when using knshnb configs:

- fullbody: main source
- fullbody_charm: secondary fullbody detector source
- backfin: important for dorsal fin-only images
- Detic boxes
- no crop regularization

Inference should first use the known high-value crop combination:

- fullbody
- fullbody_charm
- horizontal flip TTA already present in the repo

Backfin, YOLOv5, and Detic variants can be used where the original configs support them, but the first goal is to reproduce B6/B7 with the expected fullbody/fullbody_charm inference artifacts.

## Training Adaptation for 16 GB VRAM

The original configs are close to this machine's limit:

- knshnb B6: 1024 x 1024, batch size 6
- knshnb B7: 1024 x 1024, batch size 4

Use automatic mixed precision. If OOM occurs, reduce micro-batch size and add gradient accumulation to preserve the approximate effective batch size.

Examples:

- B6 original batch 6
  - Try batch 6
  - If OOM, batch 4 with accumulation 2, or batch 3 with accumulation 2
  - Last resort before image downscaling: batch 2 with accumulation 3
- B7 original batch 4
  - Try batch 4
  - If OOM, batch 2 with accumulation 2
  - Last resort before image downscaling: batch 1 with accumulation 4

Gradient accumulation is not perfectly equivalent to a larger batch because BatchNorm and per-step mini-batch composition still differ, but it is better than shrinking batch size without compensation.

If 1024 still cannot run:

1. Reduce `num_workers` and confirm no memory leak.
2. Use a lower micro-batch and accumulation.
3. Try 896 x 896 as a fallback.
4. Switch to B5/B6 or an extra seed rather than spending the whole schedule on one impossible run.

## Patch Policy

Small patches are allowed when needed to run the public solution on this environment. Avoid large rewrites.

Allowed patch categories:

- Add `accumulate_grad_batches` config support to knshnb `Trainer`.
- Patch PyTorch Lightning API compatibility if the old repo does not run on the selected environment.
- Override output directories or add wrapper-level result copying.
- Override batch size, number of workers, GPU count, and AMP options.
- Add minimal adapters that convert prediction artifacts into the format expected by ensemble code.

Patch records go in `docs/experiments/`, not under `results/`.

Each patch note should record:

- Timestamp
- File changed
- Reason
- Minimal diff summary
- Whether it changes model behavior or only runtime compatibility
- Related run folder, if any

Note: if a patch is generated by git, do not use `rtk git diff` to create the patch file. Instead, use `git diff` so that git itself would be able to parse.

Intermediate artifacts do not need to be `.npz`. Use the easiest reliable format for each code path:

- Keep `.npz` when using original repo outputs.
- Use `.pt`, `.pth`, `.pkl`, or other practical formats for local adapters if easier.
- Record artifact schema and reader in `run_manifest.yaml`.

## Environment Strategy

Create an isolated environment for reproduction. Do not force the main project environment to match the old public repositories. The main project can keep its current dependencies while reproduction runs in one or more dedicated environments.

The public repos depend on older packages:

- `timm==0.5.4`
- `albumentations==1.1.0`
- `pytorch-lightning==1.5.10` for knshnb
- `pytorch-lightning==1.6.3` for charmq
- `hydra-core==1.1.2` for charmq

The current global Python is 3.12 and does not include the required ML packages. Prefer Python 3.10 or 3.11 in a dedicated reproduction environment to reduce compatibility risk.

Environment selection:

1. Start with one dedicated knshnb reproduction environment, because knshnb is the primary pipeline.
2. Try to reuse that environment for charmq only if the dependency set is compatible.
3. If PyTorch Lightning, Hydra, or other pinned dependencies conflict, split into separate dedicated environments for knshnb and charmq.
4. Do not downgrade or reshape the main project environment just to satisfy either public repo.

Keep the environment creation command, package versions, and `pip freeze` or equivalent lock output in the run manifest.

## Inference and Ensemble

For each trained model, produce train and test prediction artifacts for the expected bbox variants:

- train fullbody
- test fullbody
- train fullbody_charm
- test fullbody_charm

Then run the ensemble step:

- Compute KNN score from embedding features.
- Use model logits as a second score source.
- Average or weighted-average KNN and logits as the repo expects.
- Aggregate by `individual_id`.
- Insert `new_individual` according to threshold / ratio.
- Output top 5 labels per image.

The submission validator must check:

- Row count equals `data/sample_submission.csv`.
- Columns are `image,predictions`.
- Every prediction has exactly 5 labels.
- `new_individual` rate is recorded.
- No unknown labels appear except `new_individual`.

## Pseudo Labeling

For score-first reproduction, use the public solution's `round2.csv` pseudo labels when available and properly sourced. If the pseudo label file is missing, first run without pseudo labels to get a baseline, then reproduce pseudo-label generation if time allows.

Do not start with pseudo-label generation from scratch. It is time-consuming and less valuable than first making the known B6/B7 pipeline work.

## Two-Week Schedule

Days 1-2: Environment and data alignment

- Create source manifest.
- Align `data/` to each repo's expected input layout.
- Download missing public artifacts if needed.
- Create isolated environment.
- Run a debug-sized train / inference / ensemble path.

Days 3-5: knshnb B6

- Train B6 at 1024 if possible.
- Apply batch / accumulation fallback if needed.
- Produce train/test prediction artifacts.
- Produce the first valid submission.

Days 6-8: knshnb B7

- Train B7 at 1024 if possible.
- Apply batch / accumulation fallback if needed.
- Produce train/test prediction artifacts.
- Keep B6 submission as a fallback if B7 fails.

Day 9: B6 + B7 ensemble

- Run KNN + logit ensemble.
- Save submission, config, score notes, and `new_individual` statistics.
- If Kaggle submission is available, record leaderboard score in `docs/experiments/`.

Days 10-12: Add one more source of model diversity

Priority order:

1. charmq B7 with compatible prediction artifacts
2. knshnb EfficientNetV2-M if feasible
3. knshnb B5/B6 extra seed
4. 896 / 1024 image-size variant

Day 13: Final ensemble and threshold tuning

- Combine all successful model directories.
- Sweep or compare `new_individual` thresholds / ratios if supported.
- Pick final submission by public LB score or best available local evidence.

Day 14: Stabilize and document

- Preserve best submission.
- Update `data/source_manifest.yaml`.
- Summarize final method and experiment outcomes in `docs/experiments/`.
- Record failed runs and OOM fallbacks so the report can explain constraints.

## Validation

Minimum done criteria before treating a run as usable:

- The command and config are recorded.
- The run has a timestamped folder under `results/`.
- The source manifest snapshot or referenced manifest version is recorded.
- The model produces prediction artifacts for the expected train/test crop variants.
- Ensemble completes without missing artifact errors.
- Submission passes the format validator.
- Patch notes exist for any code change.

Minimum done criteria for the overall two-week target:

- At least one valid submission exists.
- B6 or B7 single-model result exists.
- B6 + B7 ensemble exists if both trainings succeeded.
- External sources and artifacts are traceable.
- Report notes contain enough detail to describe method, results, and references.

## Risks and Mitigations

Risk: Old dependencies fail on the current system.

Mitigation: Use Python 3.10 or 3.11 isolated env first. Patch only API compatibility if needed.

Risk: B7 1024 OOM on 16 GB.

Mitigation: Use AMP, reduce micro-batch, add gradient accumulation, then try 896 only if needed.

Risk: Training takes too long.

Mitigation: Keep B6 as a valid fallback. Do not block on charmq before B6/B7 works.

Risk: Public artifacts are missing or cannot be downloaded.

Mitigation: Record the missing source, run a no-pseudo-label baseline, then decide whether to regenerate.

Risk: Two repos diverge in artifact schema.

Mitigation: Use knshnb ensemble as the integration target. Add a small adapter only after inspecting actual charmq outputs.

## Approved Direction

The approved direction is:

- Score-first reproduction.
- Use public code and artifacts.
- Keep source provenance for the later report.
- Use knshnb as the primary pipeline.
- Add charmq as an optional ensemble source after knshnb B6/B7 works.
- Allow small runtime patches, especially gradient accumulation.
- Store patch notes in `docs/experiments/`.
- Store all run outputs and submissions under timestamped `results/` folders.
