# decision: knshnb-debug

Recorded at: 2026-05-17T14:20:39

Debug run failed. Prep run folder: results/2026-05-17-142002-knshnb-debug. Wrapper run folder: results/2026-05-17-142025-knshnb-debug. Exact failure: ModuleNotFoundError: No module named 'optuna'. Wrapper raised subprocess.CalledProcessError after the train command exited with status 1. Debug config was inspected first and is a small smoke-test config.

---

Recorded at: 2026-05-17T14:30:13

Rerun after wrapper repair commit c1577cb78f4922ac0cd5d42d824a7dae739d48e3. Wrapper invoked child as /project/.venv-knshnb/bin/python, and child imports resolved from /project/.venv-knshnb/lib/python3.10/site-packages, so the env interpreter was preserved end-to-end. Related rerun folder: results/2026-05-17-142953-knshnb-debug. Exact failure: FileNotFoundError: [Errno 2] No such file or directory: 'input/train2.csv'. Wrapper then raised subprocess.CalledProcessError after the train command exited with status 1.

---

Recorded at: 2026-05-17T17:05:32

Rerun after three fixes: (1) symlinked six bbox CSVs from charmq sibling's happywhale_data/ into knshnb input/ (registered in source_manifest as phalanx_whale2_*, jpbremer_fullbody_*, jpbremer_*_backfin_via_charmq); (2) patched src/train.py:configure_optimizers to return the scheduler as `[{"scheduler": scheduler, "interval": "epoch"}]` so PL 1.5.10's dict branch is taken instead of the broken `_LRScheduler` isinstance check (the alias was removed in torch>=2.0; see docs/experiments/patch-knshnb-lr-scheduler-dict.md); (3) downgraded numpy to 1.23.5 in .venv-knshnb because albumentations==1.1.0 uses the removed `np.int` alias inside RandomGridShuffle (env note updated).

Related run folder: results/2026-05-17-170232-knshnb-debug. Exit code 0. Training ran 5 epochs of resnet18d at 128x128 on n_data=1000 with all four bbox variants loaded (detic low_conf 0/51033, fullbody 0/51033, fullbody_charm 10/51033, backfin 1587/51033). Train loss dropped 13.77 → 8.79 over epochs 0–3, val MAP rose 0.0 → 0.02, train accuracy rose 0.001 → 0.41. Inference produced four prediction artifacts (train+test × fullbody+fullbody_charm) under predictions/debug/0/. The trailing NCCL ALLGATHER warning during DDP teardown is benign process-exit noise; exit status was 0.

Status: smoke test green. Cleared to proceed to Task 10 (B6 and B7 training).
