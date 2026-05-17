# decision: knshnb-debug

Recorded at: 2026-05-17T14:20:39

Debug run failed. Prep run folder: results/2026-05-17-142002-knshnb-debug. Wrapper run folder: results/2026-05-17-142025-knshnb-debug. Exact failure: ModuleNotFoundError: No module named 'optuna'. Wrapper raised subprocess.CalledProcessError after the train command exited with status 1. Debug config was inspected first and is a small smoke-test config.

---

Recorded at: 2026-05-17T14:30:13

Rerun after wrapper repair commit c1577cb78f4922ac0cd5d42d824a7dae739d48e3. Wrapper invoked child as /project/.venv-knshnb/bin/python, and child imports resolved from /project/.venv-knshnb/lib/python3.10/site-packages, so the env interpreter was preserved end-to-end. Related rerun folder: results/2026-05-17-142953-knshnb-debug. Exact failure: FileNotFoundError: [Errno 2] No such file or directory: 'input/train2.csv'. Wrapper then raised subprocess.CalledProcessError after the train command exited with status 1.
