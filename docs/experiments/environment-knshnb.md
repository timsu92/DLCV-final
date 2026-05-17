# environment: knshnb reproduction

Purpose: dedicated environment for `3rd-party/kaggle-happywhale-1st-place`.

This environment must not replace the main project environment.

Reproduced against knshnb submodule commit `6e78f87caa7f0242ffe3288d46f8567d40dae3f3`.

Preferred Python: 3.10.

Python 3.11 is not a drop-in fallback for the frozen environment in `results/knshnb-freeze.txt`. That freeze includes `scipy==1.8.1`, which is not reproducible unchanged on Python 3.11. Use 3.11 only for a separate compatibility investigation and record the resulting dependency changes.

Pinned repo requirements:

- `3rd-party/kaggle-happywhale-1st-place/requirements.txt`

Initial attempt from the Task 8 plan:

```bash
uv venv .venv-knshnb --python 3.10
. .venv-knshnb/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r 3rd-party/kaggle-happywhale-1st-place/requirements.txt
python -m pip freeze > results/knshnb-freeze.txt
```

Working reproducible sequence used on 2026-05-17:

```bash
uv venv .venv-knshnb --python 3.10
uv pip install --python .venv-knshnb/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-knshnb/bin/python -r 3rd-party/kaggle-happywhale-1st-place/requirements.txt
uv pip install --python .venv-knshnb/bin/python pip
uv pip install --python .venv-knshnb/bin/python protobuf==3.20.3
.venv-knshnb/bin/python -m pip freeze > results/knshnb-freeze.txt
```

Verification command:

```bash
.venv-knshnb/bin/python -c "import torch, timm, pytorch_lightning, albumentations, pandas, sklearn; print(torch.__version__); print(timm.__version__)"
```

If old PyTorch Lightning fails with installed torch, record the failure and command output before applying any further compatibility change.

Observed on 2026-05-17:

- `uv venv .venv-knshnb --python 3.10` succeeded after downloading CPython 3.10.20.
- The initial attempt was not sufficient as written: `python -m pip freeze` failed because the environment did not include `pip`; `uv pip install --python .venv-knshnb/bin/python pip` was required before freezing.
- Import verification initially failed with `TypeError: Descriptors cannot be created directly` while `timm` imported `wandb`; `uv pip install --python .venv-knshnb/bin/python protobuf==3.20.3` fixed the compatibility issue.

Observed on 2026-05-17 (second pin, during smoke-test data loader):

- Training data loader crashed with `AttributeError: module 'numpy' has no attribute 'int'` inside `albumentations==1.1.0` (`RandomGridShuffle.get_params_dependent_on_targets`). `np.int` was removed in numpy 1.24. `requirements.txt` pins `albumentations==1.1.0` but does not pin numpy, so it had resolved to numpy 1.24.4. Fixed with `.venv-knshnb/bin/pip install 'numpy<1.24'` which installed numpy 1.23.5. Refresh `results/knshnb-freeze.txt` after confirming the smoke test passes.
