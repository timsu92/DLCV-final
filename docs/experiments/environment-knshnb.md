# environment: knshnb reproduction

Purpose: dedicated environment for `3rd-party/kaggle-happywhale-1st-place`.

This environment must not replace the main project environment.

Preferred Python: 3.10 or 3.11.

Pinned repo requirements:

- `3rd-party/kaggle-happywhale-1st-place/requirements.txt`

Installation command to try first:

```bash
uv venv .venv-knshnb --python 3.10
. .venv-knshnb/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r 3rd-party/kaggle-happywhale-1st-place/requirements.txt
python -m pip freeze > results/knshnb-freeze.txt
```

If Python 3.10 is not available locally, use Python 3.11 and record the change in this file.

If old PyTorch Lightning fails with installed torch, patch compatibility only after recording the failure and command output.

Observed on 2026-05-17:

- `uv venv .venv-knshnb --python 3.10` succeeded after downloading CPython 3.10.20.
- `python -m pip freeze` initially failed because the environment did not include `pip`; `uv pip install --python .venv-knshnb/bin/python pip` was required before freezing.
- Import verification initially failed with `TypeError: Descriptors cannot be created directly` while `timm` imported `wandb`; `uv pip install --python .venv-knshnb/bin/python protobuf==3.20.3` fixed the compatibility issue.
