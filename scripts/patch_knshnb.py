"""Record all modifications needed to apply to knshnb's repo.
However, since I'm using my fork repo, this file isn't much needed anymore."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

TRAIN_BLOCK_START = "    trainer = Trainer(\n"
TRAIN_BLOCK_END = "    )\n"
TRAIN_NEEDLE = "        logger=loggers,\n"
TRAIN_INSERT = '        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),\n'
YAML_LINE = "accumulate_grad_batches: 1\n"

# foreach=False patch: disables PyTorch 2.x multi-tensor Adam which requires all
# optimizer state tensors on the same device — breaks DDP checkpoint resume with PL 1.5.x
FOREACH_PATCHES = [
    (
        "optimizer = torch.optim.Adam(params)\n",
        "optimizer = torch.optim.Adam(params, foreach=False)\n",
    ),
    (
        "optimizer = torch.optim.AdamW(params)\n",
        "optimizer = torch.optim.AdamW(params, foreach=False)\n",
    ),
    (
        "optimizer = torch.optim.RAdam(params)\n",
        "optimizer = torch.optim.RAdam(params, foreach=False)\n",
    ),
]


def has_top_level_yaml_key(text: str, key: str) -> bool:
    pattern = re.compile(rf"^{re.escape(key)}\s*:")
    return any(pattern.match(line) for line in text.splitlines())


def patch_default_yaml(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if has_top_level_yaml_key(text, "accumulate_grad_batches"):
        return False
    path.write_text(text.rstrip() + "\n" + YAML_LINE, encoding="utf-8")
    return True


def trainer_block_range(text: str, path: Path) -> tuple[int, int]:
    start = text.find(TRAIN_BLOCK_START)
    if start == -1:
        raise ValueError(f"Could not find trainer = Trainer(...) block in {path}")
    end = text.find(TRAIN_BLOCK_END, start + len(TRAIN_BLOCK_START))
    if end == -1:
        raise ValueError(f"Could not find end of trainer = Trainer(...) block in {path}")
    return start, end + len(TRAIN_BLOCK_END)


def patch_train_py(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    start, end = trainer_block_range(text, path)
    block = text[start:end]
    if TRAIN_INSERT in block:
        return False
    if TRAIN_NEEDLE not in block:
        raise ValueError(f"Could not find logger=loggers line inside trainer = Trainer(...) block in {path}")
    patched_block = block.replace(TRAIN_NEEDLE, TRAIN_NEEDLE + TRAIN_INSERT, 1)
    path.write_text(text[:start] + patched_block + text[end:], encoding="utf-8")
    return True


def write_patch_note(path: Path, changed_files: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# patch: knshnb gradient accumulation",
            "",
            f"Recorded at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
            "",
            "Reason: allow smaller 16 GB micro-batches while preserving approximate effective batch size.",
            "",
            "Behavior impact: changes optimizer step frequency when `accumulate_grad_batches` is greater than 1. Model architecture is unchanged.",
            "",
            "Changed files:",
            *[f"- `{file}`" for file in changed_files],
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def patch_optimizer_foreach(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "foreach=False" in text:
        return False
    patched = text
    for old, new in FOREACH_PATCHES:
        patched = patched.replace(old, new)
    if patched == text:
        raise ValueError(f"Could not find optimizer lines to patch in {path}")
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="3rd-party/kaggle-happywhale-1st-place")
    parser.add_argument("--note", default="docs/experiments/patch-knshnb-gradient-accumulation.md")
    args = parser.parse_args()

    repo = Path(args.repo)
    changed: list[str] = []
    if patch_default_yaml(repo / "config" / "default.yaml"):
        changed.append(str(repo / "config" / "default.yaml"))
    if patch_train_py(repo / "src" / "train.py"):
        changed.append(str(repo / "src" / "train.py"))
    if patch_optimizer_foreach(repo / "src" / "train.py"):
        changed.append(str(repo / "src" / "train.py") + " (foreach=False)")
    if changed:
        write_patch_note(Path(args.note), changed)
    print(f"changed={len(changed)}")


if __name__ == "__main__":
    main()
