from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


TRAIN_NEEDLE = "        logger=loggers,\n"
TRAIN_INSERT = '        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),\n'
YAML_LINE = "accumulate_grad_batches: 1\n"


def patch_default_yaml(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "accumulate_grad_batches:" in text:
        return False
    path.write_text(text.rstrip() + "\n" + YAML_LINE, encoding="utf-8")
    return True


def patch_train_py(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if TRAIN_INSERT.strip() in text:
        return False
    if TRAIN_NEEDLE not in text:
        raise ValueError(f"Could not find Trainer logger line in {path}")
    path.write_text(text.replace(TRAIN_NEEDLE, TRAIN_NEEDLE + TRAIN_INSERT, 1), encoding="utf-8")
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
    write_patch_note(Path(args.note), changed)
    print(f"changed={len(changed)}")


if __name__ == "__main__":
    main()
