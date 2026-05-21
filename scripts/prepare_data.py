"""準備數據的腳本，通過創建符號鏈接將數據集從原始位置鏈接到指定的目錄。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

COMMON_ENTRIES = (
    "train.csv",
    "sample_submission.csv",
    "train_images",
    "test_images",
)
VALID_REPOS = {"knshnb", "charmq"}


@dataclass(frozen=True)
class LinkSpec:
    source: Path
    target: Path


def _validate_target(spec: LinkSpec) -> None:
    if spec.target.is_symlink():
        if spec.target.resolve() == spec.source.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing target: {spec.target}")
    if spec.target.exists():
        raise FileExistsError(f"Refusing to replace existing target: {spec.target}")


def build_specs(data_dir: Path, target_dir: Path, repo: str) -> list[LinkSpec]:
    if repo not in VALID_REPOS:
        raise ValueError(f"repo must be knshnb or charmq, got {repo}")
    return [LinkSpec(data_dir / name, target_dir / name) for name in COMMON_ENTRIES]


def prepare_links(specs: list[LinkSpec], dry_run: bool) -> list[str]:
    actions: list[str] = []
    for spec in specs:
        if not spec.source.exists():
            raise FileNotFoundError(spec.source)
        actions.append(f"link {spec.target} -> {spec.source}")
        _validate_target(spec)

    if dry_run:
        return actions

    for spec in specs:
        if spec.target.is_symlink() and spec.target.resolve() == spec.source.resolve():
            continue
        spec.target.parent.mkdir(parents=True, exist_ok=True)
        spec.target.symlink_to(spec.source.resolve(), target_is_directory=spec.source.is_dir())

    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", choices=sorted(VALID_REPOS), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = build_specs(Path(args.data_dir), Path(args.target_dir), args.repo)
    for action in prepare_links(specs, dry_run=args.dry_run):
        print(action)


if __name__ == "__main__":
    main()
