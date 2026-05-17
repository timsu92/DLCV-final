from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_train_command(
    *,
    config_path: str,
    exp_name: str,
    out_base_dir: str,
    in_base_dir: str,
    save_checkpoint: bool,
) -> list[str]:
    command = [
        "python",
        "-m",
        "src.train",
        "--config_path",
        config_path,
        "--exp_name",
        exp_name,
        "--out_base_dir",
        out_base_dir,
        "--in_base_dir",
        in_base_dir,
    ]
    if save_checkpoint:
        command.append("--save_checkpoint")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="3rd-party/kaggle-happywhale-1st-place")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out-base-dir", required=True)
    parser.add_argument("--in-base-dir", default="input")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = build_train_command(
        config_path=args.config_path,
        exp_name=args.exp_name,
        out_base_dir=args.out_base_dir,
        in_base_dir=args.in_base_dir,
        save_checkpoint=args.save_checkpoint,
    )
    print(" ".join(command))

    if not args.dry_run:
        subprocess.run(command, cwd=Path(args.repo), check=True)


if __name__ == "__main__":
    main()
