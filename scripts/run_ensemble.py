from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from scripts.run_knshnb import build_execution_command, command_to_text
except ModuleNotFoundError:
    from run_knshnb import build_execution_command, command_to_text


def build_ensemble_command(*, model_dirs: list[str], out_prefix: str) -> list[str]:
    return [
        "python",
        "-m",
        "src.ensemble",
        "--model_dirs",
        *model_dirs,
        "--out_prefix",
        out_prefix,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="3rd-party/kaggle-happywhale-1st-place")
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command = build_ensemble_command(
        model_dirs=args.model_dir, out_prefix=args.out_prefix
    )
    command = build_execution_command(command)
    print(command_to_text(command))
    if not args.dry_run:
        subprocess.run(command, cwd=Path(args.repo), check=True)


if __name__ == "__main__":
    main()
