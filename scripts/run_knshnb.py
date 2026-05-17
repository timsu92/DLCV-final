from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from scripts.run_manager import create_run_dir, safe_run_name, write_run_manifest
except ModuleNotFoundError:
    from run_manager import create_run_dir, safe_run_name, write_run_manifest


DEBUG_CONFIG_PATH = "config/debug.yaml"
DEBUG_EXP_NAME = "debug"
DEBUG_RUN_NAME = "knshnb-debug"
DEBUG_NOTES = "Debug-sized knshnb train/inference smoke test"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="3rd-party/kaggle-happywhale-1st-place")
    parser.add_argument("--config-path")
    parser.add_argument("--exp-name")
    parser.add_argument("--out-base-dir")
    parser.add_argument("--in-base-dir", default="input")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base-dir", default="results")
    parser.add_argument("--timestamp")
    parser.add_argument("--source-manifest", default="data/source_manifest.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def prepare_debug_run(
    *,
    base_dir: Path,
    timestamp: str | None = None,
    in_base_dir: str = "input",
    save_checkpoint: bool = False,
    source_manifest: str = "data/source_manifest.yaml",
) -> tuple[Path, list[str]]:
    run_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir = create_run_dir(base_dir, run_timestamp, DEBUG_RUN_NAME)
    write_run_manifest(
        run_dir,
        run_name=safe_run_name(DEBUG_RUN_NAME),
        command=f"python -m src.train --config_path {DEBUG_CONFIG_PATH} --exp_name {DEBUG_EXP_NAME}",
        notes=DEBUG_NOTES,
        source_manifest=source_manifest,
    )
    command = build_train_command(
        config_path=DEBUG_CONFIG_PATH,
        exp_name=DEBUG_EXP_NAME,
        out_base_dir=str(run_dir / "predictions"),
        in_base_dir=in_base_dir,
        save_checkpoint=save_checkpoint,
    )
    return run_dir, command


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.debug:
        if args.out_base_dir is not None or args.config_path is not None or args.exp_name is not None:
            parser.error("--debug cannot be combined with --config-path, --exp-name, or --out-base-dir")
        run_dir, command = prepare_debug_run(
            base_dir=Path(args.base_dir),
            timestamp=args.timestamp,
            in_base_dir=args.in_base_dir,
            save_checkpoint=args.save_checkpoint,
            source_manifest=args.source_manifest,
        )
        print(run_dir)
    else:
        if args.config_path is None or args.exp_name is None or args.out_base_dir is None:
            parser.error(
                "the following arguments are required without --debug: --config-path, --exp-name, --out-base-dir"
            )
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
