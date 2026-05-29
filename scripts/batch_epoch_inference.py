"""Run inference + ensemble for every epoch checkpoint, saving a submission CSV per epoch.
Activate Python virtualenv first to avoid crash with system Python packages"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def read_ckpt_epoch(ckpt_path: Path) -> int:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt["epoch"])


def find_epoch_checkpoints(
    ckpt_dir: Path, include_last: bool = False
) -> list[tuple[int, Path]]:
    pattern = re.compile(r"epoch=epoch=(\d+)\.ckpt$")
    checkpoints = []
    for f in ckpt_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            checkpoints.append((int(m.group(1)), f))
    if include_last:
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            epoch_num = read_ckpt_epoch(last)
            # only add if not already covered by a numbered checkpoint
            if not any(n == epoch_num for n, _ in checkpoints):
                checkpoints.append((epoch_num, last))
    return sorted(checkpoints, reverse=True)  # latest first


def build_inference_cmd(
    ckpt_path: Path,
    out_dir: Path,
    config_path: Path,
    python: str,
    batch_size: int,
    nproc: int,
) -> list[str]:
    if nproc > 1:
        launcher = ["torchrun", f"--nproc_per_node={nproc}", "-m"]
    else:
        launcher = [python, "-m"]
    return [
        *launcher,
        "src.rerun_inference",
        "--ckpt",
        str(ckpt_path),
        "--out_dir",
        str(out_dir),
        "--config",
        str(config_path),
        "--batch_size",
        str(batch_size),
    ]


def build_ensemble_cmd(model_dir: Path, out_prefix: str, python: str) -> list[str]:
    return [
        python,
        "-m",
        "src.ensemble",
        "--model_dirs",
        str(model_dir),
        "--out_prefix",
        out_prefix,
    ]


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch per-epoch inference for B6/B7 checkpoints"
    )
    parser.add_argument(
        "--ckpt-dir",
        required=True,
        help="Directory containing epoch=epoch=N.ckpt files and last.ckpt",
    )
    parser.add_argument("--config", required=True, help="Model config YAML path")
    parser.add_argument("--repo", default="3rd-party/kaggle-happywhale-1st-place")
    parser.add_argument(
        "--python", default=None, help="Python interpreter (default: sys.executable)"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="GPUs per node for inference (1=python, >1=torchrun)",
    )
    parser.add_argument(
        "--submission-dir",
        default=None,
        help="Where to copy submission CSVs (default: <run>/submissions/)",
    )
    parser.add_argument(
        "--out-prefix",
        default="b6",
        help="Prefix for submission CSV filenames (e.g. 'b6' → 'b6-epoch39-...')",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Only process epochs >= this 0-indexed value",
    )
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=None,
        help="Only process epochs <= this 0-indexed value",
    )
    parser.add_argument(
        "--include-last",
        action="store_true",
        help="Also run inference on last.ckpt if its epoch isn't already covered",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = os.path.abspath(args.python or sys.executable)
    repo = Path(args.repo).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    config_path = Path(args.config).resolve()

    if args.submission_dir:
        submission_dir = Path(args.submission_dir)
    else:
        # ckpt_dir is .../predictions/<exp>/<fold>/  → run root is 3 levels up
        submission_dir = ckpt_dir.parent.parent.parent / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = find_epoch_checkpoints(ckpt_dir, include_last=args.include_last)
    if not checkpoints:
        print("No epoch=epoch=N.ckpt files found in", ckpt_dir, file=sys.stderr)
        sys.exit(1)

    if args.start_epoch is not None:
        checkpoints = [(n, p) for n, p in checkpoints if n >= args.start_epoch]
    if args.end_epoch is not None:
        checkpoints = [(n, p) for n, p in checkpoints if n <= args.end_epoch]

    print(f"Will process {len(checkpoints)} checkpoints (latest first):")
    for n, p in checkpoints:
        print(f"  epoch={n}  {p.name}")
    print()

    repo_submission_dir = repo / "submission"

    for i, (epoch_num, ckpt_path) in enumerate(checkpoints):
        human_epoch = epoch_num + 1  # 0-indexed PL epoch → human-readable
        out_prefix = f"{args.out_prefix}-epoch{epoch_num}"
        out_dir = ckpt_dir / "by-epoch" / f"epoch{epoch_num}"

        print(
            f"[{i + 1}/{len(checkpoints)}] epoch={epoch_num} (training epoch #{human_epoch})"
        )

        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        inf_cmd = build_inference_cmd(
            ckpt_path,
            out_dir,
            config_path,
            python,
            args.batch_size,
            args.nproc_per_node,
        )
        ens_cmd = build_ensemble_cmd(out_dir, out_prefix, python)

        if args.dry_run:
            print("  [inference]", " ".join(inf_cmd))
            print("  [ensemble] ", " ".join(ens_cmd))
            continue

        print("  → inference")
        run(inf_cmd, repo)

        print("  → ensemble")
        run(ens_cmd, repo)

        # Copy generated CSVs to the run's submissions dir
        copied = []
        for csv_file in repo_submission_dir.glob(f"{out_prefix}-*.csv"):
            dest = submission_dir / csv_file.name
            shutil.copy2(csv_file, dest)
            copied.append(dest.name)
        print(f"  → saved: {', '.join(copied) or '(none found)'}")
        print()

    if not args.dry_run:
        print(f"Done. Submission CSVs are in: {submission_dir}")
        print("Upload order (latest epoch first) matches the order processed above.")


if __name__ == "__main__":
    main()
