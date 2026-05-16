from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


RUN_CHILDREN = ["logs", "configs", "predictions", "submissions", "metrics"]


def safe_run_name(name: str) -> str:
    lowered = name.strip().lower()
    replaced = re.sub(r"[^a-z0-9]+", "-", lowered)
    return replaced.strip("-")


def create_run_dir(base_dir: Path, timestamp: str, run_name: str) -> Path:
    safe_name = safe_run_name(run_name)
    run_dir = base_dir / f"{timestamp}-{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in RUN_CHILDREN:
        (run_dir / child).mkdir()
    return run_dir


def write_run_manifest(
    run_dir: Path,
    *,
    run_name: str,
    command: str,
    notes: str,
    source_manifest: str,
) -> None:
    text = "\n".join(
        [
            f"run_name: {run_name}",
            f"created_at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
            f"command: {command}",
            f"source_manifest: {source_manifest}",
            f"notes: {notes}",
            "",
        ]
    )
    (run_dir / "run_manifest.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="results")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--command", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--source-manifest", default="data/source_manifest.yaml")
    args = parser.parse_args()
    run_dir = create_run_dir(Path(args.base_dir), args.timestamp, args.run_name)
    write_run_manifest(
        run_dir,
        run_name=safe_run_name(args.run_name),
        command=args.command,
        notes=args.notes,
        source_manifest=args.source_manifest,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
