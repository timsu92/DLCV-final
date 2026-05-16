from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


RUN_CHILDREN = ["logs", "configs", "predictions", "submissions", "metrics"]


def _slugify(value: str, *, fallback: str) -> str:
    lowered = value.strip().lower()
    replaced = re.sub(r"[^a-z0-9]+", "-", lowered)
    cleaned = replaced.strip("-")
    if cleaned:
        return cleaned
    return fallback


def _safe_timestamp(timestamp: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", timestamp.strip())
    normalized = normalized.strip("-")
    normalized = normalized.replace("..", "-")
    if normalized:
        return normalized
    return "timestamp"


def _yaml_scalar(value: str) -> str:
    text = str(value)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in normalized:
        lines = normalized.split("\n")
        return "|\n" + "\n".join(f"  {line}" for line in lines)
    if normalized == "":
        return '""'
    escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
    if escaped != normalized or re.search(r"[:#\[\]{}&,*!|>'\"%@`]|^\s|\s$", normalized):
        return f'"{escaped}"'
    return normalized


def safe_run_name(name: str) -> str:
    return _slugify(name, fallback="run")


def create_run_dir(base_dir: Path, timestamp: str, run_name: str) -> Path:
    safe_timestamp = _safe_timestamp(timestamp)
    safe_name = safe_run_name(run_name)
    run_dir = base_dir / f"{safe_timestamp}-{safe_name}"
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
            f"run_name: {_yaml_scalar(run_name)}",
            f"created_at: {_yaml_scalar(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))}",
            f"command: {_yaml_scalar(command)}",
            f"source_manifest: {_yaml_scalar(source_manifest)}",
            f"notes: {_yaml_scalar(notes)}",
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
