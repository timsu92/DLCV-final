from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path


HEADER = """# Canonical source manifest for Happywhale reproduction.
# Update this file before first using any external source, document, artifact, bbox, pseudo label, or weight.
generated_by: scripts/source_manifest.py
sources:
"""


def yaml_scalar(value: str) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    if text == "":
        return '""'
    if any(
        ch in text
        for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|", ">", "'", '"', "%", "@", "`"]
    ):
        return f'"{text}"'
    if text.lower() in {"true", "false", "null"}:
        return f'"{text}"'
    return text


def yaml_field(prefix: str, value: str, indent: str) -> list[str]:
    text = str(value)
    if "\n" not in text:
        return [f"{prefix}{yaml_scalar(text)}"]

    lines = [f"{prefix}|-"]
    for part in text.splitlines():
        lines.append(f"{indent}{part}")
    if text.endswith("\n"):
        lines.append(indent)
    return lines


def ensure_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_block(
    *,
    source_id: str,
    name: str,
    source_type: str,
    url: str,
    local_path: str,
    purpose: str,
    notes: str,
    commit: str = "",
    checksum: str = "",
    retrieved_at: str | None = None,
) -> str:
    retrieved = retrieved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    rows.extend(yaml_field("  - id: ", source_id, "      "))
    rows.extend(yaml_field("    name: ", name, "      "))
    rows.extend(yaml_field("    type: ", source_type, "      "))
    rows.extend(yaml_field("    url: ", url, "      "))
    rows.extend(yaml_field("    local_path: ", local_path, "      "))
    rows.extend(yaml_field("    commit: ", commit, "      "))
    rows.extend(yaml_field("    checksum_sha256: ", checksum, "      "))
    rows.extend(yaml_field("    retrieved_at: ", retrieved, "      "))
    rows.extend(yaml_field("    purpose: ", purpose, "      "))
    rows.extend(yaml_field("    notes: ", notes, "      "))
    return "\n".join(rows) + "\n"


def remove_existing_block(text: str, source_id: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    serialized_id = yaml_scalar(source_id)
    needle = f"  - id: {serialized_id}"

    while index < len(lines):
        line = lines[index]
        if line == needle:
            index += 1
            while index < len(lines) and not lines[index].startswith("  - id: "):
                index += 1
            continue
        out.append(line)
        index += 1

    return "\n".join(out).rstrip() + "\n"


def upsert_source(
    manifest_path: Path,
    *,
    source_id: str,
    name: str,
    source_type: str,
    url: str,
    local_path: str,
    purpose: str,
    notes: str,
    commit: str = "",
    checksum: str = "",
) -> None:
    ensure_manifest(manifest_path)
    text = manifest_path.read_text(encoding="utf-8")
    text = remove_existing_block(text, source_id)
    text = text.rstrip() + "\n" + source_block(
        source_id=source_id,
        name=name,
        source_type=source_type,
        url=url,
        local_path=local_path,
        purpose=purpose,
        notes=notes,
        commit=commit,
        checksum=checksum,
    )
    manifest_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/source_manifest.yaml")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--url", default="")
    parser.add_argument("--local-path", default="")
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--checksum-path", default="")
    args = parser.parse_args()

    checksum = sha256_file(Path(args.checksum_path)) if args.checksum_path else ""
    upsert_source(
        Path(args.manifest),
        source_id=args.id,
        name=args.name,
        source_type=args.type,
        url=args.url,
        local_path=args.local_path,
        purpose=args.purpose,
        notes=args.notes,
        commit=args.commit,
        checksum=checksum,
    )


if __name__ == "__main__":
    main()
