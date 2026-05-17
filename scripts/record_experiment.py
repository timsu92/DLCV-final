from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "untitled"


def title_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def note_filename(kind: str, title: str) -> str:
    return f"{slug(kind)}-{slug(title)}-{title_digest(title)}.md"


def append_note(base_dir: Path, *, kind: str, title: str, body: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / note_filename(kind, title)
    prefix = f"# {kind}: {title}\n\n" if not path.exists() else "\n---\n\n"
    entry = "\n".join(
        [
            prefix.rstrip(),
            "",
            f"Recorded at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
            "",
            body.rstrip(),
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="docs/experiments")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", required=True)
    args = parser.parse_args()
    path = append_note(Path(args.base_dir), kind=args.kind, title=args.title, body=args.body)
    print(path)


if __name__ == "__main__":
    main()
