# Happywhale First-Place Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project-side tooling and execution path needed to reproduce a first-place-level Happywhale submission with public Preferred Dolphin code, tracked sources, timestamped results, and report-ready experiment notes.

**Architecture:** Keep third-party training code in `3rd-party/` and add small project-owned scripts for provenance, run folders, data linking, submission validation, patch application, and command orchestration. Use knshnb as the primary executable pipeline, add charmq only after B6/B7 runs are stable, and record every external source in `data/source_manifest.yaml` before using it.

**Tech Stack:** Python standard library for project tooling and tests; shell commands through `rtk`; third-party reproduction environments for PyTorch, timm, albumentations, pandas, scikit-learn, PyTorch Lightning, and Hydra.

---

## File Structure

- Create `scripts/source_manifest.py`
  - Owns source manifest creation, append/update behavior, and SHA256 calculation.
- Create `scripts/run_manager.py`
  - Owns timestamped `results/<timestamp>-<run_name>/` folder creation and `run_manifest.yaml` writing.
- Create `scripts/prepare_data.py`
  - Owns symlink/copy preparation from `data/` into knshnb `input/` and charmq `happywhale_data/`.
- Create `scripts/validate_submission.py`
  - Validates row count, columns, prediction count, allowed IDs, and `new_individual` ratio.
- Create `scripts/patch_knshnb.py`
  - Applies the minimal knshnb gradient accumulation patch and records a patch note in `docs/experiments/`.
- Create `scripts/run_knshnb.py`
  - Builds and records debug/B6/B7 train commands, without hiding the original repo invocation.
- Create `scripts/run_ensemble.py`
  - Builds and records knshnb ensemble commands and validates generated submissions.
- Create `scripts/record_experiment.py`
  - Appends report-friendly experiment notes under `docs/experiments/`.
- Create `tests/`
  - Uses `python -m unittest`; no extra main-project test dependency is required.
- Create or update `data/source_manifest.yaml`
  - Canonical source log, updated during work.
- Create or update `docs/experiments/README.md`
  - Explains patch notes, score notes, and failed-run notes.

Do not move the submodules. Use symlinks or wrapper paths unless a third-party script requires a local file.

---

### Task 1: Source Manifest Tooling

**Files:**
- Create: `scripts/source_manifest.py`
- Create: `tests/test_source_manifest.py`
- Create: `data/source_manifest.yaml`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_source_manifest.py`:

```python
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.source_manifest import ensure_manifest, sha256_file, upsert_source


class SourceManifestTests(unittest.TestCase):
    def test_ensure_manifest_creates_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_manifest.yaml"
            ensure_manifest(path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("sources:", text)
            self.assertIn("generated_by: scripts/source_manifest.py", text)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            path.write_text("happywhale\n", encoding="utf-8")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"happywhale\n").hexdigest())

    def test_upsert_source_replaces_existing_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_manifest.yaml"
            ensure_manifest(path)
            upsert_source(
                path,
                source_id="kaggle_competition",
                name="Happywhale competition",
                source_type="competition_data",
                url="https://www.kaggle.com/competitions/happy-whale-and-dolphin",
                local_path="data",
                purpose="Official train/test data",
                notes="Already downloaded by user",
            )
            upsert_source(
                path,
                source_id="kaggle_competition",
                name="Happywhale competition data",
                source_type="competition_data",
                url="https://www.kaggle.com/competitions/happy-whale-and-dolphin",
                local_path="data",
                purpose="Official competition data",
                notes="Already downloaded by user",
            )
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("id: kaggle_competition"), 1)
            self.assertIn("name: Happywhale competition data", text)
            self.assertIn("purpose: Official competition data", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_source_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.source_manifest'`.

- [ ] **Step 3: Implement the manifest tool**

Create `scripts/source_manifest.py`:

```python
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
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "&", "*", "!", "|", ">", "'", '"', "%", "@", "`"]):
        return f'"{text}"'
    if text.lower() in {"true", "false", "null"}:
        return f'"{text}"'
    return text


def ensure_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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
    rows = [
        f"  - id: {yaml_scalar(source_id)}",
        f"    name: {yaml_scalar(name)}",
        f"    type: {yaml_scalar(source_type)}",
        f"    url: {yaml_scalar(url)}",
        f"    local_path: {yaml_scalar(local_path)}",
        f"    commit: {yaml_scalar(commit)}",
        f"    checksum_sha256: {yaml_scalar(checksum)}",
        f"    retrieved_at: {yaml_scalar(retrieved)}",
        f"    purpose: {yaml_scalar(purpose)}",
        f"    notes: {yaml_scalar(notes)}",
    ]
    return "\n".join(rows) + "\n"


def remove_existing_block(text: str, source_id: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    needle = f"  - id: {source_id}"
    quoted = f'  - id: "{source_id}"'
    while i < len(lines):
        line = lines[i]
        if line == needle or line == quoted:
            i += 1
            while i < len(lines) and not lines[i].startswith("  - id: "):
                i += 1
            continue
        out.append(line)
        i += 1
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
```

- [ ] **Step 4: Bootstrap required manifest entries**

Run:

```bash
rtk python scripts/source_manifest.py --id kaggle_competition --name "Happywhale competition data" --type competition_data --url "https://www.kaggle.com/competitions/happy-whale-and-dolphin" --local-path data --purpose "Official train/test/sample submission data" --notes "Already downloaded by user before reproduction work"
rtk python scripts/source_manifest.py --id kaggle_first_place_writeup --name "Preferred Dolphin first-place writeup" --type documentation --url "https://www.kaggle.com/competitions/happy-whale-and-dolphin/writeups/preferred-dolphin-1st-place-solution" --local-path docs/happywhale-1st-place-solution.zh-TW.md --purpose "Method reference and report citation" --notes "Kaggle page may require browser login; local Chinese summary is available"
rtk python scripts/source_manifest.py --id knshnb_repo --name "knshnb Happywhale first-place repo" --type git_submodule --url "https://github.com/knshnb/kaggle-happywhale-1st-place" --local-path 3rd-party/kaggle-happywhale-1st-place --commit 69690142177ecb69e8a4d1720837ec02c25071d6 --purpose "Primary training, inference, and ensemble pipeline" --notes "Preferred Dolphin teammate pipeline"
rtk python scripts/source_manifest.py --id charmq_repo --name "charmq Happywhale first-place repo" --type git_submodule --url "https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq" --local-path 3rd-party/kaggle-happywhale-1st-place-solution-charmq --commit c4ca5e2fe63cf5952fc8afaf0e85d7613a29d00b --purpose "Optional second pipeline for prediction-level ensemble" --notes "Preferred Dolphin teammate pipeline"
```

Expected: `data/source_manifest.yaml` exists and contains four unique entries.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_source_manifest.py -v
```

Expected: PASS, three tests.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add scripts/source_manifest.py tests/test_source_manifest.py data/source_manifest.yaml
rtk git commit -m "feat: add Happywhale source manifest tooling"
```

Expected: commit succeeds. If unrelated files are staged, use:

```bash
rtk git commit -m "feat: add Happywhale source manifest tooling" -- scripts/source_manifest.py tests/test_source_manifest.py data/source_manifest.yaml
```

---

### Task 2: Timestamped Run Folder Manager

**Files:**
- Create: `scripts/run_manager.py`
- Create: `tests/test_run_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_manager.py`:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.run_manager import create_run_dir, safe_run_name, write_run_manifest


class RunManagerTests(unittest.TestCase):
    def test_safe_run_name(self):
        self.assertEqual(safe_run_name("knshnb B7 debug"), "knshnb-b7-debug")
        self.assertEqual(safe_run_name("  B6/B7 ensemble  "), "b6-b7-ensemble")

    def test_create_run_dir_creates_expected_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(Path(tmp), "2026-05-17-010203", "knshnb-b6")
            self.assertTrue((run_dir / "logs").is_dir())
            self.assertTrue((run_dir / "configs").is_dir())
            self.assertTrue((run_dir / "predictions").is_dir())
            self.assertTrue((run_dir / "submissions").is_dir())
            self.assertTrue((run_dir / "metrics").is_dir())
            self.assertEqual(run_dir.name, "2026-05-17-010203-knshnb-b6")

    def test_write_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(Path(tmp), "2026-05-17-010203", "debug")
            write_run_manifest(
                run_dir,
                run_name="debug",
                command="python -m src.train",
                notes="small data check",
                source_manifest="data/source_manifest.yaml",
            )
            text = (run_dir / "run_manifest.yaml").read_text(encoding="utf-8")
            self.assertIn("run_name: debug", text)
            self.assertIn("command: python -m src.train", text)
            self.assertIn("source_manifest: data/source_manifest.yaml", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_run_manager.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_manager'`.

- [ ] **Step 3: Implement the run manager**

Create `scripts/run_manager.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_run_manager.py -v
```

Expected: PASS, three tests.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add scripts/run_manager.py tests/test_run_manager.py
rtk git commit -m "feat: add timestamped run manager"
```

Expected: commit succeeds.

---

### Task 3: Data Layout Preparation

**Files:**
- Create: `scripts/prepare_data.py`
- Create: `tests/test_prepare_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prepare_data.py`:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import LinkSpec, build_specs, prepare_links


class PrepareDataTests(unittest.TestCase):
    def test_build_specs_contains_required_knshnb_links(self):
        specs = build_specs(Path("data"), Path("3rd-party/kaggle-happywhale-1st-place/input"), "knshnb")
        pairs = {(str(s.source), str(s.target)) for s in specs}
        self.assertIn(("data/train.csv", "3rd-party/kaggle-happywhale-1st-place/input/train.csv"), pairs)
        self.assertIn(("data/sample_submission.csv", "3rd-party/kaggle-happywhale-1st-place/input/sample_submission.csv"), pairs)
        self.assertIn(("data/train_images", "3rd-party/kaggle-happywhale-1st-place/input/train_images"), pairs)
        self.assertIn(("data/test_images", "3rd-party/kaggle-happywhale-1st-place/input/test_images"), pairs)

    def test_prepare_links_dry_run_does_not_create_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "train.csv"
            dst = root / "repo" / "input" / "train.csv"
            src.parent.mkdir(parents=True)
            src.write_text("image,species,individual_id\n", encoding="utf-8")
            actions = prepare_links([LinkSpec(src, dst)], dry_run=True)
            self.assertIn("link", actions[0])
            self.assertFalse(dst.exists())

    def test_prepare_links_creates_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "train.csv"
            dst = root / "repo" / "input" / "train.csv"
            src.parent.mkdir(parents=True)
            src.write_text("image,species,individual_id\n", encoding="utf-8")
            prepare_links([LinkSpec(src, dst)], dry_run=False)
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.resolve(), src.resolve())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_prepare_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.prepare_data'`.

- [ ] **Step 3: Implement data preparation**

Create `scripts/prepare_data.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinkSpec:
    source: Path
    target: Path


COMMON_FILES = ["train.csv", "sample_submission.csv"]
COMMON_DIRS = ["train_images", "test_images"]


def build_specs(data_dir: Path, target_dir: Path, repo: str) -> list[LinkSpec]:
    if repo not in {"knshnb", "charmq"}:
        raise ValueError(f"repo must be knshnb or charmq, got {repo}")
    specs: list[LinkSpec] = []
    for name in COMMON_FILES + COMMON_DIRS:
        specs.append(LinkSpec(data_dir / name, target_dir / name))
    return specs


def prepare_links(specs: list[LinkSpec], dry_run: bool) -> list[str]:
    actions: list[str] = []
    for spec in specs:
        if not spec.source.exists():
            raise FileNotFoundError(spec.source)
        actions.append(f"link {spec.target} -> {spec.source}")
        if dry_run:
            continue
        spec.target.parent.mkdir(parents=True, exist_ok=True)
        if spec.target.exists() or spec.target.is_symlink():
            if spec.target.is_symlink() and spec.target.resolve() == spec.source.resolve():
                continue
            raise FileExistsError(f"Refusing to replace existing target: {spec.target}")
        spec.target.symlink_to(spec.source.resolve(), target_is_directory=spec.source.is_dir())
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", choices=["knshnb", "charmq"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = build_specs(Path(args.data_dir), Path(args.target_dir), args.repo)
    for action in prepare_links(specs, dry_run=args.dry_run):
        print(action)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run dry-run for knshnb**

Run:

```bash
rtk python scripts/prepare_data.py --repo knshnb --data-dir data --target-dir 3rd-party/kaggle-happywhale-1st-place/input --dry-run
```

Expected output contains links for `train.csv`, `sample_submission.csv`, `train_images`, and `test_images`.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_prepare_data.py -v
```

Expected: PASS, three tests.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add scripts/prepare_data.py tests/test_prepare_data.py
rtk git commit -m "feat: add Happywhale data layout preparation"
```

Expected: commit succeeds.

---

### Task 4: Submission Validator

**Files:**
- Create: `scripts/validate_submission.py`
- Create: `tests/test_validate_submission.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate_submission.py`:

```python
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.validate_submission import validate_submission


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class ValidateSubmissionTests(unittest.TestCase):
    def test_valid_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample_submission.csv"
            train = root / "train.csv"
            sub = root / "submission.csv"
            write_csv(sample, [{"image": "a.jpg", "predictions": "x y z q new_individual"}])
            write_csv(train, [{"image": "t.jpg", "species": "s", "individual_id": "x"}])
            write_csv(sub, [{"image": "a.jpg", "predictions": "x new_individual x x x"}])
            result = validate_submission(sub, sample, train)
            self.assertEqual(result.row_count, 1)
            self.assertEqual(result.new_individual_count, 1)

    def test_rejects_wrong_prediction_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample_submission.csv"
            train = root / "train.csv"
            sub = root / "submission.csv"
            write_csv(sample, [{"image": "a.jpg", "predictions": "x y z q new_individual"}])
            write_csv(train, [{"image": "t.jpg", "species": "s", "individual_id": "x"}])
            write_csv(sub, [{"image": "a.jpg", "predictions": "x new_individual"}])
            with self.assertRaises(ValueError):
                validate_submission(sub, sample, train)

    def test_rejects_unknown_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample_submission.csv"
            train = root / "train.csv"
            sub = root / "submission.csv"
            write_csv(sample, [{"image": "a.jpg", "predictions": "x y z q new_individual"}])
            write_csv(train, [{"image": "t.jpg", "species": "s", "individual_id": "x"}])
            write_csv(sub, [{"image": "a.jpg", "predictions": "unknown x x x x"}])
            with self.assertRaises(ValueError):
                validate_submission(sub, sample, train)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_validate_submission.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.validate_submission'`.

- [ ] **Step 3: Implement validator**

Create `scripts/validate_submission.py`:

```python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationResult:
    row_count: int
    new_individual_count: int
    new_individual_ratio: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_submission(submission_path: Path, sample_path: Path, train_path: Path) -> ValidationResult:
    submission = read_rows(submission_path)
    sample = read_rows(sample_path)
    train = read_rows(train_path)
    if not submission:
        raise ValueError("submission has no rows")
    if set(submission[0].keys()) != {"image", "predictions"}:
        raise ValueError(f"submission columns must be image,predictions, got {list(submission[0].keys())}")
    if len(submission) != len(sample):
        raise ValueError(f"row count mismatch: submission={len(submission)} sample={len(sample)}")
    expected_images = [row["image"] for row in sample]
    actual_images = [row["image"] for row in submission]
    if actual_images != expected_images:
        raise ValueError("submission image order does not match sample_submission.csv")
    allowed = {row["individual_id"] for row in train}
    allowed.add("new_individual")
    new_count = 0
    for row in submission:
        labels = row["predictions"].split()
        if len(labels) != 5:
            raise ValueError(f"{row['image']} has {len(labels)} predictions, expected 5")
        for label in labels:
            if label == "new_individual":
                new_count += 1
            if label not in allowed:
                raise ValueError(f"{row['image']} contains unknown label {label}")
    return ValidationResult(
        row_count=len(submission),
        new_individual_count=new_count,
        new_individual_ratio=new_count / (len(submission) * 5),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--train", default="data/train.csv")
    args = parser.parse_args()
    result = validate_submission(Path(args.submission), Path(args.sample), Path(args.train))
    print(f"rows={result.row_count}")
    print(f"new_individual_count={result.new_individual_count}")
    print(f"new_individual_ratio={result.new_individual_ratio:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_validate_submission.py -v
```

Expected: PASS, three tests.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add scripts/validate_submission.py tests/test_validate_submission.py
rtk git commit -m "feat: add Happywhale submission validator"
```

Expected: commit succeeds.

---

### Task 5: Experiment Notes and Patch Log Structure

**Files:**
- Create: `docs/experiments/README.md`
- Create: `scripts/record_experiment.py`
- Create: `tests/test_record_experiment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_record_experiment.py`:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.record_experiment import append_note, note_filename


class RecordExperimentTests(unittest.TestCase):
    def test_note_filename(self):
        self.assertEqual(note_filename("patch", "knshnb B7"), "patch-knshnb-b7.md")

    def test_append_note_creates_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = append_note(Path(tmp), kind="patch", title="knshnb B7", body="Added accumulation.")
            text = path.read_text(encoding="utf-8")
            self.assertIn("# patch: knshnb B7", text)
            self.assertIn("Added accumulation.", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_record_experiment.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.record_experiment'`.

- [ ] **Step 3: Implement experiment recorder**

Create `scripts/record_experiment.py`:

```python
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def note_filename(kind: str, title: str) -> str:
    return f"{slug(kind)}-{slug(title)}.md"


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
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
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
```

Create `docs/experiments/README.md`:

```markdown
# Happywhale Experiment Notes

This directory stores report-friendly notes created during reproduction.

Use these note kinds:

- `patch`: code changes to third-party or wrapper files, including reason and behavior impact.
- `score`: Kaggle public/private score observations and submission paths.
- `oom`: failed run settings, GPU memory behavior, and fallback used next.
- `source`: notes about external documents, datasets, pseudo labels, weights, or bbox files.
- `decision`: experiment choices that affect the final report.

Every note should name the related `results/<timestamp>-<run_name>/` folder when a run exists.
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_record_experiment.py -v
```

Expected: PASS, two tests.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add docs/experiments/README.md scripts/record_experiment.py tests/test_record_experiment.py
rtk git commit -m "feat: add Happywhale experiment note tooling"
```

Expected: commit succeeds.

---

### Task 6: knshnb Gradient Accumulation Patch Tool

**Files:**
- Create: `scripts/patch_knshnb.py`
- Create: `tests/test_patch_knshnb.py`
- Modify during execution: `3rd-party/kaggle-happywhale-1st-place/src/train.py`
- Modify during execution: `3rd-party/kaggle-happywhale-1st-place/config/default.yaml`
- Create during execution: `docs/experiments/patch-knshnb-gradient-accumulation.md`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_patch_knshnb.py`:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.patch_knshnb import patch_default_yaml, patch_train_py


class PatchKnshnbTests(unittest.TestCase):
    def test_patch_default_yaml_adds_accumulation_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default.yaml"
            path.write_text("batch_size: 8\nimage_size:\n- 768\n- 768\n", encoding="utf-8")
            patch_default_yaml(path)
            patch_default_yaml(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("accumulate_grad_batches: 1"), 1)

    def test_patch_train_py_adds_trainer_argument_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.py"
            path.write_text(
                "    trainer = Trainer(\n"
                "        gpus=torch.cuda.device_count(),\n"
                "        max_epochs=cfg[\"max_epochs\"],\n"
                "        logger=loggers,\n"
                "    )\n",
                encoding="utf-8",
            )
            patch_train_py(path)
            patch_train_py(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("accumulate_grad_batches=cfg.get(\"accumulate_grad_batches\", 1),"), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_patch_knshnb.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.patch_knshnb'`.

- [ ] **Step 3: Implement the patch tool**

Create `scripts/patch_knshnb.py`:

```python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


TRAIN_NEEDLE = '        logger=loggers,\n'
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_patch_knshnb.py -v
```

Expected: PASS, two tests.

- [ ] **Step 5: Apply patch to knshnb**

Run:

```bash
rtk python scripts/patch_knshnb.py
```

Expected output: `changed=2` on first run, `changed=0` on later runs. The patch note exists at `docs/experiments/patch-knshnb-gradient-accumulation.md`.

- [ ] **Step 6: Commit wrapper and patch record**

Run:

```bash
rtk git add scripts/patch_knshnb.py tests/test_patch_knshnb.py docs/experiments/patch-knshnb-gradient-accumulation.md 3rd-party/kaggle-happywhale-1st-place/src/train.py 3rd-party/kaggle-happywhale-1st-place/config/default.yaml
rtk git commit -m "fix: add gradient accumulation support for knshnb training"
```

Expected: commit succeeds. If submodule changes cannot be committed as normal files, commit the superproject gitlink state and keep the patch note; do not revert the patch.

---

### Task 7: knshnb Command Builder and Debug Run

**Files:**
- Create: `scripts/run_knshnb.py`
- Create: `tests/test_run_knshnb.py`
- Create during execution: `results/<timestamp>-knshnb-debug/`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_knshnb.py`:

```python
import unittest

from scripts.run_knshnb import build_train_command


class RunKnshnbTests(unittest.TestCase):
    def test_build_debug_command(self):
        command = build_train_command(
            config_path="config/debug.yaml",
            exp_name="debug",
            out_base_dir="/project/results/run/predictions",
            in_base_dir="input",
            save_checkpoint=False,
        )
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "src.train",
                "--config_path",
                "config/debug.yaml",
                "--exp_name",
                "debug",
                "--out_base_dir",
                "/project/results/run/predictions",
                "--in_base_dir",
                "input",
            ],
        )

    def test_build_command_with_checkpoint(self):
        command = build_train_command(
            config_path="config/efficientnet_b6.yaml",
            exp_name="b6",
            out_base_dir="/project/results/run/predictions",
            in_base_dir="input",
            save_checkpoint=True,
        )
        self.assertIn("--save_checkpoint", command)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_run_knshnb.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_knshnb'`.

- [ ] **Step 3: Implement command builder**

Create `scripts/run_knshnb.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_run_knshnb.py -v
```

Expected: PASS, two tests.

- [ ] **Step 5: Create debug run folder**

Run:

```bash
rtk python scripts/run_manager.py --run-name knshnb-debug --command "python -m src.train --config_path config/debug.yaml --exp_name debug" --notes "Debug-sized knshnb train/inference smoke test"
```

Expected: prints a path like `results/2026-05-17-153000-knshnb-debug`.

- [ ] **Step 6: Run dry-run command builder**

Replace `<RUN_DIR>` with the printed run folder:

```bash
rtk python scripts/run_knshnb.py --config-path config/debug.yaml --exp-name debug --out-base-dir /project/<RUN_DIR>/predictions --dry-run
```

Expected: prints the exact `python -m src.train ...` command and does not train.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add scripts/run_knshnb.py tests/test_run_knshnb.py
rtk git commit -m "feat: add knshnb training command wrapper"
```

Expected: commit succeeds.

---

### Task 8: Reproduction Environment Setup

**Files:**
- Create: `docs/experiments/environment-knshnb.md`
- Update: `data/source_manifest.yaml`
- Create during execution: reproduction environment outside main project dependency graph

- [ ] **Step 1: Record environment source intent before installing**

Run:

```bash
rtk python scripts/source_manifest.py --id knshnb_requirements --name "knshnb pinned Python requirements" --type dependency_spec --url "https://github.com/knshnb/kaggle-happywhale-1st-place/blob/master/requirements.txt" --local-path 3rd-party/kaggle-happywhale-1st-place/requirements.txt --purpose "Dependency source for knshnb reproduction environment" --notes "Used to create dedicated reproduction env, not main project env" --checksum-path 3rd-party/kaggle-happywhale-1st-place/requirements.txt
```

Expected: `data/source_manifest.yaml` contains `id: knshnb_requirements`.

- [ ] **Step 2: Create environment note**

Create `docs/experiments/environment-knshnb.md`:

```markdown
# environment: knshnb reproduction

Purpose: dedicated environment for `3rd-party/kaggle-happywhale-1st-place`.

This environment must not replace the main project environment.

Preferred Python: 3.10 or 3.11.

Pinned repo requirements:

- `3rd-party/kaggle-happywhale-1st-place/requirements.txt`

Installation command to try first:

```bash
uv venv .venv-knshnb --python 3.10
. .venv-knshnb/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r 3rd-party/kaggle-happywhale-1st-place/requirements.txt
python -m pip freeze > results/knshnb-freeze.txt
```

If Python 3.10 is not available locally, use Python 3.11 and record the change in this file.

If old PyTorch Lightning fails with installed torch, patch compatibility only after recording the failure and command output.
```

- [ ] **Step 3: Create the environment**

Run the commands from `docs/experiments/environment-knshnb.md`. Network access is required for package downloads.

Expected:

```bash
python -c "import torch, timm, pytorch_lightning, albumentations, pandas, sklearn; print(torch.__version__); print(timm.__version__)"
```

prints package versions without import errors.

- [ ] **Step 4: Commit environment documentation and manifest**

Run:

```bash
rtk git add docs/experiments/environment-knshnb.md data/source_manifest.yaml
rtk git commit -m "docs: record knshnb reproduction environment"
```

Expected: commit succeeds.

---

### Task 9: Data Linking and Debug Smoke Test

**Files:**
- Update: `docs/experiments/`
- Update: `data/source_manifest.yaml` if new bbox or pseudo-label artifacts are first used
- Create during execution: symlinks under `3rd-party/kaggle-happywhale-1st-place/input/`
- Create during execution: `results/<timestamp>-knshnb-debug/`

- [ ] **Step 1: Link official data into knshnb input**

Run:

```bash
rtk python scripts/prepare_data.py --repo knshnb --data-dir data --target-dir 3rd-party/kaggle-happywhale-1st-place/input
```

Expected: symlinks exist for `train.csv`, `sample_submission.csv`, `train_images`, and `test_images`.

- [ ] **Step 2: Check required knshnb artifact files**

Run:

```bash
rtk ls 3rd-party/kaggle-happywhale-1st-place/input
```

Expected: official data links plus existing bbox/encoder files needed by the repo. If `pseudo_labels/round2.csv` is absent, record that in `docs/experiments/source-missing-pseudo-label-round2.md` before running a no-pseudo baseline.

- [ ] **Step 3: Record bbox artifact sources before use**

Run one manifest command for each bbox file that will be used in debug or full training. Example for `fullbody_train_charm.csv`:

```bash
rtk python scripts/source_manifest.py --id fullbody_train_charm_knshnb --name "fullbody_train_charm.csv from knshnb repo" --type bbox_artifact --url "https://github.com/knshnb/kaggle-happywhale-1st-place" --local-path 3rd-party/kaggle-happywhale-1st-place/input/fullbody_train_charm.csv --purpose "Fullbody charm bbox crop source for knshnb training/inference" --notes "Repo artifact; source lineage described in public solution docs" --checksum-path 3rd-party/kaggle-happywhale-1st-place/input/fullbody_train_charm.csv
```

Expected: manifest entry is present before the file is used.

- [ ] **Step 4: Create debug run folder**

Run:

```bash
rtk python scripts/run_manager.py --run-name knshnb-debug --command "python -m src.train --config_path config/debug.yaml --exp_name debug" --notes "Debug run before B6/B7"
```

Expected: prints `<RUN_DIR>`.

- [ ] **Step 5: Run debug training in knshnb env**

Activate the knshnb env, replace `<RUN_DIR>`, then run:

```bash
python scripts/run_knshnb.py --config-path config/debug.yaml --exp-name debug --out-base-dir /project/<RUN_DIR>/predictions
```

Expected: command runs from `3rd-party/kaggle-happywhale-1st-place`, creates prediction artifacts under `<RUN_DIR>/predictions/debug/`, or fails with a recorded dependency/runtime error.

- [ ] **Step 6: Record debug outcome**

Run:

```bash
rtk python scripts/record_experiment.py --kind decision --title knshnb-debug --body "Debug run completed or failed. Related run folder: <RUN_DIR>. Record exact failure text here before retrying with patches."
```

Expected: `docs/experiments/decision-knshnb-debug.md` exists.

- [ ] **Step 7: Commit debug preparation records**

Run:

```bash
rtk git add data/source_manifest.yaml docs/experiments
rtk git commit -m "docs: record knshnb debug preparation"
```

Expected: commit succeeds if docs or manifest changed. Do not commit large result artifacts.

---

### Task 10: B6 and B7 Training Execution

**Files:**
- Update: `data/source_manifest.yaml`
- Update: `docs/experiments/`
- Create during execution: `results/<timestamp>-knshnb-b6/`
- Create during execution: `results/<timestamp>-knshnb-b7/`

- [ ] **Step 1: Record pseudo-label source before using it**

If `3rd-party/kaggle-happywhale-1st-place/input/pseudo_labels/round2.csv` exists, run:

```bash
rtk python scripts/source_manifest.py --id pseudo_labels_round2_knshnb --name "Preferred Dolphin round2 pseudo labels" --type pseudo_label --url "https://github.com/knshnb/kaggle-happywhale-1st-place" --local-path 3rd-party/kaggle-happywhale-1st-place/input/pseudo_labels/round2.csv --purpose "Pseudo labels used by knshnb B6/B7 reproduction configs" --notes "Public solution artifact if present in repo or downloaded source" --checksum-path 3rd-party/kaggle-happywhale-1st-place/input/pseudo_labels/round2.csv
```

Expected: manifest has `pseudo_labels_round2_knshnb`. If file is missing, create `docs/experiments/source-missing-pseudo-label-round2.md` and set `pseudo_label:` to empty in copied config for the first baseline.

- [ ] **Step 2: Create B6 run folder**

Run:

```bash
rtk python scripts/run_manager.py --run-name knshnb-b6 --command "python -m src.train --config_path config/efficientnet_b6.yaml --exp_name b6" --notes "B6 1024 reproduction target"
```

Expected: prints `<B6_RUN_DIR>`.

- [ ] **Step 3: Run B6**

Activate the knshnb env, replace `<B6_RUN_DIR>`, then run:

```bash
python scripts/run_knshnb.py --config-path config/efficientnet_b6.yaml --exp-name b6 --out-base-dir /project/<B6_RUN_DIR>/predictions --save-checkpoint
```

Expected: B6 training and inference start. Successful run creates train/test crop prediction artifacts under `<B6_RUN_DIR>/predictions/b6/-1/`.

- [ ] **Step 4: If B6 OOM occurs, record and retry with accumulation**

Run:

```bash
rtk python scripts/record_experiment.py --kind oom --title knshnb-b6 --body "B6 OOM at batch_size 6. Related run folder: <B6_RUN_DIR>. Next retry: batch_size 3, accumulate_grad_batches 2."
```

Then copy `config/efficientnet_b6.yaml` to `<B6_RUN_DIR>/configs/efficientnet_b6_batch3_accum2.yaml` and set:

```yaml
batch_size: 3
accumulate_grad_batches: 2
```

Run:

```bash
python scripts/run_knshnb.py --config-path /project/<B6_RUN_DIR>/configs/efficientnet_b6_batch3_accum2.yaml --exp-name b6-batch3-accum2 --out-base-dir /project/<B6_RUN_DIR>/predictions --save-checkpoint
```

Expected: retry starts with smaller micro-batch.

- [ ] **Step 5: Create B7 run folder**

Run:

```bash
rtk python scripts/run_manager.py --run-name knshnb-b7 --command "python -m src.train --config_path config/efficientnet_b7.yaml --exp_name b7" --notes "B7 1024 reproduction target"
```

Expected: prints `<B7_RUN_DIR>`.

- [ ] **Step 6: Run B7**

Activate the knshnb env, replace `<B7_RUN_DIR>`, then run:

```bash
python scripts/run_knshnb.py --config-path config/efficientnet_b7.yaml --exp-name b7 --out-base-dir /project/<B7_RUN_DIR>/predictions --save-checkpoint
```

Expected: B7 training and inference start. Successful run creates train/test crop prediction artifacts under `<B7_RUN_DIR>/predictions/b7/-1/`.

- [ ] **Step 7: If B7 OOM occurs, record and retry with accumulation**

Run:

```bash
rtk python scripts/record_experiment.py --kind oom --title knshnb-b7 --body "B7 OOM at batch_size 4. Related run folder: <B7_RUN_DIR>. Next retry: batch_size 2, accumulate_grad_batches 2."
```

Then copy `config/efficientnet_b7.yaml` to `<B7_RUN_DIR>/configs/efficientnet_b7_batch2_accum2.yaml` and set:

```yaml
batch_size: 2
accumulate_grad_batches: 2
```

Run:

```bash
python scripts/run_knshnb.py --config-path /project/<B7_RUN_DIR>/configs/efficientnet_b7_batch2_accum2.yaml --exp-name b7-batch2-accum2 --out-base-dir /project/<B7_RUN_DIR>/predictions --save-checkpoint
```

Expected: retry starts with smaller micro-batch.

- [ ] **Step 8: Commit experiment records**

Run:

```bash
rtk git add data/source_manifest.yaml docs/experiments
rtk git commit -m "docs: record knshnb B6 and B7 training setup"
```

Expected: commit succeeds if docs or manifest changed. Do not commit checkpoints or prediction artifacts.

---

### Task 11: Ensemble Wrapper and Submission Validation

**Files:**
- Create: `scripts/run_ensemble.py`
- Create: `tests/test_run_ensemble.py`
- Create during execution: `results/<timestamp>-knshnb-b6-b7-ensemble/`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_ensemble.py`:

```python
import unittest

from scripts.run_ensemble import build_ensemble_command


class RunEnsembleTests(unittest.TestCase):
    def test_build_ensemble_command(self):
        command = build_ensemble_command(
            model_dirs=["/project/results/b6/predictions/b6/-1", "/project/results/b7/predictions/b7/-1"],
            out_prefix="/project/results/ensemble/submissions/b6-b7",
        )
        self.assertEqual(
            command,
            [
                "python",
                "-m",
                "src.ensemble",
                "--model_dirs",
                "/project/results/b6/predictions/b6/-1",
                "/project/results/b7/predictions/b7/-1",
                "--out_prefix",
                "/project/results/ensemble/submissions/b6-b7",
            ],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
rtk python -m unittest tests/test_run_ensemble.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_ensemble'`.

- [ ] **Step 3: Implement ensemble command builder**

Create `scripts/run_ensemble.py`:

```python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_ensemble_command(*, model_dirs: list[str], out_prefix: str) -> list[str]:
    return ["python", "-m", "src.ensemble", "--model_dirs", *model_dirs, "--out_prefix", out_prefix]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="3rd-party/kaggle-happywhale-1st-place")
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = build_ensemble_command(model_dirs=args.model_dir, out_prefix=args.out_prefix)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=Path(args.repo), check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
rtk python -m unittest tests/test_run_ensemble.py -v
```

Expected: PASS, one test.

- [ ] **Step 5: Create ensemble run folder**

Run:

```bash
rtk python scripts/run_manager.py --run-name knshnb-b6-b7-ensemble --command "python -m src.ensemble --model_dirs b6 b7" --notes "B6+B7 KNN/logit ensemble"
```

Expected: prints `<ENSEMBLE_RUN_DIR>`.

- [ ] **Step 6: Run ensemble**

Activate the knshnb env, replace run dirs, then run:

```bash
python scripts/run_ensemble.py --model-dir /project/<B6_RUN_DIR>/predictions/b6/-1 --model-dir /project/<B7_RUN_DIR>/predictions/b7/-1 --out-prefix /project/<ENSEMBLE_RUN_DIR>/submissions/b6-b7
```

Expected: ensemble creates a CSV under `<ENSEMBLE_RUN_DIR>/submissions/`.

- [ ] **Step 7: Validate submission**

Run:

```bash
rtk python scripts/validate_submission.py <ENSEMBLE_RUN_DIR>/submissions/<GENERATED_SUBMISSION>.csv
```

Expected output contains `rows=27956` and a `new_individual_ratio`.

- [ ] **Step 8: Record score if submitted**

After Kaggle submission, run:

```bash
rtk python scripts/record_experiment.py --kind score --title b6-b7-ensemble --body "Submission: <ENSEMBLE_RUN_DIR>/submissions/<GENERATED_SUBMISSION>.csv. Public LB: <PUBLIC_SCORE>. Private LB unavailable until competition rerun context. Notes: B6+B7 knshnb ensemble."
```

Expected: `docs/experiments/score-b6-b7-ensemble.md` exists. Replace `<PUBLIC_SCORE>` with the actual number before committing.

- [ ] **Step 9: Commit wrapper and score notes**

Run:

```bash
rtk git add scripts/run_ensemble.py tests/test_run_ensemble.py docs/experiments
rtk git commit -m "feat: add knshnb ensemble wrapper"
```

Expected: commit succeeds. Do not commit generated submissions unless the user asks to version them.

---

### Task 12: charmq Integration Decision Gate

**Files:**
- Update: `data/source_manifest.yaml`
- Create: `docs/experiments/decision-charmq-integration.md`
- Create during execution: charmq run folders only if B6+B7 is stable

- [ ] **Step 1: Record charmq dependency source before use**

Run:

```bash
rtk python scripts/source_manifest.py --id charmq_requirements --name "charmq pinned Python requirements" --type dependency_spec --url "https://github.com/tyamaguchi17/kaggle-happywhale-1st-place-solution-charmq/blob/master/requirements.txt" --local-path 3rd-party/kaggle-happywhale-1st-place-solution-charmq/requirements.txt --purpose "Dependency source for optional charmq reproduction environment" --notes "Use only after knshnb B6/B7 ensemble is stable" --checksum-path 3rd-party/kaggle-happywhale-1st-place-solution-charmq/requirements.txt
```

Expected: manifest contains `charmq_requirements`.

- [ ] **Step 2: Write decision note**

Create `docs/experiments/decision-charmq-integration.md`:

```markdown
# decision: charmq integration

Prerequisite: knshnb B6/B7 ensemble has a valid submission and recorded score.

Use charmq only if at least one of these is true:

- B6+B7 score is below target and there are at least three full days left.
- B7 failed and charmq B7 or another charmq model looks faster to adapt.
- Existing charmq artifacts can be converted into knshnb ensemble format with a small adapter.

Do not merge charmq training code into knshnb. Integration target is prediction artifacts consumed by knshnb ensemble.

If charmq starts, record:

- environment command
- dependency versions
- model selected
- bbox variants inferred
- output schema
- adapter changes, if any
```

- [ ] **Step 3: Link official data into charmq layout**

Run:

```bash
rtk python scripts/prepare_data.py --repo charmq --data-dir data --target-dir 3rd-party/kaggle-happywhale-1st-place-solution-charmq/happywhale_data --dry-run
```

Expected: dry-run prints official data links. Run without `--dry-run` only when deciding to execute charmq.

- [ ] **Step 4: Commit decision artifacts**

Run:

```bash
rtk git add data/source_manifest.yaml docs/experiments/decision-charmq-integration.md
rtk git commit -m "docs: record charmq integration decision gate"
```

Expected: commit succeeds.

---

## Plan Self-Review

Spec coverage:

- Source manifest updated during work: Task 1, Task 8, Task 9, Task 10, Task 12.
- User-approved directories: Task 1 creates `data/source_manifest.yaml`; Task 2 creates `results/<timestamp>-<run_name>/`; Task 5 creates `docs/experiments/`; tasks use `data/external/` when downloads are needed.
- Main knshnb-first approach: Task 6 through Task 11.
- Batch size and gradient accumulation: Task 6 and Task 10.
- Submission validation: Task 4 and Task 11.
- charmq as optional ensemble source: Task 12.
- Report-ready experiment records: Task 5, Task 8 through Task 12.

Placeholder scan:

- The angle-bracket tokens in command examples (`<RUN_DIR>`, `<B6_RUN_DIR>`, `<PUBLIC_SCORE>`) are execution-time values explicitly produced by earlier steps or by Kaggle submission. They are not missing design details.
- No task uses unfinished-marker wording or unspecified error handling.

Type consistency:

- `Path` is used consistently for filesystem functions.
- Command builder functions return `list[str]`.
- Validation returns `ValidationResult`.
- Manifest functions use `Path` and plain strings.
