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
            self.assertEqual(
                sha256_file(path), hashlib.sha256(b"happywhale\n").hexdigest()
            )

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

    def test_upsert_source_serializes_multiline_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_manifest.yaml"
            ensure_manifest(path)
            upsert_source(
                path,
                source_id="multiline_notes",
                name="Multiline notes source",
                source_type="documentation",
                url="https://example.com/source",
                local_path="docs/source.md",
                purpose="Exercise multiline serialization",
                notes="line one\nline two",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("notes: |-\n", text)
            self.assertIn("      line one\n", text)
            self.assertIn("      line two\n", text)

    def test_upsert_source_replaces_existing_block_with_escaped_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source_manifest.yaml"
            ensure_manifest(path)
            source_id = 'repo"mirror'
            upsert_source(
                path,
                source_id=source_id,
                name="Quoted id source",
                source_type="documentation",
                url="https://example.com/source",
                local_path="docs/source.md",
                purpose="First write",
                notes="first",
            )
            upsert_source(
                path,
                source_id=source_id,
                name="Quoted id source updated",
                source_type="documentation",
                url="https://example.com/source",
                local_path="docs/source.md",
                purpose="Second write",
                notes="second",
            )
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count('id: "repo\\"mirror"'), 1)
            self.assertIn("name: Quoted id source updated", text)
            self.assertIn("purpose: Second write", text)


if __name__ == "__main__":
    unittest.main()
