import tempfile
import unittest
from pathlib import Path

from scripts.run_manager import create_run_dir, safe_run_name, write_run_manifest


class RunManagerTests(unittest.TestCase):
    def test_safe_run_name(self):
        self.assertEqual(safe_run_name("knshnb B7 debug"), "knshnb-b7-debug")
        self.assertEqual(safe_run_name("  B6/B7 ensemble  "), "b6-b7-ensemble")
        self.assertEqual(safe_run_name(" !!! "), "run")

    def test_create_run_dir_creates_expected_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(Path(tmp), "2026-05-17-010203", "knshnb-b6")
            self.assertTrue((run_dir / "logs").is_dir())
            self.assertTrue((run_dir / "configs").is_dir())
            self.assertTrue((run_dir / "predictions").is_dir())
            self.assertTrue((run_dir / "submissions").is_dir())
            self.assertTrue((run_dir / "metrics").is_dir())
            self.assertEqual(run_dir.name, "2026-05-17-010203-knshnb-b6")

    def test_create_run_dir_normalizes_unsafe_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            run_dir = create_run_dir(base_dir, "../2026/05/17 010203", "debug")
            self.assertEqual(run_dir.parent, base_dir)
            self.assertEqual(run_dir.name, "2026-05-17-010203-debug")

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

    def test_write_run_manifest_escapes_empty_and_multiline_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_dir(Path(tmp), "2026-05-17-010203", "debug")
            write_run_manifest(
                run_dir,
                run_name="",
                command="python -m src.train\n--epochs 1",
                notes="line one\nline two",
                source_manifest="",
            )
            text = (run_dir / "run_manifest.yaml").read_text(encoding="utf-8")
            self.assertIn('run_name: ""', text)
            self.assertIn("command: |", text)
            self.assertIn("  python -m src.train", text)
            self.assertIn("  --epochs 1", text)
            self.assertIn("notes: |", text)
            self.assertIn("  line one", text)
            self.assertIn("  line two", text)
            self.assertIn('source_manifest: ""', text)


if __name__ == "__main__":
    unittest.main()
