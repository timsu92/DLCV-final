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
