import tempfile
import unittest
from pathlib import Path

from scripts.record_experiment import append_note, note_filename


class RecordExperimentTests(unittest.TestCase):
    def test_note_filename(self):
        self.assertRegex(
            note_filename("patch", "knshnb B7"),
            r"^patch-knshnb-b7-[0-9a-f]{64}\.md$",
        )

    def test_note_filename_distinguishes_slug_collisions(self):
        self.assertNotEqual(
            note_filename("patch", "A/B"),
            note_filename("patch", "A B"),
        )

    def test_note_filename_uses_non_empty_basename_for_junk_title(self):
        filename = note_filename("patch", "????")
        self.assertNotEqual(filename, "patch-.md")
        self.assertRegex(filename, r"^patch-[a-z0-9-]+-[0-9a-f]{64}\.md$")

    def test_append_note_creates_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = append_note(Path(tmp), kind="patch", title="knshnb B7", body="Added accumulation.")
            text = path.read_text(encoding="utf-8")
            self.assertIn("# patch: knshnb B7", text)
            self.assertIn("Added accumulation.", text)

    def test_append_note_same_title_appends_to_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            first_path = append_note(base_dir, kind="patch", title="A/B", body="First entry.")
            second_path = append_note(base_dir, kind="patch", title="A/B", body="Second entry.")

            self.assertEqual(first_path, second_path)

            text = first_path.read_text(encoding="utf-8")
            self.assertEqual(text.count("# patch: A/B"), 1)
            self.assertIn("First entry.", text)
            self.assertIn("Second entry.", text)
            self.assertIn("---", text)


if __name__ == "__main__":
    unittest.main()
