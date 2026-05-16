import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import LinkSpec, build_specs, prepare_links


class PrepareDataTests(unittest.TestCase):
    def test_build_specs_contains_required_knshnb_links(self):
        specs = build_specs(
            Path("data"),
            Path("3rd-party/kaggle-happywhale-1st-place/input"),
            "knshnb",
        )
        pairs = {(str(spec.source), str(spec.target)) for spec in specs}
        self.assertIn(
            (
                "data/train.csv",
                "3rd-party/kaggle-happywhale-1st-place/input/train.csv",
            ),
            pairs,
        )
        self.assertIn(
            (
                "data/sample_submission.csv",
                "3rd-party/kaggle-happywhale-1st-place/input/sample_submission.csv",
            ),
            pairs,
        )
        self.assertIn(
            (
                "data/train_images",
                "3rd-party/kaggle-happywhale-1st-place/input/train_images",
            ),
            pairs,
        )
        self.assertIn(
            (
                "data/test_images",
                "3rd-party/kaggle-happywhale-1st-place/input/test_images",
            ),
            pairs,
        )

    def test_build_specs_contains_required_charmq_links(self):
        specs = build_specs(
            Path("data"),
            Path("3rd-party/happywhale-2022/happywhale_data"),
            "charmq",
        )
        pairs = {(str(spec.source), str(spec.target)) for spec in specs}
        self.assertEqual(len(specs), 4)
        self.assertIn(
            (
                "data/train.csv",
                "3rd-party/happywhale-2022/happywhale_data/train.csv",
            ),
            pairs,
        )
        self.assertIn(
            (
                "data/sample_submission.csv",
                "3rd-party/happywhale-2022/happywhale_data/sample_submission.csv",
            ),
            pairs,
        )
        self.assertIn(
            (
                "data/train_images",
                "3rd-party/happywhale-2022/happywhale_data/train_images",
            ),
            pairs,
        )
        self.assertIn(
            (
                "data/test_images",
                "3rd-party/happywhale-2022/happywhale_data/test_images",
            ),
            pairs,
        )

    def test_build_specs_rejects_unknown_repo(self):
        with self.assertRaisesRegex(ValueError, "repo must be knshnb or charmq"):
            build_specs(Path("data"), Path("target"), "unknown")

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

    def test_prepare_links_dry_run_raises_for_conflicting_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "train.csv"
            dst = root / "repo" / "input" / "train.csv"
            src.parent.mkdir(parents=True)
            dst.parent.mkdir(parents=True)
            src.write_text("image,species,individual_id\n", encoding="utf-8")
            dst.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_links([LinkSpec(src, dst)], dry_run=True)

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

    def test_prepare_links_raises_for_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "missing.csv"
            dst = root / "repo" / "input" / "missing.csv"

            with self.assertRaises(FileNotFoundError):
                prepare_links([LinkSpec(src, dst)], dry_run=False)

    def test_prepare_links_keeps_existing_matching_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "train.csv"
            dst = root / "repo" / "input" / "train.csv"
            src.parent.mkdir(parents=True)
            dst.parent.mkdir(parents=True)
            src.write_text("image,species,individual_id\n", encoding="utf-8")
            dst.symlink_to(src.resolve())

            actions = prepare_links([LinkSpec(src, dst)], dry_run=False)

            self.assertEqual(actions, [f"link {dst} -> {src}"])
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.resolve(), src.resolve())

    def test_prepare_links_raises_for_existing_non_symlink_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "train.csv"
            dst = root / "repo" / "input" / "train.csv"
            src.parent.mkdir(parents=True)
            dst.parent.mkdir(parents=True)
            src.write_text("image,species,individual_id\n", encoding="utf-8")
            dst.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_links([LinkSpec(src, dst)], dry_run=False)

    def test_prepare_links_does_not_create_any_links_if_later_entry_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_src = root / "data" / "train.csv"
            second_src = root / "data" / "test.csv"
            first_dst = root / "repo" / "input" / "train.csv"
            second_dst = root / "repo" / "input" / "test.csv"
            first_src.parent.mkdir(parents=True)
            second_dst.parent.mkdir(parents=True)
            first_src.write_text("image,species,individual_id\n", encoding="utf-8")
            second_src.write_text("image\n", encoding="utf-8")
            second_dst.write_text("existing\n", encoding="utf-8")

            specs = [
                LinkSpec(first_src, first_dst),
                LinkSpec(second_src, second_dst),
            ]

            with self.assertRaises(FileExistsError):
                prepare_links(specs, dry_run=False)

            self.assertFalse(first_dst.exists())
            self.assertTrue(second_dst.is_file())


if __name__ == "__main__":
    unittest.main()
