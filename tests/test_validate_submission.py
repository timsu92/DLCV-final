import csv
import tempfile
import unittest
from pathlib import Path

from scripts.validate_submission import validate_submission


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
