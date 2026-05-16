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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_submission(
    submission_path: Path, sample_path: Path, train_path: Path
) -> ValidationResult:
    submission = read_rows(submission_path)
    sample = read_rows(sample_path)
    train = read_rows(train_path)
    if not submission:
        raise ValueError("submission has no rows")
    if set(submission[0].keys()) != {"image", "predictions"}:
        raise ValueError(
            f"submission columns must be image,predictions, got {list(submission[0].keys())}"
        )
    if len(submission) != len(sample):
        raise ValueError(
            f"row count mismatch: submission={len(submission)} sample={len(sample)}"
        )
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
