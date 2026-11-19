"""Score predictions on the held-out clean_split test set.

This script reads test_labeled.txt only after a model has already produced
predictions for test.txt. It never trains or selects a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_DATA_DIR = ROOT / "clean_split"


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as file:
        lines = [raw.rstrip("\r\n") for raw in file]
    if not lines or any(not line for line in lines):
        raise ValueError(f"{path}: empty file or blank row")
    return lines


def report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    test_path = args.data_dir / "test.txt"
    labeled_path = args.data_dir / "test_labeled.txt"
    test_titles = read_lines(test_path)
    labeled_rows = read_lines(labeled_path)
    predictions = read_lines(args.predictions)
    if len(test_titles) != len(labeled_rows) or len(test_titles) != len(predictions):
        raise ValueError(
            f"row counts differ: test={len(test_titles)}, "
            f"test_labeled={len(labeled_rows)}, predictions={len(predictions)}"
        )

    true_labels: list[str] = []
    for row_number, (title, row) in enumerate(zip(test_titles, labeled_rows), start=1):
        if "\t" not in row:
            raise ValueError(f"{labeled_path}:{row_number}: expected title<TAB>label")
        labeled_title, label = row.rsplit("\t", 1)
        if title != labeled_title or not label:
            raise ValueError(f"{labeled_path}:{row_number}: title mismatch or empty label")
        true_labels.append(label)

    labels = sorted(set(true_labels))
    invalid = set(predictions) - set(labels)
    if invalid:
        raise ValueError(f"unknown predicted labels: {sorted(invalid)}")

    matrix = confusion_matrix(true_labels, predictions, labels=labels)
    largest_confusions = sorted(
        (
            {"true": true_label, "predicted": predicted_label, "count": int(matrix[i, j])}
            for i, true_label in enumerate(labels)
            for j, predicted_label in enumerate(labels)
            if i != j and matrix[i, j] > 0
        ),
        key=lambda item: item["count"],
        reverse=True,
    )[:10]
    report = {
        "path_base": "repository root (for relative paths)",
        "split": "dataset/clean_split/test",
        "test_file": report_path(test_path),
        "test_labeled_file": report_path(labeled_path),
        "predictions_file": report_path(args.predictions),
        "test_labeled_sha256": sha256(labeled_path),
        "predictions_sha256": sha256(args.predictions),
        "rows": len(true_labels),
        "labels": labels,
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(f1_score(true_labels, predictions, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            true_labels, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
        "largest_confusions": largest_confusions,
    }
    output = args.output or args.predictions.with_name("test_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rows={report['rows']} accuracy={report['accuracy']:.4f} macro-F1={report['macro_f1']:.4f}")
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
