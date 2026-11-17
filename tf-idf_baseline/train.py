"""Train a character TF-IDF + LinearSVC baseline on clean_split.

The script selects no settings with test_labeled.txt. It reports dev metrics,
saves the full inference bundle, and writes predictions for clean_split/test.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "clean_split"


def report_path(path: Path) -> str:
    """Use a repository-relative path when the file is inside this project."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_labeled(path: Path) -> tuple[list[str], list[str]]:
    titles: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, raw in enumerate(file, start=1):
            line = raw.rstrip("\r\n")
            if "\t" not in line:
                raise ValueError(f"{path}:{line_number}: expected title<TAB>label")
            title, label = line.rsplit("\t", 1)
            if not title or not label:
                raise ValueError(f"{path}:{line_number}: empty title or label")
            titles.append(title)
            labels.append(label)
    if not titles:
        raise ValueError(f"{path}: no rows")
    return titles, labels


def read_titles(path: Path) -> list[str]:
    titles: list[str] = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, raw in enumerate(file, start=1):
            title = raw.rstrip("\r\n")
            if not title:
                raise ValueError(f"{path}:{line_number}: empty title")
            titles.append(title)
    if not titles:
        raise ValueError(f"{path}: no rows")
    return titles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--ngram-range", type=int, nargs=2, default=(1, 2),
                        metavar=("MIN", "MAX"))
    parser.add_argument("--dev-only", action="store_true", help="skip test predictions during tuning")
    parser.add_argument("--predict-only", action="store_true", help="use a saved model to predict test.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_df < 1 or args.c <= 0:
        raise ValueError("min-df must be at least 1 and c must be positive")
    ngram_range = tuple(args.ngram_range)
    if not 1 <= ngram_range[0] <= ngram_range[1]:
        raise ValueError("ngram-range must satisfy 1 <= MIN <= MAX")

    output_dir = args.output_dir or ROOT / "outputs" / f"min_df_{args.min_df}"
    if args.predict_only:
        bundle = joblib.load(output_dir / "model.joblib")
        test_titles = read_titles(args.data_dir / "test.txt")
        test_matrix = bundle["vectorizer"].transform(test_titles)
        predictions = bundle["classifier"].predict(test_matrix)
        labels = bundle["labels"]
        (output_dir / "predictions.txt").write_text(
            "\n".join(labels[int(index)] for index in predictions) + "\n",
            encoding="utf-8",
        )
        print(f"test predictions={len(predictions)}; saved: {output_dir.resolve()}")
        return

    train_titles, train_labels = read_labeled(args.data_dir / "train.txt")
    dev_titles, dev_labels = read_labeled(args.data_dir / "dev.txt")
    test_titles = None if args.dev_only else read_titles(args.data_dir / "test.txt")

    # One fixed mapping is shared by training, validation, and prediction.
    labels = sorted(set(train_labels))
    label_to_id = {label: index for index, label in enumerate(labels)}
    unknown_dev_labels = set(dev_labels) - set(labels)
    if unknown_dev_labels:
        raise ValueError(f"dev has unseen labels: {sorted(unknown_dev_labels)}")
    y_train = [label_to_id[label] for label in train_labels]
    y_dev = [label_to_id[label] for label in dev_labels]

    # Learn vocabulary and IDF on train only; reuse them for dev and test.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=args.min_df,
        lowercase=False,
    )
    x_train = vectorizer.fit_transform(train_titles)
    x_dev = vectorizer.transform(dev_titles)

    classifier = LinearSVC(C=args.c, random_state=2026)
    classifier.fit(x_train, y_train)
    dev_predictions = classifier.predict(x_dev)

    accuracy = float(accuracy_score(y_dev, dev_predictions))
    macro_f1 = float(f1_score(y_dev, dev_predictions, average="macro", zero_division=0))
    print(f"train: {x_train.shape}; dev: {x_dev.shape}; classes: {len(labels)}")
    print(f"dev accuracy: {accuracy:.4f}; dev macro-F1: {macro_f1:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier, "labels": labels},
        output_dir / "model.joblib",
    )
    if test_titles is not None:
        test_predictions = classifier.predict(vectorizer.transform(test_titles))
        predicted_labels = [labels[int(index)] for index in test_predictions]
        (output_dir / "predictions.txt").write_text(
            "\n".join(predicted_labels) + "\n", encoding="utf-8"
        )
    metrics = {
        "path_base": "repository root (for relative paths)",
        "data_dir": report_path(args.data_dir),
        "train_rows": len(train_titles),
        "dev_rows": len(dev_titles),
        "test_rows": len(test_titles) if test_titles is not None else 0,
        "vocabulary_size": x_train.shape[1],
        "labels": labels,
        "settings": {"analyzer": "char", "ngram_range": ngram_range,
                     "min_df": args.min_df, "lowercase": False, "c": args.c},
        "dev_accuracy": accuracy,
        "dev_macro_f1": macro_f1,
        "dev_classification_report": classification_report(
            y_dev,
            dev_predictions,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
