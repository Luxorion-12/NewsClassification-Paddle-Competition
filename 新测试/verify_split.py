"""Verify split membership and pin the unlabeled evaluation input."""
from pathlib import Path

from evaluate_test import sha256
from tfidf_baseline import read_labeled, read_titles
from tfidf_retrieval import DATA, ROOT, save_json


def main():
    train, train_labels = read_labeled(DATA / "train.txt")
    dev, dev_labels = read_labeled(DATA / "dev.txt")
    test = read_titles(DATA / "test.txt")
    texts = {"train": train, "dev": dev, "test": test}
    seen = {name: set(titles) for name, titles in texts.items()}
    rows = {name: len(titles) for name, titles in texts.items()}
    if rows != {"train": 646510, "dev": 68736, "test": 71826}:
        raise ValueError(f"Unexpected fixed split counts: {rows}")
    if any(rows[name] != len(titles) for name, titles in seen.items()):
        raise ValueError("Duplicate title within a split")
    overlap = {
        f"{a}_{b}": len(seen[a] & seen[b])
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))
    }
    if any(overlap.values()) or len(set(train_labels)) != 14 or set(train_labels) != set(dev_labels):
        raise ValueError(f"Unexpected split overlap or label mapping: {overlap}")
    result = {
        "rows": rows,
        "exact_title_overlap": overlap,
        "files_sha256": {name: sha256(DATA / f"{name}.txt") for name in texts},
        "labels": sorted(set(train_labels)),
    }
    save_json(ROOT / "split_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()

