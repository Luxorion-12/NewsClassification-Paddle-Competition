"""Fit pairwise LinearSVC models for common dev confusions."""
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from tfidf_baseline import read_labeled, read_titles
from tfidf_retrieval import DATA, OUT as BASE, ROOT, combine, features, normalized_title, save_json

OUT = ROOT / "outputs" / "tfidf_pairwise"


def candidates(scores):
    return np.argsort(-scores, axis=1, kind="stable")[:, :2]


def pairwise_decisions(base, pairs, titles, top):
    decisions = np.full(len(titles), np.nan, dtype=np.float32)
    low = np.minimum(top[:, 0], top[:, 1])
    high = np.maximum(top[:, 0], top[:, 1])
    for (a, b), classifier in pairs.items():
        rows = np.flatnonzero((low == a) & (high == b))
        if not len(rows):
            continue
        matrix = base["vectorizer"].transform([titles[row] for row in rows])
        margin = classifier.decision_function(matrix)
        decisions[rows] = margin * np.where(top[rows, 0] == b, 1, -1)
    return decisions


def rerank(scores, top, pairwise, weight, max_gap):
    rows = np.arange(len(scores))
    gap = scores[rows, top[:, 0]] - scores[rows, top[:, 1]]
    swap = (gap <= max_gap) & (gap + weight * pairwise < 0)
    return np.where(swap, top[:, 1], top[:, 0])


def train(base, data_dir):
    titles, names = read_labeled(data_dir / "dev.txt")
    label_ids = {label: index for index, label in enumerate(base["labels"])}
    target = np.array([label_ids[name] for name in names])
    scores = combine(*features(base, titles), **base["settings"])
    top = candidates(scores)
    mistakes = (top[:, 0] != target) & (top[:, 1] == target)
    frequent = Counter(tuple(sorted(pair)) for pair in top[mistakes])
    train_titles, train_names = read_labeled(data_dir / "train.txt")
    train_y = np.array([label_ids[name] for name in train_names])
    pairwise = {}
    for (a, b), errors in frequent.most_common(12):
        print(f"{base['labels'][a]} vs {base['labels'][b]}: {errors} dev top-2 errors", flush=True)
        train_rows = np.flatnonzero((train_y == a) | (train_y == b))
        matrix = base["vectorizer"].transform([train_titles[row] for row in train_rows])
        model = LinearSVC(C=0.8, max_iter=2500, dual=True, random_state=2026)
        model.fit(matrix, train_y[train_rows])
        pairwise[a, b] = model

    pair_margin = pairwise_decisions(base, pairwise, titles, top)
    grid = []
    for weight, gap in [(0, 0)] + [
        (weight, gap) for weight in (0.25, 0.5, 1.0, 2.0) for gap in (0.2, 0.4, 0.8, 1.6)
    ]:
        settings = {"weight": weight, "max_gap": gap}
        pred = rerank(scores, top, pair_margin, **settings)
        grid.append({"settings": settings, "dev_accuracy": float(accuracy_score(target, pred)),
                     "dev_macro_f1": float(f1_score(target, pred, average="macro"))})
    best = max(grid, key=lambda item: item["dev_accuracy"])
    temporary = OUT / "pairwise.tmp"
    joblib.dump({"pairs": pairwise, "settings": best["settings"]}, temporary)
    temporary.replace(OUT / "pairwise.joblib")
    save_json(OUT / "metrics.json", {"best": best, "base": grid[0], "validation_grid": grid})
    print(best, flush=True)
    return pairwise, best["settings"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = joblib.load(BASE / "model.joblib")
    if args.predict_only:
        artifact = joblib.load(OUT / "pairwise.joblib")
        pairwise, settings = artifact["pairs"], artifact["settings"]
    else:
        pairwise, settings = train(base, args.data_dir)
    titles = read_titles(args.data_dir / "test.txt")
    scores = combine(*features(base, titles), **base["settings"])
    top = candidates(scores)
    predicted = rerank(scores, top, pairwise_decisions(base, pairwise, titles, top), **settings)
    (OUT / "predictions.txt").write_text(
        "\n".join(base["labels"][index] for index in predicted) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(predicted)} test predictions", flush=True)


if __name__ == "__main__":
    main()

