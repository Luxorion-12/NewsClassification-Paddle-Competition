"""Optionally rerank MacBERT with training-title retrieval selected on dev."""
import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from tfidf_baseline import read_labeled, read_titles
from tfidf_retrieval import DATA, ROOT, features, normalized_title, save_json

SVM = ROOT / "outputs" / "tfidf"
MODEL = ROOT / "outputs" / "macbert"
OUT = ROOT / "outputs" / "macbert_retrieval"


def combine(logits, near, bonus, threshold):
    return (logits + bonus * np.maximum(near - threshold, 0)).argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", choices=("macbert", "macbert_average"), default="macbert")
    args = parser.parse_args()
    inputs = ROOT / "outputs" / args.logits
    bundle = joblib.load(SVM / "model.joblib")
    metrics_path = MODEL / "metrics.json"
    labels = json.loads(metrics_path.read_text(encoding="utf-8"))["labels"]
    if bundle["labels"] != labels:
        raise ValueError("TF-IDF and Transformer class mappings differ")
    ids = {label: index for index, label in enumerate(labels)}
    titles, names = read_labeled(DATA / "dev.txt")
    target = np.array([ids[name] for name in names])
    train_neighbors = np.load(SVM / "dev_retrieval.npy")
    dev_logits = np.load(inputs / "dev_logits.npy")
    if not train_neighbors.shape == dev_logits.shape == (len(titles), len(labels)):
        raise ValueError("Dev artifact shape mismatch")

    candidates = []
    for bonus, threshold in [(0, 0)] + [
        (bonus, threshold) for bonus in (2, 4, 8, 16) for threshold in (0.35, 0.5, 0.65, 0.8)
    ]:
        predicted = combine(dev_logits, train_neighbors, bonus, threshold)
        candidates.append({
            "bonus": bonus, "threshold": threshold,
            "dev_accuracy": float(accuracy_score(target, predicted)),
            "dev_macro_f1": float(f1_score(target, predicted, average="macro")),
        })
    best = max(candidates, key=lambda row: row["dev_accuracy"])
    print(best, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "metrics.json", {"logits": args.logits, "best": best, "validation_grid": candidates})

    test_titles = read_titles(DATA / "test.txt")
    test_logits = np.load(inputs / "test_logits.npy")
    if test_logits.shape != (len(test_titles), len(labels)):
        raise ValueError("Test artifact shape mismatch")
    if best["bonus"] > 0:
        near = features(bundle, test_titles, retrieval_only=True)
        predicted = combine(test_logits, near, best["bonus"], best["threshold"])
    else:
        predicted = test_logits.argmax(axis=1)
    (OUT / "predictions.txt").write_text(
        "\n".join(labels[index] for index in predicted) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(predicted)} predictions", flush=True)


if __name__ == "__main__":
    main()

