"""Train and validate a character TF-IDF classifier with train-only retrieval."""
import argparse
import gc
import json
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from tfidf_baseline import read_labeled, read_titles

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "database"
OUT = ROOT / "outputs" / "tfidf"
NUMBER = re.compile(r"[0-9]+")
SPACE = re.compile(r"\s+")


def normalized_title(title):
    title = unicodedata.normalize("NFKC", title).casefold()
    return SPACE.sub("", NUMBER.sub("0", title))


def save_json(path, content):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def retrieval_columns(vectorizer, matrix):
    counts = np.bincount(matrix.indices, minlength=matrix.shape[1])
    columns = np.flatnonzero((counts >= 3) & (counts <= 1800))
    terms = vectorizer.get_feature_names_out()
    return columns[np.fromiter((len(terms[col]) >= 2 for col in columns), dtype=bool)]


def features(bundle, titles, retrieval_only=False):
    vectorizer = bundle["vectorizer"]
    index = bundle["index"]
    train_labels = bundle["train_labels"]
    original = bundle.get("original")
    classes = len(bundle["labels"])
    margins = []
    raw_margins = []
    retrieval_scores = []
    for offset in range(0, len(titles), 1024):
        batch = titles[offset:offset + 1024]
        matrix = vectorizer.transform(batch)
        if not retrieval_only:
            margins.append(bundle["classifier"].decision_function(matrix))
            if original is not None:
                raw = original["vectorizer"].transform(batch)
                raw_margins.append(original["classifier"].decision_function(raw))
        query = normalize(matrix[:, bundle["columns"]].tocsr(), copy=False)
        scores = np.zeros((len(batch), classes), dtype=np.float32)
        for start in range(0, len(batch), 64):
            matching = (query[start:start + 64] @ index).tocsr()
            for row in range(matching.shape[0]):
                lo, hi = matching.indptr[row:row + 2]
                dots = matching.data[lo:hi]
                target_rows = matching.indices[lo:hi]
                np.maximum.at(scores[start + row], train_labels[target_rows], dots)
        retrieval_scores.append(scores)
    if retrieval_only:
        return np.concatenate(retrieval_scores)
    return (
        np.concatenate(margins),
        np.concatenate(raw_margins) if original is not None else np.concatenate(margins),
        np.concatenate(retrieval_scores),
    )


def combine(margins, raw, retrieval, weight, bonus, threshold):
    return weight * margins + (1 - weight) * raw + bonus * np.maximum(retrieval - threshold, 0)


def predict(bundle, titles):
    return combine(*features(bundle, titles), **bundle["settings"]).argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--baseline", type=Path, default=ROOT / "baseline_svm.joblib")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.predict_only:
        titles, names = read_labeled(args.data_dir / "train.txt")
        dev, dev_names = read_labeled(args.data_dir / "dev.txt")
        labels = sorted(set(names))
        ids = {label: index for index, label in enumerate(labels)}
        train_y = np.array([ids[name] for name in names], dtype=np.int8)
        dev_y = np.array([ids[name] for name in dev_names], dtype=np.int8)
        vectorizer = TfidfVectorizer(
            preprocessor=normalized_title, analyzer="char", ngram_range=(1, 4),
            min_df=3, sublinear_tf=True, dtype=np.float32,
        )
        print(f"Vectorizing {len(titles)} train titles", flush=True)
        matrix = vectorizer.fit_transform(titles)
        print(f"Fitting LinearSVC on {matrix.shape}, nnz={matrix.nnz}", flush=True)
        classifier = LinearSVC(C=0.8, dual=True, max_iter=2500, random_state=2026)
        classifier.fit(matrix, train_y)
        columns = retrieval_columns(vectorizer, matrix)
        index = normalize(matrix[:, columns].tocsr(), copy=False).T.tocsr()
        del titles, names, matrix
        gc.collect()
        original = joblib.load(args.baseline)
        if list(original["labels"]) != labels:
            raise ValueError("baseline class mapping differs")
        bundle = {
            "vectorizer": vectorizer, "classifier": classifier, "labels": labels,
            "original": original, "index": index, "columns": columns,
            "train_labels": train_y,
        }
        print("Validating both SVM representations and retrieval", flush=True)
        margins, raw, retrieval = features(bundle, dev)
        candidates = []
        for weight in (0, 0.25, 0.5, 0.75, 1):
            for bonus, threshold in [(0, 0)] + [
                (b, t) for b in (0.35, 0.7, 1.4, 2.8) for t in (0.35, 0.5, 0.65, 0.8)
            ]:
                settings = {"weight": weight, "bonus": bonus, "threshold": threshold}
                predicted = combine(margins, raw, retrieval, **settings).argmax(axis=1)
                candidates.append({**settings, "accuracy": float(accuracy_score(dev_y, predicted))})
        best = max(candidates, key=lambda candidate: candidate["accuracy"])
        bundle["settings"] = {key: best[key] for key in ("weight", "bonus", "threshold")}
        predicted = combine(margins, raw, retrieval, **bundle["settings"]).argmax(axis=1)
        result = {
            "train_rows": len(train_y), "dev_rows": len(dev_y),
            "dev_accuracy": best["accuracy"],
            "dev_macro_f1": float(f1_score(dev_y, predicted, average="macro")),
            "dev_normalized_svm_accuracy": float(accuracy_score(dev_y, margins.argmax(axis=1))),
            "dev_original_svm_accuracy": float(accuracy_score(dev_y, raw.argmax(axis=1))),
            "dev_best_svm_only_accuracy": max(row["accuracy"] for row in candidates if row["bonus"] == 0),
            "settings": bundle["settings"], "validation_grid": candidates,
        }
        np.save(OUT / "dev_retrieval.npy", retrieval)
        print({k: v for k, v in result.items() if k != "validation_grid"}, flush=True)
        temporary = OUT / "model.tmp"
        joblib.dump(bundle, temporary)
        temporary.replace(OUT / "model.joblib")
        save_json(OUT / "metrics.json", result)
    else:
        bundle = joblib.load(OUT / "model.joblib")
    titles = read_titles(args.data_dir / "test.txt")
    labels = bundle["labels"]
    predictions = predict(bundle, titles)
    (OUT / "predictions.txt").write_text(
        "\n".join(labels[index] for index in predictions) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(predictions)} test predictions", flush=True)


if __name__ == "__main__":
    main()

