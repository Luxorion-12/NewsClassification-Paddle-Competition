"""Test prefix, suffix and title-shape features with a linear SVM."""
import argparse
import gc
import re
import unicodedata

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from tfidf_baseline import read_labeled, read_titles
from tfidf_retrieval import DATA, OUT as BASE, ROOT, combine, features, normalized_title, save_json

OUT = ROOT / "outputs" / "tfidf_position"
DIGITS = re.compile(r"\d+")
LATIN = re.compile(r"[A-Za-z]+")
CJK = re.compile(r"[\u3400-\u9fff]+")


def position_terms(title):
    raw = unicodedata.normalize("NFKC", title)
    title = normalized_title(raw)
    # Separate ordinary substrings from boundary substrings and title shape.
    for width in range(1, 5):
        yield from (title[i:i + width] for i in range(len(title) - width + 1))
        if len(title) >= width:
            yield "start:" + title[:width]
            yield "end:" + title[-width:]
        if len(title) > width + 1:
            yield "start:" + title[1:1 + width]
            yield "end:" + title[-1 - width:-1]
    for token in re.split(r"\s+", raw):
        if len(token) >= 2:
            yield "token:" + token.casefold()
    shape = CJK.sub("汉", LATIN.sub("a", DIGITS.sub("0", raw)))
    for width in range(1, 4):
        yield from ("shape:" + shape[i:i + width] for i in range(len(shape) - width + 1))
    yield f"size:{min(len(title) // 4, 20)}"
    yield f"spaces:{min(sum(ch.isspace() for ch in raw), 8)}"


def predict_margins(bundle, titles):
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    return np.concatenate([
        classifier.decision_function(vectorizer.transform(titles[start:start + 1024]))
        for start in range(0, len(titles), 1024)
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.predict_only:
        artifact = joblib.load(OUT / "position.joblib")
    else:
        titles, labels = read_labeled(DATA / "train.txt")
        labels_sorted = sorted(set(labels))
        ids = {label: index for index, label in enumerate(labels_sorted)}
        target = np.array([ids[label] for label in labels])
        vectorizer = TfidfVectorizer(analyzer=position_terms, min_df=4,
                                     dtype=np.float32, sublinear_tf=True)
        print("Fitting TF-IDF with character n-grams and title boundary features", flush=True)
        matrix = vectorizer.fit_transform(titles)
        print(f"Fitting on {matrix.shape}; nnz={matrix.nnz}", flush=True)
        classifier = LinearSVC(C=1.0, max_iter=2500, dual=True, random_state=2026)
        classifier.fit(matrix, target)
        del matrix, titles, labels
        gc.collect()
        artifact = {"vectorizer": vectorizer, "classifier": classifier}
    base = joblib.load(BASE / "model.joblib")
    if not args.predict_only:
        if base["labels"] != labels_sorted:
            raise ValueError("SVM label mappings differ")
        dev, labels = read_labeled(DATA / "dev.txt")
        target = np.array([ids[label] for label in labels])
        pos = predict_margins(artifact, dev)
        initial = combine(*features(base, dev), **base["settings"])
        grid = []
        for weight in (0, 0.25, 0.5, 0.75, 1):
            pred = (weight * pos + (1 - weight) * initial).argmax(axis=1)
            grid.append({"weight": weight, "dev_accuracy": float(accuracy_score(target, pred)),
                         "dev_macro_f1": float(f1_score(target, pred, average="macro"))})
        best = max(grid, key=lambda row: row["dev_accuracy"])
        artifact["weight"] = best["weight"]
        print(best, flush=True)
        if best["weight"] != 0:
            temporary = OUT / "position.tmp"
            joblib.dump(artifact, temporary)
            temporary.replace(OUT / "position.joblib")
        save_json(OUT / "metrics.json", {"best": best, "validation_grid": grid})
        if best["weight"] == 0:
            return
    test = read_titles(DATA / "test.txt")
    pos = predict_margins(artifact, test)
    initial = combine(*features(base, test), **base["settings"])
    weight = artifact["weight"]
    predicted = (weight * pos + (1 - weight) * initial).argmax(axis=1)
    (OUT / "predictions.txt").write_text(
        "\n".join(base["labels"][index] for index in predicted) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(predicted)} test predictions", flush=True)


if __name__ == "__main__":
    main()

