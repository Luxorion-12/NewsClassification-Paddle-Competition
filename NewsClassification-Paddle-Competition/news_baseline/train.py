"""A small, reproducible PyTorch baseline for Chinese news title classification."""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "SkrnjMg2"


def read_labeled(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            if "\t" not in line:
                raise ValueError(f"{path}:{line_no}: expected title<TAB>label")
            title, label = line.rsplit("\t", 1)
            if not title or not label:
                raise ValueError(f"{path}:{line_no}: empty title or label")
            texts.append(title)
            labels.append(label)
    if not texts:
        raise ValueError(f"{path}: empty dataset")
    return texts, labels


def read_test(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [line.rstrip("\r\n") for line in handle]


def ngrams(title: str, max_chars: int) -> list[str]:
    chars = list(title[:max_chars])
    return ["1:" + ch for ch in chars] + [
        "2:" + chars[i] + chars[i + 1] for i in range(len(chars) - 1)
    ]


def build_vocab(texts: list[str], max_chars: int, max_features: int, min_freq: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for title in texts:
        counts.update(ngrams(title, max_chars))
    features = [token for token, count in counts.most_common() if count >= min_freq]
    return {token: index + 1 for index, token in enumerate(features[:max_features])}


def make_batch(
    texts: list[str], labels: list[int] | None, indices: np.ndarray | list[int],
    vocab: dict[str, int], max_chars: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    token_ids: list[int] = []
    offsets: list[int] = []
    for index in indices:
        offsets.append(len(token_ids))
        found = [vocab.get(token, 0) for token in ngrams(texts[int(index)], max_chars)]
        token_ids.extend(found or [0])
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    starts = torch.tensor(offsets, dtype=torch.long, device=device)
    targets = None if labels is None else torch.tensor(
        [labels[int(index)] for index in indices], dtype=torch.long, device=device
    )
    return ids, starts, targets


class NewsClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size + 1, embedding_dim, mode="mean", sparse=True)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(ids, offsets))


def predict(
    model: NewsClassifier, texts: list[str], vocab: dict[str, int],
    max_chars: int, batch_size: int, device: torch.device,
) -> list[int]:
    model.eval()
    result: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            indices = list(range(start, min(start + batch_size, len(texts))))
            ids, offsets, _ = make_batch(texts, None, indices, vocab, max_chars, device)
            result.extend(model(ids, offsets).argmax(dim=1).cpu().tolist())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--train-limit", type=int, default=None, help="Random training subset for a quick run")
    parser.add_argument("--dev-limit", type=int, default=None, help="Random validation subset for a quick run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=64)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_chars < 1:
        raise ValueError("epochs, batch-size and max-chars must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_texts, train_labels = read_labeled(args.data_dir / "train.txt")
    dev_texts, dev_labels = read_labeled(args.data_dir / "dev.txt")
    test_texts = read_test(args.data_dir / "test.txt")
    original_counts = (len(train_texts), len(dev_texts), len(test_texts))
    rng = random.Random(args.seed)
    if args.train_limit is not None and args.train_limit < len(train_texts):
        selected = rng.sample(range(len(train_texts)), args.train_limit)
        train_texts = [train_texts[i] for i in selected]
        train_labels = [train_labels[i] for i in selected]
    if args.dev_limit is not None and args.dev_limit < len(dev_texts):
        selected = rng.sample(range(len(dev_texts)), args.dev_limit)
        dev_texts = [dev_texts[i] for i in selected]
        dev_labels = [dev_labels[i] for i in selected]

    classes = sorted(set(train_labels))
    missing = set(dev_labels) - set(classes)
    if missing:
        raise ValueError(f"Validation contains unseen labels: {sorted(missing)}")
    label_to_id = {name: index for index, name in enumerate(classes)}
    train_y = [label_to_id[label] for label in train_labels]
    dev_y = [label_to_id[label] for label in dev_labels]
    train_title_set = set(train_texts)
    unseen_dev_indices = [i for i, title in enumerate(dev_texts) if title not in train_title_set]
    if not unseen_dev_indices:
        raise ValueError("All validation titles occur in training data; cannot measure unseen-title accuracy")
    unseen_dev_y = [dev_y[i] for i in unseen_dev_indices]
    print(f"Device: {device}; rows train/dev/test: {len(train_texts)}/{len(dev_texts)}/{len(test_texts)}", flush=True)
    print(f"Classes ({len(classes)}): {', '.join(classes)}", flush=True)
    print(f"Validation titles unseen in training: {len(unseen_dev_indices)}/{len(dev_texts)}", flush=True)

    vocab = build_vocab(train_texts, args.max_chars, args.max_features, args.min_freq)
    print(f"Vocabulary: {len(vocab)} character n-grams", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = NewsClassifier(len(vocab), len(classes), args.embedding_dim).to(device)
    sparse_optimizer = torch.optim.SparseAdam(model.embedding.parameters(), lr=args.learning_rate)
    dense_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    history = []
    checkpoint = args.output_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_texts))
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            ids, offsets, targets = make_batch(train_texts, train_y, indices, vocab, args.max_chars, device)
            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(ids, offsets), targets)
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()
            loss_sum += loss.item() * len(indices)
        dev_pred = predict(model, dev_texts, vocab, args.max_chars, args.batch_size, device)
        accuracy = accuracy_score(dev_y, dev_pred)
        macro_f1 = f1_score(dev_y, dev_pred, average="macro", zero_division=0)
        unseen_pred = [dev_pred[i] for i in unseen_dev_indices]
        unseen_accuracy = accuracy_score(unseen_dev_y, unseen_pred)
        unseen_macro_f1 = f1_score(unseen_dev_y, unseen_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_texts),
                        "dev_accuracy": accuracy, "dev_macro_f1": macro_f1,
                        "unseen_dev_accuracy": unseen_accuracy,
                        "unseen_dev_macro_f1": unseen_macro_f1})
        print(f"Epoch {epoch}: loss={history[-1]['train_loss']:.4f}, "
              f"dev_accuracy={accuracy:.4f}, unseen_dev_accuracy={unseen_accuracy:.4f}, "
              f"unseen_dev_macro_f1={unseen_macro_f1:.4f}", flush=True)
        if unseen_accuracy > best_accuracy:
            best_accuracy = unseen_accuracy
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    dev_pred = predict(model, dev_texts, vocab, args.max_chars, args.batch_size, device)
    test_pred = predict(model, test_texts, vocab, args.max_chars, args.batch_size, device)
    predictions = [classes[index] for index in test_pred]
    result_file = args.output_dir / "result.txt"
    result_file.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    zip_file = args.output_dir / "submission.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_file, arcname="result.txt")
    (args.output_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "architecture": "PyTorch EmbeddingBag, character unigrams and bigrams, mean pooling, linear classifier",
        "seed": args.seed,
        "device": str(device),
        "source_rows": dict(zip(("train", "dev", "test"), original_counts)),
        "used_rows": {"train": len(train_texts), "dev": len(dev_texts), "test": len(test_texts)},
        "dev_title_overlap_with_train": len(dev_texts) - len(unseen_dev_indices),
        "unseen_dev_rows": len(unseen_dev_indices),
        "classes": classes,
        "vocab_size": len(vocab),
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                            "embedding_dim": args.embedding_dim, "max_chars": args.max_chars,
                            "max_features": args.max_features, "min_freq": args.min_freq,
                            "learning_rate": args.learning_rate},
        "history": history,
        "model_selection_metric": "unseen_dev_accuracy",
        "best_epoch": max(history, key=lambda row: row["unseen_dev_accuracy"])["epoch"],
        "dev_classification_report": classification_report(
            dev_y, dev_pred, labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "unseen_dev_classification_report": classification_report(
            unseen_dev_y, [dev_pred[i] for i in unseen_dev_indices],
            labels=list(range(len(classes))), target_names=classes,
            output_dict=True, zero_division=0,
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(predictions)} predictions to {result_file}", flush=True)
    print(f"Saved archive to {zip_file}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
