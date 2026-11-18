"""Fine-tune a small pretrained Chinese Transformer on clean_split.

The default model is a local copy of uer/chinese_roberta_L-2_H-128. Validation
labels are used only for metrics and model selection; test_labeled.txt is unused.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "clean_split"
DEFAULT_MODEL_DIR = ROOT / "pretrained" / "chinese_roberta_tiny"


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


class TitleDataset(Dataset):
    def __init__(self, titles: list[str], labels: list[int] | None = None):
        self.titles = titles
        self.labels = labels

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, index: int) -> tuple[str, int | None]:
        label = None if self.labels is None else self.labels[index]
        return self.titles[index], label


def sample_rows(
    titles: list[str], labels: list[str], limit: int | None, seed: int
) -> tuple[list[str], list[str]]:
    if limit is None or limit >= len(titles):
        return titles, labels
    indices = sorted(random.Random(seed).sample(range(len(titles)), limit))
    return [titles[i] for i in indices], [labels[i] for i in indices]


def make_collate(tokenizer, max_length: int, with_labels: bool):
    def collate(rows: list[tuple[str, int | None]]) -> dict[str, torch.Tensor]:
        titles = [title for title, _ in rows]
        batch = tokenizer(
            titles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if with_labels:
            batch["labels"] = torch.tensor([label for _, label in rows], dtype=torch.long)
        return batch

    return collate


def predict(model, loader: DataLoader, device: torch.device) -> list[int]:
    model.eval()
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tiny")
    parser.add_argument("--train-limit", type=int, help="random training subset for a pilot run")
    parser.add_argument("--dev-limit", type=int, help="random validation subset for a pilot run")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--predict-only", action="store_true", help="use a saved model to predict test.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value < 1 for value in (args.epochs, args.batch_size, args.max_length, args.threads)):
        raise ValueError("epochs, batch-size, max-length, and threads must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.train_limit is not None and args.train_limit < 1:
        raise ValueError("train-limit must be positive")
    if args.dev_limit is not None and args.dev_limit < 1:
        raise ValueError("dev-limit must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.predict_only:
        saved_model_dir = args.output_dir / "model"
        saved_metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
        saved_max_length = int(saved_metrics["settings"]["max_length"])
        tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            saved_model_dir, local_files_only=True
        ).to(device)
        test_titles = read_titles(args.data_dir / "test.txt")
        test_loader = DataLoader(
            TitleDataset(test_titles), batch_size=args.batch_size, shuffle=False,
            collate_fn=make_collate(tokenizer, saved_max_length, with_labels=False),
        )
        test_predictions = predict(model, test_loader, device)
        (args.output_dir / "predictions.txt").write_text(
            "\n".join(model.config.id2label[index] for index in test_predictions) + "\n",
            encoding="utf-8",
        )
        print(f"test predictions={len(test_predictions)}; saved: {args.output_dir.resolve()}", flush=True)
        return

    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"pretrained model directory missing: {args.model_dir}")

    train_titles, train_labels = read_labeled(args.data_dir / "train.txt")
    dev_titles, dev_labels = read_labeled(args.data_dir / "dev.txt")
    source_rows = {"train": len(train_titles), "dev": len(dev_titles)}
    train_titles, train_labels = sample_rows(train_titles, train_labels, args.train_limit, args.seed)
    dev_titles, dev_labels = sample_rows(dev_titles, dev_labels, args.dev_limit, args.seed + 1)

    labels = sorted(set(train_labels))
    label_to_id = {label: index for index, label in enumerate(labels)}
    unknown_dev_labels = set(dev_labels) - set(labels)
    if unknown_dev_labels:
        raise ValueError(f"dev has unseen labels: {sorted(unknown_dev_labels)}")
    train_y = [label_to_id[label] for label in train_labels]
    dev_y = [label_to_id[label] for label in dev_labels]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id=label_to_id,
        local_files_only=True,
    ).to(device)
    train_loader = DataLoader(
        TitleDataset(train_titles, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_length, with_labels=True),
    )
    dev_loader = DataLoader(
        TitleDataset(dev_titles),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, args.max_length, with_labels=False),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(
        f"device={device}; train={len(train_titles)}; dev={len(dev_titles)}; "
        f"classes={len(labels)}; max_length={args.max_length}",
        flush=True,
    )
    output_model_dir = args.output_dir / "model"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            batch_size = batch["labels"].shape[0]
            loss_sum += float(output.loss.item()) * batch_size
            seen += batch_size
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch}/{args.epochs}, step {step}/{len(train_loader)}, "
                    f"train_loss={loss_sum / seen:.4f}",
                    flush=True,
                )

        dev_predictions = predict(model, dev_loader, device)
        accuracy = float(accuracy_score(dev_y, dev_predictions))
        macro_f1 = float(f1_score(dev_y, dev_predictions, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "dev_accuracy": accuracy, "dev_macro_f1": macro_f1})
        print(f"epoch {epoch}: dev accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    final_metrics = history[-1]
    metrics = {
        "path_base": "repository root (for relative paths)",
        "data_dir": report_path(args.data_dir),
        "pretrained_model": report_path(args.model_dir),
        "device": str(device),
        "source_rows": source_rows,
        "used_rows": {"train": len(train_titles), "dev": len(dev_titles)},
        "labels": labels,
        "settings": {"epochs": args.epochs, "batch_size": args.batch_size,
                     "max_length": args.max_length, "learning_rate": args.learning_rate,
                     "seed": args.seed},
        "history": history,
        "saved_epoch": args.epochs,
        "dev_accuracy": final_metrics["dev_accuracy"],
        "dev_macro_f1": final_metrics["dev_macro_f1"],
        "dev_classification_report": classification_report(
            dev_y, dev_predictions, labels=list(range(len(labels))),
            target_names=labels, output_dict=True, zero_division=0,
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.predict_test:
        test_titles = read_titles(args.data_dir / "test.txt")
        test_loader = DataLoader(
            TitleDataset(test_titles), batch_size=args.batch_size, shuffle=False,
            collate_fn=make_collate(tokenizer, args.max_length, with_labels=False),
        )
        test_predictions = predict(model, test_loader, device)
        (args.output_dir / "predictions.txt").write_text(
            "\n".join(labels[index] for index in test_predictions) + "\n",
            encoding="utf-8",
        )
        print(f"test predictions={len(test_predictions)}", flush=True)
    print(f"saved: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
