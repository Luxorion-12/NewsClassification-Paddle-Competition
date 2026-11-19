"""Fine-tune MacBERT on the fixed train/dev split, then predict unlabeled titles."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup

from tfidf_baseline import read_labeled, read_titles

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "database"
OUT = ROOT / "outputs" / "macbert"
PRETRAINED = ROOT / "pretrained" / "macbert-base"


def save_json(path, content):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def encode(tokenizer, titles, max_length):
    result = []
    for offset in range(0, len(titles), 8192):
        encoded = tokenizer(
            titles[offset:offset + 8192], max_length=max_length,
            truncation=True, padding="max_length", return_tensors="np",
            return_token_type_ids=False,
        )
        result.append(encoded["input_ids"].astype(np.int32))
    tokens = np.concatenate(result)
    lengths = (tokens != tokenizer.pad_token_id).sum(axis=1)
    return tokens, lengths


def batches(lengths, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    boundaries = ((lengths + 7) // 8) * 8
    result = []
    for boundary in np.unique(boundaries):
        rows = np.flatnonzero(boundaries == boundary)
        if seed is not None:
            rng.shuffle(rows)
        result.extend((rows[i:i + batch_size], boundary) for i in range(0, len(rows), batch_size))
    if seed is not None:
        rng.shuffle(result)
    return result


def model_input(tokens, rows, length, pad_id, device):
    ids = torch.from_numpy(tokens[rows, :length]).to(device=device, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": ids.ne(pad_id)}


def predict(model, encoded, batch_size, pad_id, device):
    tokens, lengths = encoded
    result = np.empty((len(tokens), model.config.num_labels), dtype=np.float32)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for rows, length in batches(lengths, batch_size):
            logits = model(**model_input(tokens, rows, length, pad_id, device)).logits
            result[rows] = logits.float().cpu().numpy()
    return result


def fine_tune(args, device):
    torch.manual_seed(args.seed)
    titles, names = read_labeled(args.data_dir / "train.txt")
    dev_titles, dev_names = read_labeled(args.data_dir / "dev.txt")
    labels = sorted(set(names))
    ids = {name: index for index, name in enumerate(labels)}
    y = torch.tensor([ids[name] for name in names], device=device)
    dev_y = np.array([ids[name] for name in dev_names])
    config = {"epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "data_dir": str(args.data_dir.resolve()), "model_dir": str(args.model_dir.resolve())}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=len(labels), local_files_only=True,
        id2label=dict(enumerate(labels)), label2id=ids,
    ).to(device)
    model.config.max_title_length = args.max_length
    print(f"Tokenizing {len(titles)} train and {len(dev_titles)} dev rows; device={device}", flush=True)
    tokens, lengths = encode(tokenizer, titles, args.max_length)
    dev = encode(tokenizer, dev_titles, args.max_length)
    del titles, names, dev_titles
    steps_per_epoch = len(batches(lengths, args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=(device.type == "cuda"))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.065), total_steps)
    history = []
    first_epoch = 1
    best = -1.0
    pad = tokenizer.pad_token_id
    if args.resume:
        checkpoint = torch.load(OUT / "last_checkpoint.pt", weights_only=False, map_location="cpu")
        if checkpoint["config"] != config or checkpoint["labels"] != labels:
            raise ValueError("Resume requires the original configuration and label mapping")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        history = checkpoint["history"]
        best = checkpoint["best"]
        first_epoch = checkpoint["epoch"] + 1
        del checkpoint

    def validate(epoch, step):
        nonlocal best
        logits = predict(model, dev, args.batch_size, pad, device)
        predicted = logits.argmax(axis=1)
        accuracy = float(accuracy_score(dev_y, predicted))
        f1 = float(f1_score(dev_y, predicted, average="macro"))
        row = {"epoch": epoch, "step": step, "dev_accuracy": accuracy, "dev_macro_f1": f1}
        history.append(row)
        print(row, flush=True)
        if accuracy > best:
            best = accuracy
            model.save_pretrained(OUT / "model")
            tokenizer.save_pretrained(OUT / "model")
            np.save(OUT / "dev_logits.npy", logits)
        save_json(OUT / "metrics.json", {
            "best": max(history, key=lambda item: item["dev_accuracy"]),
            "history": history, "settings": config, "device": str(device),
            "train_rows": len(tokens), "dev_rows": len(dev_y), "labels": labels,
        })
        model.train()

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        start = time.monotonic()
        for step, (rows, length) in enumerate(batches(lengths, args.batch_size, args.seed + epoch), 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch = model_input(tokens, rows, length, pad, device)
                targets = y[torch.from_numpy(rows).to(device)]
                loss = model(**batch, labels=targets).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % 200 == 0:
                print(f"epoch={epoch}/{args.epochs} step={step}/{steps_per_epoch} "
                      f"loss={loss.item():.4f} elapsed={time.monotonic() - start:.0f}s", flush=True)
            if step == steps_per_epoch or (epoch == 1 and step == steps_per_epoch // 2):
                validate(epoch, step)
        temporary = OUT / "last_checkpoint.tmp"
        torch.save({
            "epoch": epoch, "config": config, "labels": labels, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
            "history": history, "best": best,
        }, temporary)
        temporary.replace(OUT / "last_checkpoint.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--model-dir", type=Path, default=PRETRAINED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.predict_only:
        fine_tune(args, device)
    tokenizer = AutoTokenizer.from_pretrained(OUT / "model", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "model", local_files_only=True).to(device)
    titles = read_titles(args.data_dir / "test.txt")
    encoded = encode(tokenizer, titles, model.config.max_title_length)
    logits = predict(model, encoded, args.batch_size, tokenizer.pad_token_id, device)
    np.save(OUT / "test_logits.npy", logits)
    (OUT / "predictions.txt").write_text(
        "\n".join(model.config.id2label[index] for index in logits.argmax(axis=1)) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(titles)} test predictions", flush=True)


if __name__ == "__main__":
    main()

