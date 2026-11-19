"""Compare averaging epochs 2 and 3 with the best individual MacBERT checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tfidf_baseline import read_labeled, read_titles
from train_macbert import DATA, OUT as BASE, ROOT, encode, predict, save_json

OUT = ROOT / "outputs" / "macbert_average"
SNAPSHOT = OUT / "epoch2.safetensors"


def snapshot():
    checkpoint = torch.load(BASE / "last_checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    if checkpoint["epoch"] != 2:
        raise ValueError("Snapshot expects the epoch 2 checkpoint")
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = SNAPSHOT.with_suffix(".tmp")
    save_file({name: value.contiguous() for name, value in checkpoint["model"].items()}, temporary)
    temporary.replace(SNAPSHOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-epoch2", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    if args.snapshot_epoch2:
        snapshot()
        return
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_metrics = json.loads((BASE / "metrics.json").read_text(encoding="utf-8"))
    labels = base_metrics["labels"]
    base_model = OUT / "model" if args.predict_only else BASE / "model"
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, local_files_only=True)
    batch_size = base_metrics["settings"]["batch_size"]
    length = model.config.max_title_length
    if not args.predict_only:
        epoch2 = load_file(SNAPSHOT)
        last = torch.load(BASE / "last_checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
        if last["epoch"] != 3 or last["labels"] != labels:
            raise ValueError("The final three-epoch run is required")
        state = last["model"]
        titles, names = read_labeled(DATA / "dev.txt")
        ids = {name: index for index, name in enumerate(labels)}
        target = np.array([ids[name] for name in names])
        encoded = encode(tokenizer, titles, length)
        best = {"weight_epoch2": None, **base_metrics["best"]}
        grid = [best]
        for weight in (0.5, 0.25):
            averaged = {name: value.lerp(epoch2[name], weight) for name, value in state.items()}
            model.load_state_dict(averaged)
            del averaged
            logits = predict(model.to(device), encoded, batch_size, tokenizer.pad_token_id, device)
            predicted = logits.argmax(axis=1)
            result = {
                "weight_epoch2": weight,
                "dev_accuracy": float(accuracy_score(target, predicted)),
                "dev_macro_f1": float(f1_score(target, predicted, average="macro")),
            }
            grid.append(result)
            print(result, flush=True)
            if result["dev_accuracy"] > best["dev_accuracy"]:
                best = result
                model.save_pretrained(OUT / "model")
                tokenizer.save_pretrained(OUT / "model")
                np.save(OUT / "dev_logits.npy", logits)
        save_json(OUT / "metrics.json", {"best": best, "validation_grid": grid})
        if best["weight_epoch2"] is None:
            return
        del model, last, state, epoch2
        model = AutoModelForSequenceClassification.from_pretrained(OUT / "model", local_files_only=True)
    model.to(device)
    titles = read_titles(DATA / "test.txt")
    logits = predict(model, encode(tokenizer, titles, length), batch_size, tokenizer.pad_token_id, device)
    np.save(OUT / "test_logits.npy", logits)
    (OUT / "predictions.txt").write_text(
        "\n".join(labels[index] for index in logits.argmax(axis=1)) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(titles)} test predictions", flush=True)


if __name__ == "__main__":
    main()

