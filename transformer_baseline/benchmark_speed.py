"""Short CPU/device speed measurement; never saves or replaces model weights."""
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import TitleDataset, make_collate, read_labeled


def main():
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    data = root.parent / 'database'
    torch.manual_seed(2026)
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_texts, train_labels = read_labeled(data / 'train.txt')
    dev_texts, _ = read_labeled(data / 'dev.txt')
    classes = sorted(set(train_labels))
    mapping = {label: i for i, label in enumerate(classes)}
    tokenizer = AutoTokenizer.from_pretrained(root / 'pretrained/chinese_roberta_tiny', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        root / 'pretrained/chinese_roberta_tiny', num_labels=len(classes), local_files_only=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    train_loader = DataLoader(TitleDataset(train_texts, [mapping[x] for x in train_labels]),
        batch_size=64, shuffle=True, collate_fn=make_collate(tokenizer, 64, True))
    dev_loader = DataLoader(TitleDataset(dev_texts), batch_size=64, shuffle=False,
        collate_fn=make_collate(tokenizer, 64, False))

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    setup_seconds = time.perf_counter() - started
    model.train()
    iterator = iter(train_loader)

    def train_step():
        batch = {k: v.to(device) for k, v in next(iterator).items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        # Match the scalar transfer in the training script.
        return loss.item()

    for _ in range(10):
        train_step()
    sync()
    print('Warmup complete; timing 200 training batches.', flush=True)
    blocks = []
    for block in range(4):
        sync()
        begin = time.perf_counter()
        for _ in range(50):
            train_step()
        sync()
        seconds = time.perf_counter() - begin
        blocks.append(seconds)
        print(f'Train block {block + 1}/4: {seconds:.2f}s, {seconds / 50:.4f}s/batch', flush=True)

    model.eval()
    dev_iterator = iter(dev_loader)
    with torch.inference_mode():
        for _ in range(5):
            batch = {k: v.to(device) for k, v in next(dev_iterator).items()}
            model(**batch).logits.argmax(dim=1).cpu().tolist()
        sync()
        begin = time.perf_counter()
        for _ in range(100):
            batch = {k: v.to(device) for k, v in next(dev_iterator).items()}
            model(**batch).logits.argmax(dim=1).cpu().tolist()
        sync()
        dev_seconds = time.perf_counter() - begin
    train_per_batch = sum(blocks) / 200
    dev_per_batch = dev_seconds / 100
    epoch_seconds = len(train_loader) * train_per_batch + len(dev_loader) * dev_per_batch
    report = {
        'device': str(device), 'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'threads': 8, 'batch_size': 64,
        'max_length': 64, 'learning_rate': 2e-4, 'data_dir': str(data),
        'setup_seconds': setup_seconds, 'training_batches_per_epoch': len(train_loader),
        'validation_batches_per_epoch': len(dev_loader), 'warmup_train_batches': 10,
        'measured_train_batches': 200, 'train_50_batch_blocks_seconds': blocks,
        'train_seconds_per_batch': train_per_batch, 'measured_dev_batches': 100,
        'dev_seconds_per_batch': dev_per_batch,
        'estimated_epoch_seconds': epoch_seconds, 'estimated_five_epochs_seconds': epoch_seconds * 5,
        'estimated_five_epochs_block_range_seconds': [
            5 * (len(train_loader) * x / 50 + len(dev_loader) * dev_per_batch)
            for x in (min(blocks), max(blocks))],
        'benchmark_elapsed_seconds': time.perf_counter() - started,
        'limitations': 'Extrapolation; excludes checkpoint writes, full metric computation and final test prediction. Long-run thermal/load effects may differ. No trained weights saved.'
    }
    output = root / 'outputs' / ('speed_benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'timing.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f'Report: {output / "timing.json"}', flush=True)


if __name__ == '__main__':
    main()
