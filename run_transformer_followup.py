"""Compare a seven-epoch LR schedule, then screen three batch sizes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rerun-constant", action="store_true",
                        help="build the constant reference on the current device and environment")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = root / "experiments/batch_and_tfidf_20261117_124422/summary.json"
    baseline_path = root / "transformer_baseline/outputs/lr_search_20261116_132433/lr_2e-4_restart/metrics.json"
    baseline = read_json(baseline_path)
    write_json(output / "batch_size_one_epoch.json",
               [row for row in read_json(previous) if row["run"].startswith("batch_")])
    write_json(output / "constant_7_reference.json", baseline)

    def run(name, batch_size, epochs, schedule):
        folder = output / name
        folder.mkdir(exist_ok=True)
        metrics_file = folder / "metrics.json"
        if metrics_file.exists():
            return read_json(metrics_file)
        command = [sys.executable, "-X", "utf8", "-u", str(root / "transformer_baseline/train.py"),
                   "--data-dir", "database", "--output-dir", str(folder), "--epochs", str(epochs),
                   "--learning-rate", "2e-4", "--batch-size", str(batch_size), "--max-length", "64",
                   "--threads", "8", "--seed", "2026", "--lr-schedule", schedule,
                   "--warmup-ratio", "0.05", "--early-stop-patience", "3",
                   "--early-stop-min-epochs", "7"]
        checkpoint = folder / "last_checkpoint.pt"
        if checkpoint.exists():
            command.extend(("--resume-checkpoint", str(checkpoint)))
        print(f"Starting {name}", flush=True)
        with (folder / "training.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        return read_json(metrics_file)

    if args.rerun_constant:
        baseline = run("constant_bs64_7e", 64, 7, "constant")
    tuned = run("linear_bs64_7e", 64, 7, "linear")
    keys = ("device", "best_epoch", "dev_accuracy", "dev_macro_f1", "history")
    summary = {"constant_7": {key: baseline[key] for key in keys},
               "linear_7": {key: tuned[key] for key in keys}}
    schedule = "linear" if tuned["dev_macro_f1"] > baseline["dev_macro_f1"] else "constant"
    summary["batch_comparison_schedule"] = schedule
    summary["batch_comparison"] = {}
    write_json(output / "summary.json", summary)
    print(f"Schedule for batch comparison: {schedule}", flush=True)
    for batch_size in (32, 64, 128):
        metrics = run(f"batch_{batch_size}_3e", batch_size, 3, schedule)
        summary["batch_comparison"][str(batch_size)] = {
            key: metrics[key] for key in ("best_epoch", "dev_accuracy", "dev_macro_f1", "history")
        }
        write_json(output / "summary.json", summary)
    print("All experiments completed.", flush=True)


if __name__ == "__main__":
    main()
