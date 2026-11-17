"""Run the full-data batch-size and TF-IDF validation comparisons."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


def main():
    root = Path(__file__).resolve().parent
    output = root / "experiments" / ("batch_and_tfidf_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    summary = []

    def run(name, script, *options):
        folder = output / name
        folder.mkdir()
        command = [sys.executable, "-X", "utf8", "-u", str(root / script),
                   "--data-dir", "database", "--output-dir", str(folder), *options]
        print(f"Starting {name}", flush=True)
        started = perf_counter()
        with (folder / "training.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        row = {"run": name, "seconds": perf_counter() - started, "exit_code": result.returncode}
        if result.returncode == 0:
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            row.update(accuracy=metrics["dev_accuracy"], macro_f1=metrics["dev_macro_f1"],
                       settings=metrics["settings"])
        summary.append(row)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        return row

    for size in (32, 64, 128):
        run(f"batch_{size}", "transformer_baseline/train.py", "--epochs", "1",
            "--learning-rate", "2e-4", "--batch-size", str(size), "--max-length", "64",
            "--threads", "8", "--seed", "2026")

    def svm(c, lo, hi):
        return run(f"svm_c{c}_char{lo}-{hi}", "tf-idf_baseline/train.py",
                   "--c", str(c), "--min-df", "5", "--ngram-range", str(lo), str(hi),
                   "--dev-only")

    c_results = [svm(c, 1, 2) for c in (1.0, 0.25, 4.0)]
    completed = [row for row in c_results if row["exit_code"] == 0]
    best_c = max(completed, key=lambda row: row["macro_f1"])["settings"]["c"] if completed else 1.0
    svm(best_c, 1, 3)
    svm(best_c, 2, 3)
    if any(row["exit_code"] for row in summary):
        raise SystemExit("One or more experiments failed; see the individual logs.")
    print("All experiments completed.", flush=True)


if __name__ == "__main__":
    main()
