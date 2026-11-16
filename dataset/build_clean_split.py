"""Create one reproducible local split without exact-title duplicates or conflicts."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "clean_split"
SEED = 2026
OFFICIAL_COUNTS = {"train": 752_471, "dev": 80_000, "test": 83_599}
OFFICIAL_TOTAL = sum(OFFICIAL_COUNTS.values())


def main() -> None:
    pair_counts: Counter[tuple[str, str]] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    for source in ("train", "dev"):
        class_counts: Counter[str] = Counter()
        with (ROOT / f"{source}.txt").open(encoding="utf-8-sig") as handle:
            for line_number, raw in enumerate(handle, 1):
                parts = raw.rstrip("\r\n").rsplit("\t", 1)
                if len(parts) != 2 or not all(parts):
                    raise ValueError(f"{source}.txt:{line_number}: expected title<TAB>label")
                title, label = parts
                pair_counts[(title, label)] += 1
                class_counts[label] += 1
        source_counts[source] = class_counts

    labels_by_title: dict[str, set[str]] = defaultdict(set)
    for title, label in pair_counts:
        labels_by_title[title].add(label)
    conflicts = {title for title, labels in labels_by_title.items() if len(labels) > 1}
    by_label: dict[str, list[str]] = defaultdict(list)
    for title, labels in labels_by_title.items():
        if title not in conflicts:
            by_label[next(iter(labels))].append(title)

    # Sort before shuffling so reruns do not depend on source-file row order.
    rows: dict[str, list[tuple[str, str]]] = {name: [] for name in OFFICIAL_COUNTS}
    class_split_counts: dict[str, dict[str, int]] = {}
    for label in sorted(by_label):
        titles = sorted(by_label[label])
        random.Random(f"{SEED}:{label}").shuffle(titles)
        n_test = round(len(titles) * OFFICIAL_COUNTS["test"] / OFFICIAL_TOTAL)
        n_dev = round(len(titles) * OFFICIAL_COUNTS["dev"] / OFFICIAL_TOTAL)
        parts = {
            "test": titles[:n_test],
            "dev": titles[n_test : n_test + n_dev],
            "train": titles[n_test + n_dev :],
        }
        class_split_counts[label] = {name: len(part) for name, part in parts.items()}
        for name, part in parts.items():
            rows[name].extend((title, label) for title in part)

    OUTPUT.mkdir(exist_ok=True)
    for name, items in rows.items():
        random.Random(f"{SEED}:{name}").shuffle(items)
        path = OUTPUT / ("test_labeled.txt" if name == "test" else f"{name}.txt")
        with path.open("w", encoding="utf-8", newline="") as handle:
            for title, label in items:
                handle.write(f"{title}\t{label}\n")
    with (OUTPUT / "test.txt").open("w", encoding="utf-8", newline="") as handle:
        for title, _ in rows["test"]:
            handle.write(f"{title}\n")

    with (OUTPUT / "class_counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "original_train", "original_dev", "clean_total", "new_train", "new_dev", "new_test"))
        for label in sorted(by_label):
            writer.writerow((label, source_counts["train"][label], source_counts["dev"][label],
                             len(by_label[label]), class_split_counts[label]["train"],
                             class_split_counts[label]["dev"], class_split_counts[label]["test"]))

    conflict_rows = sum(count for (title, _), count in pair_counts.items() if title in conflicts)
    report = {
        "seed": SEED,
        "official_counts": OFFICIAL_COUNTS,
        "official_ratio": {name: count / OFFICIAL_TOTAL for name, count in OFFICIAL_COUNTS.items()},
        "input_labeled_rows": sum(pair_counts.values()),
        "extra_same_title_same_label_rows": sum(count - 1 for count in pair_counts.values()),
        "conflicting_titles_removed": len(conflicts),
        "conflicting_rows_removed": conflict_rows,
        "note": "Duplicate and conflicting row counts overlap; do not add them to obtain total removed rows.",
        "clean_unique_titles": sum(map(len, by_label.values())),
        "new_split_rows": {name: len(items) for name, items in rows.items()},
        "class_split_counts": class_split_counts,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clean_unique_titles": report["clean_unique_titles"],
                      "new_split_rows": report["new_split_rows"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
