"""Select a reproducible Jester subset without moving or copying image files."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/jester_subset.json"
FIELDS = ["clip_id", "label", "source_label", "source_split", "split"]


def read_annotations(path):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.reader(handle, delimiter=";"), 1):
            if not row:
                continue
            if len(row) != 2 or not row[0].strip().isascii() or not row[0].strip().isdigit():
                raise ValueError(f"{path}:{line}: expected clip_id;gesture_label")
            clip_id, label = (value.strip() for value in row)
            if clip_id in seen:
                raise ValueError(f"Duplicate clip ID {clip_id} in {path}")
            seen.add(clip_id)
            rows.append({"clip_id": clip_id, "source_label": label})
    if not rows:
        raise ValueError(f"No annotations in {path}")
    return rows


def select_rows(train_rows, validation_rows, config):
    train_ids = {row["clip_id"] for row in train_rows}
    if train_ids & {row["clip_id"] for row in validation_rows}:
        raise ValueError("Official training and validation files overlap")
    fraction = config["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    labels = [label for labels in config["classes"].values() for label in labels]
    if len(labels) != len(set(labels)):
        raise ValueError("Source labels may only belong to one output class")
    rng = random.Random(config["seed"])
    selected = []
    # Balance constituent labels, so e.g. forward rolling cannot dominate backward.
    for source_split, rows, cap in (
        ("train", train_rows, config["train_pool_per_class"]),
        ("validation", validation_rows, config["test_per_class"]),
    ):
        for label, sources in config["classes"].items():
            if cap < len(sources) * (2 if source_split == "train" else 1):
                raise ValueError(f"Clip cap too small for {label}")
            for index, source in enumerate(sources):
                candidates = sorted(
                    (row for row in rows if row["source_label"] == source),
                    key=lambda row: int(row["clip_id"]),
                )
                quota = cap // len(sources) + (index < cap % len(sources))
                if len(candidates) < quota:
                    raise ValueError(f"{source_split}: {source} has {len(candidates)} clips; need {quota}. Reduce caps explicitly.")
                rng.shuffle(candidates)
                chosen = candidates[:quota]
                n_val = max(1, min(quota - 1, round(quota * fraction)))
                for i, row in enumerate(chosen):
                    split = "test" if source_split == "validation" else ("validation" if i < n_val else "train")
                    selected.append(dict(row, label=label, source_split=source_split, split=split))
    return sorted(selected, key=lambda row: (row["split"], row["label"], int(row["clip_id"])))


def read_manifest(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected manifest columns: {FIELDS}")
        rows = list(reader)
    seen = set()
    for row in rows:
        clip_id = row["clip_id"]
        if not clip_id.isascii() or not clip_id.isdigit() or clip_id in seen:
            raise ValueError(f"Invalid or duplicate clip ID: {clip_id}")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        if not row["label"] or any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in row["label"]):
            raise ValueError("Invalid output label")
        expected = "validation" if row["split"] == "test" else "train"
        if row["source_split"] != expected:
            raise ValueError("Manifest mixes official split membership")
        seen.add(clip_id)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--validation_csv", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("data/jester/subset.csv"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = select_rows(read_annotations(args.train_csv), read_annotations(args.validation_csv), config)
    summary = {
        "config": config,
        "annotation_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for name, path in (("train", args.train_csv), ("validation", args.validation_csv))},
        "counts": dict(sorted(Counter(f'{row["split"]}/{row["label"]}' for row in rows).items())),
        "evaluation": "Official validation is held out as test; internal validation comes from official train. Not a participant-disjoint guarantee.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError("Manifest output already exists; choose a new path to preserve the experiment")
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], indent=2))
    print(f"Selected {len(rows)} clips -> {args.output}")


if __name__ == "__main__":
    main()
