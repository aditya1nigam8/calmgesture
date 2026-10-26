"""Measure landmark detection on a few available Jester clips, not accuracy."""
import argparse
import json
from pathlib import Path

from prepare_jester import read_manifest
from preprocess_jester import extract_clip, frame_paths, motion_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/jester/subset.csv")
    parser.add_argument("--frames_dir", type=Path, default=Path("data/jester/frames"))
    parser.add_argument("--model", type=Path, default=Path("data/models/hand_landmarker.task"))
    parser.add_argument("--per_class", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/jester_preprocessing_pilot.json"))
    args = parser.parse_args()
    results, counts = [], {}
    for row in read_manifest(args.manifest):
        if row["split"] != "train":
            continue
        label = row["label"]
        if counts.get(label, 0) >= args.per_class:
            continue
        folder = args.frames_dir / row["clip_id"]
        if not folder.is_dir():
            continue
        paths = frame_paths(folder)
        raw, detected = extract_clip(paths, args.model)
        features = motion_features(raw, detected)
        record = dict(row, frames=len(paths), detection_rate=float(detected.mean()), shape=list(features.shape))
        results.append(record)
        counts[label] = counts.get(label, 0) + 1
        print(json.dumps(record), flush=True)
    if not results:
        raise ValueError("No available selected clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "note": "Small feasibility check on currently available training frames; not recognition accuracy or full-dataset validation.",
        "clips": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
