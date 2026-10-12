"""Extract motion-preserving landmarks from selected Jester frame folders."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_jester import read_manifest
from preprocess import resample_to_fixed_length

FEATURE_VERSION = "first_wrist_xy_median_scale_v1"


def accept_clip(row, detection_rate, threshold):
    """Filter only training foreground clips; retain every evaluation failure."""
    return row["split"] != "train" or row["label"] == "background" or detection_rate >= threshold


def frame_paths(folder):
    paths = sorted(Path(folder).glob("*.jpg"), key=lambda p: int(p.stem))
    if len(paths) < 2:
        raise ValueError(f"Need at least two JPG frames: {folder}")
    return paths


def motion_features(raw, detected, target_frames=30):
    """Keep global XY travel; MediaPipe Z is already wrist-relative per frame.

    Backfill initial missing frames, then hold the last detection across gaps.
    Normalize using the first valid wrist XY and median valid palm scale.
    Input x/z must be corrected for image aspect ratio by the extractor.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(-1, 21, 3).copy()
    detected = np.asarray(detected, dtype=bool)
    if len(raw) != len(detected) or len(raw) < 2 or target_frames < 2:
        raise ValueError("Invalid sequence or target length")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite landmarks")
    valid = np.flatnonzero(detected)
    if not len(valid):
        return np.zeros((target_frames, 63), dtype=np.float32)
    first = valid[0]
    scale = np.median(np.linalg.norm(raw[valid, 9, :2] - raw[valid, 0, :2], axis=1))
    if scale < 1e-6:
        raise ValueError("Degenerate palm scale")
    raw[:first] = raw[first]
    for i in range(first + 1, len(raw)):
        if not detected[i]:
            raw[i] = raw[i - 1]
    origin = raw[first, 0, :2].copy()
    raw[:, :, :2] -= origin
    raw[:, :, 2] -= raw[:, 0:1, 2].copy()
    raw /= scale
    return resample_to_fixed_length(raw.reshape(-1, 63), target_frames)


def extract_clip(paths, model_path, fps=12):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    raw, detected = [], []
    # Reset tracking between clips so neither identities nor timestamps leak.
    with vision.HandLandmarker.create_from_options(options) as detector:
        for i, path in enumerate(paths):
            image = mp.Image.create_from_file(str(path))
            result = detector.detect_for_video(image, round(i * 1000 / fps))
            found = bool(result.hand_landmarks)
            coords = np.zeros((21, 3), dtype=np.float32)
            if found:
                coords = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
                coords[:, [0, 2]] *= image.width / image.height
            raw.append(coords)
            detected.append(found)
    return np.array(raw), np.array(detected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("data/models/hand_landmarker.task"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed_jester"))
    parser.add_argument("--target_frames", type=int, default=30)
    parser.add_argument("--min_detection_rate", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--check_only", action="store_true", help="Check selected frame folders without MediaPipe")
    args = parser.parse_args()
    if not 0 <= args.min_detection_rate <= 1 or not 0 < args.fps <= 120 or args.target_frames < 2:
        parser.error("Invalid detection rate, FPS, or target frame count")
    rows = read_manifest(args.manifest)
    extraction_state = args.frames_dir / "extraction.json"
    if extraction_state.exists():
        state = json.loads(extraction_state.read_text())
        if state["status"] != "complete":
            parser.error("Frame extraction is incomplete; let the download finish before preprocessing")
        if not {row["clip_id"] for row in rows}.issubset(state["clip_ids"]):
            parser.error("Frame extraction does not cover the selected manifest")
    # Preflight every folder before creating outputs or running the detector.
    frames = {row["clip_id"]: frame_paths(args.frames_dir / row["clip_id"]) for row in rows}
    if args.check_only:
        print(f"Found {len(frames)} selected clips, {sum(map(len, frames.values()))} JPG frames")
        return
    if not args.model.is_file():
        parser.error(f"Missing MediaPipe model: {args.model}. See README setup.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "dataset": "20BN-Jester v1", "feature_version": FEATURE_VERSION,
        "source_fps": args.fps, "target_frames": args.target_frames,
        "min_detection_rate": args.min_detection_rate,
        "quality_policy": "Filter low-detection foreground training clips only; retain all validation/test clips",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "landmarker_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "classes": sorted({row["label"] for row in rows}),
        "status": "incomplete", "clips": [],
    }
    metadata_path = args.output_dir / "dataset.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "quality.jsonl").open("x") as log:
        for index, row in enumerate(rows, 1):
            raw, detected = extract_clip(frames[row["clip_id"]], args.model, args.fps)
            rate = float(detected.mean())
            # Lack of a hand is a legitimate background example, not a broken clip.
            accepted = accept_clip(row, rate, args.min_detection_rate)
            record = dict(row, frames=len(raw), detection_rate=rate, accepted=accepted)
            record["low_detection"] = rate < args.min_detection_rate
            if accepted:
                features = motion_features(raw, detected, args.target_frames)
                relative = Path(row["split"]) / row["label"] / f'{row["clip_id"]}.npy'
                target = args.output_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, features)
                record["path"] = relative.as_posix()
                metadata["clips"].append(record)
            else:
                record["reason"] = "low_hand_detection_rate"
            log.write(json.dumps(record) + "\n")
            log.flush()
            if index % 25 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} clips; accepted {len(metadata['clips'])}", flush=True)
    metadata["counts"] = dict(Counter(f'{r["split"]}/{r["label"]}' for r in metadata["clips"]))
    missing = [f"{split}/{label}" for split in ("train", "validation", "test")
               for label in metadata["classes"] if not metadata["counts"].get(f"{split}/{label}")]
    metadata["status"] = "complete" if not missing else "insufficient_data"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if missing:
        raise ValueError(f"No accepted clips in {missing}; inspect quality.jsonl before training")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
