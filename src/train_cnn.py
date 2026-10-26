"""
train_cnn.py — Model owner: Tanvi Gupta (230911122)

Spatial CNN on the per-clip (mean, std) hand-pose "image" (21 landmarks x 3
coords, wrist-normalized). Uses the from-scratch NumPy CNN in
cnn_numpy.py.

USAGE:
    python train_cnn.py --data_dir data/processed            (real data)
    python train_cnn.py --data_dir data/processed_synthetic  (smoke test)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_training_splits, dataset_provenance, evaluate  # noqa: E402
from cnn_numpy import SimpleCNN  # noqa: E402
from baseline_inference import pose_image, save_baseline, load_baseline, predict_probabilities, benchmark


def standardize_splits(train, validation, test):
    """Fit channel statistics on training clips only; preserve held-out data."""
    mu = train.mean(axis=(0, 2, 3), keepdims=True)
    sd = train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return tuple((part - mu) / sd for part in (train, validation, test))


def main(data_dir, output_json, epochs):
    Xtr, Xva, Xte, ytr, yva, yte, classes = load_training_splits(data_dir)
    raw_val, raw_test = Xva, Xte
    Xtr, Xva, Xte = (pose_image(part) for part in (Xtr, Xva, Xte))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    ytr, yva, yte = (np.array([cls_to_idx[c] for c in part]) for part in (ytr, yva, yte))
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xtr, Xva, Xte = ((part - mean) / std for part in (Xtr, Xva, Xte))

    model = SimpleCNN(n_classes=len(classes), in_c=2, seed=0)
    t0 = time.time()
    history = model.fit(Xtr, ytr, Xva, yva, epochs=epochs, lr=0.05, batch=16)
    train_time = time.time() - t0

    checkpoint_path = Path(output_json).with_suffix(".joblib")
    save_baseline(checkpoint_path, {"format_version": 1, "architecture": "cnn", "weights": model.state_dict(),
                  "classes": classes, "target_frames": raw_test.shape[1], "mean": mean, "std": std,
                  "data": dataset_provenance(data_dir)})
    restored = load_baseline(checkpoint_path)
    probabilities = predict_probabilities(restored, raw_test)
    test_pred = probabilities.argmax(1)
    val_pred = predict_probabilities(restored, raw_val).argmax(1)

    idx_to_cls = np.array(classes)
    results = {
        "model": "CNN (from-scratch NumPy, spatial)",
        "data": dataset_provenance(data_dir),
        "owner": "Tanvi Gupta (230911122)",
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "train_time_s": round(train_time, 1),
        "latency": benchmark(lambda values: predict_probabilities(restored, values), raw_test),
        "checkpoint": str(checkpoint_path),
        "best_epoch": model.best_epoch,
        "training": {"selection": "minimum validation loss", "learning_rate": 0.05, "batch_size": 16, "seed": 0},
        "val": evaluate(idx_to_cls[yva], idx_to_cls[val_pred], classes),
        "test": evaluate(idx_to_cls[yte], idx_to_cls[test_pred], classes),
        "classes": classes,
        "history": history,
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(Path(output_json).with_suffix(".predictions.npz"), y_true=idx_to_cls[yte],
                        y_pred=idx_to_cls[test_pred], probabilities=probabilities, classes=idx_to_cls)
    print("Test accuracy:", results["test"]["accuracy"], "Test macro-F1:", results["test"]["macro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_synthetic")
    parser.add_argument("--output_json", default="results/cnn_metrics.json")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    main(args.data_dir, args.output_json, args.epochs)
