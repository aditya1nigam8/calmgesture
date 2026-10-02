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
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_dataset, split_dataset, evaluate  # noqa: E402
from cnn_numpy import SimpleCNN  # noqa: E402


def pose_image(X):
    """(N, T, 63) -> (N, 2, 21, 3): channel 0 = mean pose, channel 1 = std over time
    (std exposes how much each landmark moves, which the mean pose alone cannot)."""
    mean = X.mean(axis=1).reshape(-1, 1, 21, 3)
    std = X.std(axis=1).reshape(-1, 1, 21, 3)
    return np.concatenate([mean, std], axis=1).astype(np.float32)


def main(data_dir, output_json, epochs):
    X, y, classes = load_dataset(data_dir)
    imgs = pose_image(X)
    mu = imgs.mean(axis=(0, 2, 3), keepdims=True)
    sd = imgs.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    imgs = (imgs - mu) / sd

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_idx[c] for c in y])

    Xtr, Xva, Xte, ytr, yva, yte = split_dataset(imgs, y_idx)

    model = SimpleCNN(n_classes=len(classes), in_c=2, seed=0)
    t0 = time.time()
    history = model.fit(Xtr, ytr, Xva, yva, epochs=epochs, lr=0.05, batch=16)
    train_time = time.time() - t0

    t1 = time.time()
    test_pred = model.predict(Xte)
    infer_ms = (time.time() - t1) / len(Xte) * 1000
    val_pred = model.predict(Xva)

    idx_to_cls = np.array(classes)
    results = {
        "model": "CNN (from-scratch NumPy, spatial)",
        "owner": "Tanvi Gupta (230911122)",
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "train_time_s": round(train_time, 1),
        "inference_ms_per_clip": round(infer_ms, 3),
        "val": evaluate(idx_to_cls[yva], idx_to_cls[val_pred], classes),
        "test": evaluate(idx_to_cls[yte], idx_to_cls[test_pred], classes),
        "classes": classes,
        "history": history,
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Test accuracy:", results["test"]["accuracy"], "Test macro-F1:", results["test"]["macro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_synthetic")
    parser.add_argument("--output_json", default="results/cnn_metrics.json")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    main(args.data_dir, args.output_json, args.epochs)
