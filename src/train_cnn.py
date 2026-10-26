"""
train_cnn.py — Model owner: Tanvi Gupta (230911122)

Spatial CNN on the per-clip hand-pose "image" (21 landmarks x 3 coords,
wrist-normalized) with summary channels over time: mean, std, or both
(default). Uses the from-scratch NumPy CNN in cnn_numpy.py.

Channel normalization is fitted on the training split only. Training uses
early stopping on validation loss and restores the best epoch; the
checkpoint (.npz) stores the weights plus mu/sd/classes needed to predict.

USAGE:
    python train_cnn.py --data_dir data/processed            (real data)
    python train_cnn.py --data_dir data/processed_synthetic  (smoke test)
    python train_cnn.py --channels mean                      (mean-only input)
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

CHANNELS = ("mean", "std", "mean_std")


def pose_image(X, channels="mean_std"):
    """(N, T, 63) -> (N, C, 21, 3). "mean" = mean pose, "std" = std over time
    (how much each landmark moves, which the mean pose alone cannot show),
    "mean_std" = both stacked as two channels."""
    parts = []
    if channels in ("mean", "mean_std"):
        parts.append(X.mean(axis=1).reshape(-1, 1, 21, 3))
    if channels in ("std", "mean_std"):
        parts.append(X.std(axis=1).reshape(-1, 1, 21, 3))
    return np.concatenate(parts, axis=1).astype(np.float32)


def measure_latency_ms(model, X, warmup=10, repeats=100):
    """Per-clip classifier latency: batch size 1, `warmup` untimed calls, then
    `repeats` timed calls. Excludes capture, landmark extraction and audio."""
    x = X[:1]
    for _ in range(warmup):
        model.forward(x)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.forward(x)
        times.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
            "batch_size": 1, "warmup": warmup, "repeats": repeats}


def train_and_evaluate(X, y, classes, channels="mean_std", epochs=60, lr=0.05, batch=16,
                       patience=10, seed=0, checkpoint=None, verbose=True):
    """Train on the shared split and return a results dict (val + test metrics)."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_idx[c] for c in y])
    imgs = pose_image(X, channels)

    Xtr, Xva, Xte, ytr, yva, yte = split_dataset(imgs, y_idx)
    mu = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    sd = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xtr, Xva, Xte = [(a - mu) / sd for a in (Xtr, Xva, Xte)]

    model = SimpleCNN(n_classes=len(classes), in_c=imgs.shape[1], seed=seed)
    t0 = time.time()
    history = model.fit(Xtr, ytr, Xva, yva, epochs=epochs, lr=lr, batch=batch,
                        patience=patience, verbose=verbose)
    train_time = time.time() - t0

    test_pred = model.predict(Xte)
    val_pred = model.predict(Xva)
    if checkpoint:
        os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
        model.save(checkpoint, mu=mu, sd=sd, classes=np.array(classes), channels=channels)

    idx_to_cls = np.array(classes)
    return {
        "model": "CNN (from-scratch NumPy, spatial)",
        "owner": "Tanvi Gupta (230911122)",
        "channels": channels,
        "hyperparameters": {"lr": lr, "batch": batch, "max_epochs": epochs,
                            "patience": patience, "seed": seed},
        "epochs_run": len(history),
        "restored_epoch": model.best_epoch,
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "train_time_s": round(train_time, 1),
        "latency": measure_latency_ms(model, Xte),
        "val": evaluate(idx_to_cls[yva], idx_to_cls[val_pred], classes),
        "test": evaluate(idx_to_cls[yte], idx_to_cls[test_pred], classes),
        "classes": classes,
        "history": history,
    }


def main(data_dir, output_json, epochs, channels, patience, checkpoint, lr):
    X, y, classes = load_dataset(data_dir)
    results = train_and_evaluate(X, y, classes, channels=channels, epochs=epochs, lr=lr,
                                 patience=patience, checkpoint=checkpoint)
    results["data_dir"] = data_dir
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Restored epoch:", results["restored_epoch"], "of", results["epochs_run"])
    print("Test accuracy:", results["test"]["accuracy"], "Test macro-F1:", results["test"]["macro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_synthetic")
    parser.add_argument("--output_json", default="results/cnn_metrics.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--channels", choices=CHANNELS, default="mean_std")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--checkpoint", default="models/cnn_numpy.npz")
    args = parser.parse_args()
    main(args.data_dir, args.output_json, args.epochs, args.channels, args.patience, args.checkpoint,
         args.lr)
