"""
train_mlp.py — Model owner: Sangini Singh (230911020)

MLP baseline operating on per-clip mean/std landmark features
(MediaPipe 21-landmark, wrist-normalized). Trains an sklearn
MLPClassifier and reports accuracy / macro-F1 on a held-out test split.

USAGE:
    python train_mlp.py --data_dir data/processed            (real data)
    python train_mlp.py --data_dir data/processed_synthetic  (smoke test)
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_training_splits, dataset_provenance, mlp_features, evaluate  # noqa: E402
from baseline_inference import save_baseline, load_baseline, predict_probabilities, benchmark


def main(data_dir, output_json):
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_training_splits(data_dir)
    raw_val, raw_test = X_val, X_test
    X_train, X_val, X_test = (mlp_features(part) for part in (X_train, X_val, X_test))

    scaler = StandardScaler().fit(X_train)  # fit only on the training split
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=1,
        batch_size=32,
        random_state=42,
        early_stopping=False,
    )
    started = time.perf_counter()
    best_loss, stale, best_model, best_epoch = float("inf"), 0, None, 0
    history = []
    for epoch in range(500):
        clf.partial_fit(X_train, y_train, classes=np.asarray(classes))
        validation_loss = log_loss(y_val, clf.predict_proba(X_val), labels=classes)
        if not np.isfinite(validation_loss):
            raise ValueError("MLP validation loss became non-finite")
        history.append({"epoch": epoch + 1, "train_loss": float(clf.loss_), "validation_loss": float(validation_loss)})
        if validation_loss < best_loss:
            best_loss, stale, best_model, best_epoch = validation_loss, 0, copy.deepcopy(clf), epoch + 1
        else:
            stale += 1
        if epoch % 25 == 0:
            print(f"MLP epoch {epoch + 1}: validation_loss={validation_loss:.4f}", flush=True)
        if stale >= 20:
            break
    clf = best_model
    train_time = time.perf_counter() - started

    checkpoint_path = Path(output_json).with_suffix(".joblib")
    save_baseline(checkpoint_path, {"format_version": 1, "architecture": "mlp", "model": clf,
                  "scaler": scaler, "classes": classes, "target_frames": raw_test.shape[1],
                  "data": dataset_provenance(data_dir)})
    restored = load_baseline(checkpoint_path)
    probabilities = predict_probabilities(restored, raw_test)
    test_pred = np.asarray(classes)[probabilities.argmax(1)]
    val_pred = np.asarray(classes)[predict_probabilities(restored, raw_val).argmax(1)]
    results = {
        "model": "MLP",
        "data": dataset_provenance(data_dir),
        "owner": "Sangini Singh (230911020)",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_time_s": train_time,
        "checkpoint": str(checkpoint_path),
        "latency": benchmark(lambda values: predict_probabilities(restored, values), raw_test),
        "training": {"seed": 42, "selection": "minimum shared validation loss", "epochs_run": len(history), "patience": 20, "batch_size": 32},
        "best_epoch": best_epoch,
        "history": history,
        "val": evaluate(y_val, val_pred, classes),
        "test": evaluate(y_test, test_pred, classes),
        "classes": classes,
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(Path(output_json).with_suffix(".predictions.npz"), y_true=y_test, y_pred=test_pred,
                        probabilities=probabilities, classes=np.asarray(classes))
    print(json.dumps({k: v for k, v in results.items() if k not in ("test", "history")}, indent=2))
    print("Test accuracy:", results["test"]["accuracy"], "Test macro-F1:", results["test"]["macro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_synthetic")
    parser.add_argument("--output_json", default="results/mlp_metrics.json")
    args = parser.parse_args()
    main(args.data_dir, args.output_json)
