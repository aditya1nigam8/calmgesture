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
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_dataset, mlp_features, split_dataset, evaluate  # noqa: E402


def main(data_dir, output_json):
    X, y, classes = load_dataset(data_dir)
    feats = mlp_features(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(feats, y)

    scaler = StandardScaler().fit(X_train)  # same per-feature standardisation as the CNN
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    results = {
        "model": "MLP",
        "owner": "Sangini Singh (230911020)",
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "val": evaluate(y_val, val_pred, classes),
        "test": evaluate(y_test, test_pred, classes),
        "classes": classes,
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k not in ("test",)}, indent=2))
    print("Test accuracy:", results["test"]["accuracy"], "Test macro-F1:", results["test"]["macro_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_synthetic")
    parser.add_argument("--output_json", default="results/mlp_metrics.json")
    args = parser.parse_args()
    main(args.data_dir, args.output_json)
