"""utils.py — shared loading, feature engineering, split and metrics helpers."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_dataset(processed_dir):
    """Load all .npy clips under processed_dir/<label>/*.npy.

    Returns:
        X: (N, T, 63) float32 array
        y: (N,) string label array
        classes: sorted list of class names
    """
    classes = sorted(
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    )
    X, y = [], []
    for label in classes:
        d = os.path.join(processed_dir, label)
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".npy"):
                X.append(np.load(os.path.join(d, fname)))
            else:
                continue
            y.append(label)
    return np.stack(X, axis=0), np.array(y), classes


def mlp_features(X):
    """Flatten per-clip sequence into a tabular feature vector: per-landmark
    mean and std across the time axis -> (N, 126) feature matrix."""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    return np.concatenate([mean, std], axis=1)


def split_dataset(X, y, test_size=0.2, val_size=0.1, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - rel_val, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return {"accuracy": float(acc), "macro_f1": float(f1), "confusion_matrix": cm.tolist()}
