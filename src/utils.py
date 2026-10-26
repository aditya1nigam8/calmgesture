"""utils.py — shared loading, feature engineering, split and metrics helpers."""

import os
import json
import hashlib
from pathlib import Path
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


def load_training_splits(processed_dir):
    """Use declared Jester splits, or the historical split for synthetic data."""
    root = Path(processed_dir)
    metadata_path = root / "dataset.json"
    if not metadata_path.exists():
        X, y, classes = load_dataset(processed_dir)
        return (*split_dataset(X, y), classes)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Jester preprocessing is incomplete; inspect dataset.json and quality.jsonl")
    classes = metadata["classes"]
    parts = {split: ([], []) for split in ("train", "validation", "test")}
    seen = set()
    for record in metadata["clips"]:
        if record["clip_id"] in seen:
            raise ValueError("Duplicate clip across dataset splits")
        seen.add(record["clip_id"])
        split = record["split"]
        expected_source = "validation" if split == "test" else "train"
        if record["source_split"] != expected_source:
            raise ValueError("Official split membership changed")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Clip path is outside dataset directory")
        clip = np.load(path, allow_pickle=False)
        if clip.shape != (metadata["target_frames"], 63) or not np.isfinite(clip).all():
            raise ValueError(f"Invalid landmark array: {path}")
        parts[split][0].append(clip)
        parts[split][1].append(record["label"])
    for split, (_, labels) in parts.items():
        if set(labels) != set(classes):
            raise ValueError(f"Missing or unexpected classes in {split}")
    arrays = [np.stack(parts[s][0]) for s in parts]
    labels = [np.array(parts[s][1]) for s in parts]
    return (*arrays, *labels, classes)


def dataset_provenance(processed_dir):
    """Keep real and synthetic experiments explicitly distinguishable."""
    root = Path(processed_dir)
    metadata = root / "dataset.json"
    if metadata.exists():
        info = json.loads(metadata.read_text())
        return {"dataset": info["dataset"], "feature_version": info["feature_version"],
                "manifest_sha256": info["manifest_sha256"],
                "preprocessing_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                "split_protocol": "official train -> train/validation; official validation -> held-out test"}
    return {"dataset": "synthetic pipeline check" if root.name == "processed_synthetic" else "unverified legacy dataset",
            "split_protocol": "clip-stratified random split; seed=42"}


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
