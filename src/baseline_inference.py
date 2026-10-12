"""Persist and run local MLP/NumPy-CNN models on extracted landmark sequences."""
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from cnn_numpy import SimpleCNN
from utils import mlp_features


def pose_image(sequences):
    mean = sequences.mean(axis=1).reshape(-1, 1, 21, 3)
    std = sequences.std(axis=1).reshape(-1, 1, 21, 3)
    return np.concatenate([mean, std], axis=1).astype(np.float32)


def save_baseline(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_baseline(path):
    # Joblib is for our own generated checkpoints, not untrusted downloaded files.
    bundle = joblib.load(path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported baseline checkpoint")
    if bundle["architecture"] == "cnn":
        model = SimpleCNN(len(bundle["classes"]), in_c=2)
        model.load_state_dict(bundle["weights"])
        bundle["model"] = model
    elif bundle["architecture"] != "mlp":
        raise ValueError("Unknown baseline architecture")
    return bundle


def predict_probabilities(bundle, sequences):
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (bundle["target_frames"], 63) or not np.isfinite(values).all():
        raise ValueError("Expected finite (clips, target_frames, 63) landmarks")
    if bundle["architecture"] == "mlp":
        return bundle["model"].predict_proba(bundle["scaler"].transform(mlp_features(values)))
    images = (pose_image(values) - bundle["mean"]) / bundle["std"]
    # Keep peak memory bounded when evaluating many clips.
    probabilities = []
    for start in range(0, len(images), 32):
        logits = bundle["model"].forward(images[start:start + 32])
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities.append(exp / exp.sum(axis=1, keepdims=True))
    return np.concatenate(probabilities)


def benchmark(predict, sequences):
    timings = []
    with threadpool_limits(limits=4):
        for _ in range(10):
            predict(sequences[:1])
        for index in range(100):
            i = index % len(sequences)
            start = time.perf_counter()
            predict(sequences[i:i + 1])
            timings.append((time.perf_counter() - start) * 1000)
    return {"device": "cpu", "threads": 4, "batch_size": 1, "warmup": 10, "repeats": 100,
            "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "scope": "feature aggregation/normalization + classifier + probabilities; excludes camera, landmarks, window collection and sound"}
