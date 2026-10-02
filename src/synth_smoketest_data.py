"""
synth_smoketest_data.py
CalmGesture — Pipeline Smoke-Test Data Generator
=============================================================

IMPORTANT / HONESTY NOTE FOR THE TEAM:
This script does NOT produce real gesture data. It procedurally
generates synthetic 21-landmark hand sequences that approximate the
*shape* of the five target gesture classes (rolling, shaking, waving,
tapping, open/close) using simple parametric motion + noise, in the
SAME (T, 63) normalized array format that preprocess.py produces from
real video.

Its only purpose is to let train_mlp.py / train_cnn.py be exercised
end-to-end before the real Jester subset and custom webcam clips are
available, so that:
  (a) the preprocessing -> feature -> model -> metric pipeline is
      verified to run without errors, and
  (b) we have a sanity-check accuracy number showing the classifiers
      can separate clearly different synthetic motion patterns.

The accuracy/F1 numbers produced from this synthetic set are NOT
preliminary results on the real dataset and must NOT be reported as
such. Re-run train_mlp.py / train_cnn.py on the output of preprocess.py
(real data) before the Final Report, and ideally before Interim
submission if real clips become available in time.
"""

import os
import numpy as np

N_LANDMARKS = 21
N_COORDS = 3
CLASSES = ["rolling", "shaking", "waving", "tapping", "open_close"]
RNG = np.random.default_rng(42)


def _base_hand(t):
    """A static, roughly hand-shaped landmark layout at time t (unused dim)."""
    base = np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)
    # wrist at origin, fingers splayed out along x with slight y offsets
    for i in range(1, N_LANDMARKS):
        finger = (i - 1) // 4
        joint = (i - 1) % 4
        base[i, 0] = 0.15 * (finger - 2) + 0.05 * joint
        base[i, 1] = -0.08 * (joint + 1)
        base[i, 2] = 0.0
    return base


def _gesture_sequence(label, n_frames=30, noise=0.015):
    base = _base_hand(0)
    t = np.linspace(0, 2 * np.pi, n_frames)
    seq = np.tile(base, (n_frames, 1, 1))

    if label == "rolling":
        radius = 0.08
        seq[:, :, 0] += (radius * np.cos(t))[:, None]
        seq[:, :, 1] += (radius * np.sin(t))[:, None]
    elif label == "shaking":
        seq[:, :, 0] += (0.1 * np.sin(6 * t))[:, None]
    elif label == "waving":
        seq[:, 1:, 1] += (0.12 * np.sin(2 * t))[:, None]
        seq[:, 1:, 0] += (0.03 * t / (2 * np.pi))[:, None]
    elif label == "tapping":
        pulse = (np.sin(8 * t) > 0.6).astype(np.float32) * 0.1
        seq[:, [4, 8, 12, 16, 20], 1] -= pulse[:, None]
    elif label == "open_close":
        spread = 0.08 * np.sin(2 * t)
        for i in range(1, N_LANDMARKS):
            finger = (i - 1) // 4
            seq[:, i, 0] += spread * (finger - 2)
    seq += RNG.normal(0, noise, seq.shape)
    return seq.reshape(n_frames, -1).astype(np.float32)


def generate(output_dir="data/processed_synthetic", n_per_class=40, n_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    for label in CLASSES:
        dst = os.path.join(output_dir, label)
        os.makedirs(dst, exist_ok=True)
        for i in range(n_per_class):
            seq = _gesture_sequence(label, n_frames=n_frames)
            path = os.path.join(dst, f"{label}_{i:03d}.npy")
            np.save(path, seq)
            manifest.append((path, label))
    return manifest


if __name__ == "__main__":
    m = generate()
    print(f"Generated {len(m)} synthetic smoke-test clips across {len(CLASSES)} classes.")
