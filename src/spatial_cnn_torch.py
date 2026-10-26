"""PyTorch port of the NumPy spatial CNN (model owner: Tanvi Gupta, 230911122).

Same architecture and input as cnn_numpy.SimpleCNN, so NumPy weights load
unchanged (dense matrices are transposed) and both give the same logits:
    Conv(C->8, 3x3, pad=1) -> ReLU -> Conv(8->16, 3x3, pad=1) -> ReLU
    Flatten -> Dense(32) -> ReLU -> Dense(n_classes)
Input is the 21x3 pose image with C channels: temporal mean (+ temporal std).
This is a framework port only; it is not an adaptation of Gesture-Symphony.

Convert an existing NumPy checkpoint and check that it reproduces predictions:
    python src/spatial_cnn_torch.py --from_numpy runs/jester_v1/cnn/metrics.joblib \
        --output runs/jester_v1/cnn/model.pt --data_dir data/processed_jester
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baseline_inference import load_baseline, pose_image, predict_probabilities as numpy_probabilities
from utils import load_training_splits

# Channel 0 of pose_image is the temporal mean, channel 1 the temporal std.
CHANNELS = {"mean_std": 2, "mean": 1}


class SpatialCNN(nn.Module):
    def __init__(self, n_classes, in_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 21 * 3, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, images):
        x = torch.relu(self.conv1(images))
        x = torch.relu(self.conv2(x))
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def load_numpy_weights(model, state):
    """Copy a SimpleCNN.state_dict() into the port; NumPy dense W is (in, out)."""
    converted = {}
    for name in ("conv1", "conv2", "fc1", "fc2"):
        weight = np.asarray(state[f"{name}.W"], dtype=np.float32)
        converted[f"{name}.weight"] = torch.from_numpy(weight.T.copy() if name.startswith("fc") else weight)
        converted[f"{name}.bias"] = torch.from_numpy(np.asarray(state[f"{name}.b"], dtype=np.float32))
    model.load_state_dict(converted)
    return model


def pose_channels(sequences, channels):
    return pose_image(sequences)[:, :CHANNELS[channels]]


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "cnn2d":
        raise ValueError("Unsupported spatial CNN checkpoint")
    model = SpatialCNN(len(checkpoint["classes"]), CHANNELS[checkpoint["channels"]])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_probabilities(model, checkpoint, sequences):
    """Pose-image features, saved training channel statistics, then softmax."""
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (checkpoint["target_frames"], 63) or not np.isfinite(values).all():
        raise ValueError("Expected finite (clips, target_frames, 63) landmarks")
    images = (torch.from_numpy(pose_channels(values, checkpoint["channels"])) - checkpoint["mean"]) / checkpoint["std"]
    with torch.inference_mode():
        return torch.softmax(model(images), dim=1).numpy()


def checkpoint_from_numpy(bundle):
    """Build a port checkpoint from a loaded NumPy CNN .joblib bundle."""
    if bundle["architecture"] != "cnn":
        raise ValueError("Expected a NumPy CNN checkpoint")
    model = load_numpy_weights(SpatialCNN(len(bundle["classes"])), bundle["weights"])
    return {"format_version": 1, "architecture": "cnn2d", "channels": "mean_std",
            "state_dict": model.state_dict(), "classes": list(bundle["classes"]),
            "mean": torch.from_numpy(np.asarray(bundle["mean"], dtype=np.float32)),
            "std": torch.from_numpy(np.asarray(bundle["std"], dtype=np.float32)),
            "target_frames": int(bundle["target_frames"]), "data": bundle["data"],
            "converted_from": "NumPy SimpleCNN checkpoint"}


def main():
    parser = argparse.ArgumentParser(description="Convert a NumPy CNN checkpoint to the PyTorch port.")
    parser.add_argument("--from_numpy", type=Path, required=True, help="Locally generated train_cnn.py .joblib")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, help="Optional: compare predictions on the held-out split")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    bundle = load_baseline(args.from_numpy)
    torch.save(checkpoint_from_numpy(bundle), args.output)
    model, checkpoint = load_checkpoint(args.output)
    if args.data_dir is not None:
        test = load_training_splits(args.data_dir)[2]
        expected = numpy_probabilities(bundle, test)
        actual = predict_probabilities(model, checkpoint, test)
        agreement = float((expected.argmax(1) == actual.argmax(1)).mean())
        print(f"Held-out clips: {len(test)}; argmax agreement={agreement:.4f}; "
              f"max probability difference={np.abs(expected - actual).max():.2e}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
