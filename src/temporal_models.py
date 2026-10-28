"""Lightweight sequence classifiers for (batch, frames, 63) landmarks."""
import numpy as np
import torch
from torch import nn


class TemporalCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(63, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, sequence):
        # Conv1d convolves over time, not over the landmark-coordinate axis.
        return self.classifier(self.features(sequence.transpose(1, 2)))


class GestureGRU(nn.Module):
    def __init__(self, n_classes, hidden_size=64, pooling="last", dropout=0.2):
        super().__init__()
        if hidden_size < 1 or pooling not in {"last", "mean"} or not 0 <= dropout < 1:
            raise ValueError("Invalid GRU hidden size, pooling, or dropout")
        self.pooling = pooling
        self.gru = nn.GRU(input_size=63, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, n_classes))

    def forward(self, sequence):
        outputs, hidden = self.gru(sequence)
        representation = hidden[-1] if self.pooling == "last" else outputs.mean(dim=1)
        return self.classifier(representation)


def create_model(architecture, n_classes, model_config=None):
    model_config = model_config or {}
    if architecture == "1dcnn":
        if model_config:
            raise ValueError("GRU configuration cannot be applied to 1D-CNN")
        return TemporalCNN(n_classes)
    if architecture == "gru":
        return GestureGRU(n_classes, **model_config)
    raise ValueError(f"Unknown architecture: {architecture}")


def fit_normalizer(train):
    """One mean/std per coordinate, fitted on training clips and frames only."""
    return train.mean(axis=(0, 1), keepdims=True), np.maximum(train.std(axis=(0, 1), keepdims=True), 1e-6)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = create_model(checkpoint["architecture"], len(checkpoint["classes"]),
                         checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def predict_probabilities(model, checkpoint, sequences):
    """Predict from already extracted features, using the saved training scaler."""
    values = torch.as_tensor(sequences, dtype=torch.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (checkpoint["target_frames"], 63):
        raise ValueError("Input must be (clips, target_frames, 63)")
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite input features")
    values = (values - checkpoint["mean"]) / checkpoint["std"]
    with torch.inference_mode():
        return torch.softmax(model(values), dim=1).numpy()
