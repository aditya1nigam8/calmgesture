"""
train_cnn_torch.py — Model owner: Tanvi Gupta (230911122)

PyTorch port of train_cnn.py. Everything except the framework is held fixed:
the same pose-image input, training-only channel standardization, He
initialization (copied from the seeded NumPy model), plain SGD (lr 0.05,
batch 16), per-epoch shuffling, and minimum-validation-loss early stopping.

--channels mean drops the temporal-std channel. Comparing it with the default
mean_std run is the input-channel ablation; only the input channels change.

USAGE:
    python src/train_cnn_torch.py --data_dir data/processed_jester --output_dir runs/jester_cnn_torch
    python src/train_cnn_torch.py --data_dir data/processed_jester --output_dir runs/jester_cnn_torch_mean --channels mean
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from baseline_inference import benchmark
from cnn_numpy import SimpleCNN
from spatial_cnn_torch import CHANNELS, SpatialCNN, load_checkpoint, load_numpy_weights, pose_channels, predict_probabilities
from utils import dataset_provenance, evaluate, load_training_splits


def fit(model, Xtr, ytr, Xva, yva, epochs=60, batch_size=16, learning_rate=0.05, patience=10):
    """Mirror SimpleCNN.fit: shuffle with default_rng(epoch), mean cross-entropy, plain SGD."""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    Xtr, Xva = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    ytr, yva = torch.as_tensor(ytr, dtype=torch.long), torch.as_tensor(yva, dtype=torch.long)
    history, best_weights, best_epoch = [], None, None
    best_loss, stale = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        order = torch.from_numpy(np.random.default_rng(epoch).permutation(len(Xtr)))
        total_loss = 0.0
        for start in range(0, len(Xtr), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(Xtr[batch]), ytr[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            logits = model(Xva)
            val_loss = loss_function(logits, yva).item()
            val_acc = (logits.argmax(1) == yva).float().mean().item()
        if not np.isfinite(val_loss):
            raise ValueError("CNN validation loss became non-finite")
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(Xtr),
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            stale += 1
        print(f"cnn2d epoch {epoch + 1}: loss={total_loss / len(Xtr):.4f}, validation_loss={val_loss:.4f}, validation_accuracy={val_acc:.3f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_weights)
    return history, best_epoch


def train(data_dir, output_dir, channels="mean_std", epochs=60, batch_size=16, learning_rate=0.05, patience=10, seed=0):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("Training parameters must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh experiment directory: {output_dir}")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    Xtr, Xva, Xte, ytr, yva, yte, classes = load_training_splits(data_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    images = [pose_channels(part, channels) for part in (Xtr, Xva, Xte)]
    mean = images[0].mean(axis=(0, 2, 3), keepdims=True)
    std = images[0].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    images = [(part - mean) / std for part in images]
    label_map = {label: index for index, label in enumerate(classes)}
    ytr_idx, yva_idx = (np.array([label_map[label] for label in part]) for part in (ytr, yva))

    in_channels = CHANNELS[channels]
    model = load_numpy_weights(SpatialCNN(len(classes), in_channels),
                               SimpleCNN(len(classes), in_c=in_channels, seed=seed).state_dict())
    started = time.perf_counter()
    history, best_epoch = fit(model, images[0], ytr_idx, images[1], yva_idx, epochs, batch_size, learning_rate, patience)
    train_time = time.perf_counter() - started

    provenance = dataset_provenance(data_dir)
    checkpoint = {
        "format_version": 1, "architecture": "cnn2d", "channels": channels, "state_dict": model.state_dict(),
        "classes": classes, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "target_frames": int(Xtr.shape[1]), "data": provenance, "seed": seed, "best_epoch": best_epoch,
    }
    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    # Evaluate what will actually be deployed, after a save/load round trip.
    restored, checkpoint = load_checkpoint(checkpoint_path)
    val_probabilities = predict_probabilities(restored, checkpoint, Xva)
    test_probabilities = predict_probabilities(restored, checkpoint, Xte)
    names = np.asarray(classes)
    val_pred = names[val_probabilities.argmax(1)]
    test_pred = names[test_probabilities.argmax(1)]
    metrics = {
        "model": "CNN (PyTorch port, spatial)" + ("" if channels == "mean_std" else ", mean-only input"),
        "owner": "Tanvi Gupta (230911122)",
        "torch_version": str(torch.__version__),
        "data": provenance, "classes": classes, "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "channels": channels, "train_time_s": train_time, "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in restored.parameters()),
        "val": evaluate(yva, val_pred, classes), "test": evaluate(yte, test_pred, classes),
        "latency": benchmark(lambda values: predict_probabilities(restored, checkpoint, values), Xte),
        "training": {"epochs_requested": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                     "optimizer": "SGD", "patience": patience, "seed": seed, "initialization": "NumPy SimpleCNN He init",
                     "selection": "minimum validation loss", "device": "cpu"},
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(output_dir / "predictions.npz", y_true=yte, y_pred=test_pred,
                        probabilities=test_probabilities, classes=names)
    print(f"Saved checkpoint and metrics to {output_dir}; test accuracy={metrics['test']['accuracy']:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/processed_jester")
    parser.add_argument("--output_dir", type=Path, required=True, help="Fresh directory for this experiment")
    parser.add_argument("--channels", choices=sorted(CHANNELS), default="mean_std")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.channels, args.epochs, args.batch_size,
          args.learning_rate, args.patience, args.seed)


if __name__ == "__main__":
    main()
