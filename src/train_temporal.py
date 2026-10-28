"""Train 1D-CNN/GRU on the same split as MLP/CNN and save deployable weights.

Model assignments in the synopsis: 1D-CNN — Aditya; GRU — Shubh.
This implementation does not assert individual authorship or contribution.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from temporal_models import create_model, fit_normalizer, load_checkpoint, predict_probabilities
from utils import dataset_provenance, evaluate, load_training_splits


def train(architecture, data_dir, output_dir, epochs=50, batch_size=32, learning_rate=0.001, patience=10, seed=42,
          model_config=None, evaluate_test=True):
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
    mean, std = fit_normalizer(Xtr)
    arrays = [torch.from_numpy(((part - mean) / std).astype(np.float32)) for part in (Xtr, Xva, Xte)]
    label_map = {label: index for index, label in enumerate(classes)}
    labels = [torch.tensor([label_map[label] for label in part], dtype=torch.long) for part in (ytr, yva, yte)]
    loader = DataLoader(TensorDataset(arrays[0], labels[0]), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = create_model(architecture, len(classes), model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history, best_weights = [], None
    best_loss, stale = float("inf"), 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(features)
        model.eval()
        with torch.inference_mode():
            logits = model(arrays[1])
            val_loss = loss_function(logits, labels[1]).item()
            val_acc = (logits.argmax(1) == labels[1]).float().mean().item()
        if not np.isfinite(val_loss):
            raise ValueError("Training diverged; validation loss is non-finite")
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(Xtr),
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            stale += 1
        print(f"{architecture} epoch {epoch + 1}: loss={total_loss / len(Xtr):.4f}, validation_loss={val_loss:.4f}, validation_accuracy={val_acc:.3f}", flush=True)
        if stale >= patience:
            break
    train_time = time.perf_counter() - started
    provenance = dataset_provenance(data_dir)
    checkpoint = {
        "format_version": 1, "architecture": architecture, "state_dict": best_weights,
        "classes": classes, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "target_frames": int(Xtr.shape[1]), "data": provenance, "seed": seed,
        "best_epoch": best_epoch,
        "model_config": model_config or {},
    }
    checkpoint_path = output_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    # Evaluate what will actually be deployed, after a save/load round trip.
    restored, checkpoint = load_checkpoint(checkpoint_path)
    val_probabilities = predict_probabilities(restored, checkpoint, Xva)
    test_probabilities = predict_probabilities(restored, checkpoint, Xte) if evaluate_test else None
    names = np.asarray(classes)
    val_pred = names[val_probabilities.argmax(1)]
    test_pred = names[test_probabilities.argmax(1)] if evaluate_test else None
    # Batch-size-one CPU latency includes normalization + model + softmax only.
    for _ in range(10):
        predict_probabilities(restored, checkpoint, Xva[:1])
    timings = []
    for index in range(100):
        sample = Xva[index % len(Xva):index % len(Xva) + 1]
        start = time.perf_counter()
        predict_probabilities(restored, checkpoint, sample)
        timings.append((time.perf_counter() - start) * 1000)
    metrics = {
        "model": "1D-CNN" if architecture == "1dcnn" else "GRU",
        "torch_version": str(torch.__version__),
        "data": provenance, "classes": classes, "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "train_time_s": train_time, "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in restored.parameters()),
        "model_config": model_config or {},
        "val": evaluate(yva, val_pred, classes),
        "test": evaluate(yte, test_pred, classes) if evaluate_test else None,
        "latency": {"device": "cpu", "threads": 4, "batch_size": 1, "warmup": 10, "repeats": 100,
                    "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
                    "scope": "feature normalization + classifier + softmax; excludes camera, landmarks, window collection and sound"},
        "training": {"epochs_requested": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                     "patience": patience, "seed": seed, "selection": "minimum validation loss", "device": "cpu"},
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(output_dir / "validation_predictions.npz", y_true=yva, y_pred=val_pred,
                        probabilities=val_probabilities, classes=names)
    if evaluate_test:
        np.savez_compressed(output_dir / "predictions.npz", y_true=yte, y_pred=test_pred,
                            probabilities=test_probabilities, classes=names)
    print(f"Saved checkpoint and metrics to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["1dcnn", "gru"], required=True)
    parser.add_argument("--data_dir", default="data/processed_jester")
    parser.add_argument("--output_dir", type=Path, required=True, help="Fresh directory for this experiment")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=64, help="GRU hidden units")
    parser.add_argument("--pooling", choices=["last", "mean"], default="last", help="GRU state readout")
    parser.add_argument("--validation_only", action="store_true", help="Do not score the held-out test")
    args = parser.parse_args()
    if args.model != "gru" and (args.hidden_size != 64 or args.pooling != "last"):
        parser.error("--hidden_size and --pooling apply only to GRU")
    config = {"hidden_size": args.hidden_size, "pooling": args.pooling} if args.model == "gru" else None
    train(args.model, args.data_dir, args.output_dir, args.epochs, args.batch_size,
          args.learning_rate, args.patience, args.seed, model_config=config,
          evaluate_test=not args.validation_only)


if __name__ == "__main__":
    main()
