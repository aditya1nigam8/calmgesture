"""
cnn_channel_ablation.py — Model owner: Tanvi Gupta (230911122)

Input-channel ablation for the spatial CNN: mean-only vs std-only vs
mean+std pose images. Held fixed: shared 70/10/20 split (seed 42),
architecture, initialization seed, batch size and early-stopping rule.

Learning rate is chosen per (channels, seed) from a small grid using
validation macro-F1 only (validation accuracy as tie-breaker); the test
split never influences the choice. This keeps one setting from looking
worse merely because a single fixed learning rate makes it diverge.
With few seeds this is a small-sample comparison, not a significance test.

USAGE:
    python src/cnn_channel_ablation.py --data_dir data/processed \
        --output_json results/cnn_channel_ablation_real.json --seeds 0 1 2
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_dataset  # noqa: E402
from train_cnn import CHANNELS, train_and_evaluate  # noqa: E402


def main(data_dir, output_json, epochs, patience, seeds, lrs):
    X, y, classes = load_dataset(data_dir)
    runs = []
    for channels in CHANNELS:
        for seed in seeds:
            candidates = [train_and_evaluate(X, y, classes, channels=channels, epochs=epochs, lr=lr,
                                             patience=patience, seed=seed, verbose=False)
                          for lr in lrs]
            r = max(candidates, key=lambda c: (c["val"]["macro_f1"], c["val"]["accuracy"]))
            runs.append({
                "channels": channels,
                "seed": seed,
                "selected_lr": r["hyperparameters"]["lr"],
                "val_macro_f1_by_lr": {str(c["hyperparameters"]["lr"]): c["val"]["macro_f1"]
                                       for c in candidates},
                "restored_epoch": r["restored_epoch"],
                "epochs_run": r["epochs_run"],
                "val_accuracy": r["val"]["accuracy"],
                "val_macro_f1": r["val"]["macro_f1"],
                "test_accuracy": r["test"]["accuracy"],
                "test_macro_f1": r["test"]["macro_f1"],
                "test_confusion_matrix": r["test"]["confusion_matrix"],
                "latency_median_ms": r["latency"]["median_ms"],
            })

    summary = {}
    for channels in CHANNELS:
        rs = [r for r in runs if r["channels"] == channels]
        summary[channels] = {k: {"mean": float(np.mean([r[k] for r in rs])),
                                 "std": float(np.std([r[k] for r in rs]))}
                             for k in ("val_macro_f1", "test_accuracy", "test_macro_f1")}

    out = {"data_dir": data_dir, "classes": classes, "seeds": seeds, "lr_grid": lrs,
           "max_epochs": epochs, "patience": patience, "summary": summary, "runs": runs}
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"data: {data_dir}  seeds: {seeds}  lr grid: {lrs}")
    print(f"{'channels':<10} {'val F1':>14} {'test acc':>14} {'test F1':>14}  selected lr")
    for channels, s in summary.items():
        cells = [f"{100 * s[k]['mean']:6.2f} +/- {100 * s[k]['std']:4.2f}"
                 for k in ("val_macro_f1", "test_accuracy", "test_macro_f1")]
        chosen = [r["selected_lr"] for r in runs if r["channels"] == channels]
        print(f"{channels:<10} " + " ".join(f"{c:>14}" for c in cells) + f"  {chosen}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_synthetic")
    parser.add_argument("--output_json", default="results/cnn_channel_ablation_synthetic.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.01, 0.003])
    args = parser.parse_args()
    main(args.data_dir, args.output_json, args.epochs, args.patience, args.seeds, args.lrs)
