"""Run the four classifiers sequentially on a fixed processed Jester dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from utils import load_training_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed_jester"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/jester_v1"))
    args = parser.parse_args()
    load_training_splits(args.data_dir)  # refuse incomplete data before any run
    args.output_dir.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).resolve().parent
    jobs = [
        ("mlp", [str(src / "train_mlp.py"), "--output_json", str(args.output_dir / "mlp/metrics.json")]),
        ("cnn", [str(src / "train_cnn.py"), "--output_json", str(args.output_dir / "cnn/metrics.json")]),
        ("1dcnn", [str(src / "train_temporal.py"), "--model", "1dcnn", "--output_dir", str(args.output_dir / "1dcnn")]),
        ("gru", [str(src / "train_temporal.py"), "--model", "gru", "--output_dir", str(args.output_dir / "gru")]),
    ]
    metrics = []
    for name, command in jobs:
        print(f"Starting {name} on real Jester data", flush=True)
        subprocess.run([sys.executable, "-u", *command, "--data_dir", str(args.data_dir)], check=True)
        metrics.append(json.loads((args.output_dir / name / "metrics.json").read_text()))
    reference = metrics[0]
    for metric in metrics[1:]:
        if any(metric[key] != reference[key] for key in ("data", "classes", "n_train", "n_val", "n_test")):
            raise ValueError("Cannot compare models with different datasets or splits")
    selected = max(metrics, key=lambda row: (row["val"]["macro_f1"], row["val"]["accuracy"]))
    summary = {
        "data": reference["data"], "classes": reference["classes"],
        "selection_rule": "Highest validation macro-F1, then validation accuracy. Test metrics do not determine selection.",
        "selected_model": selected["model"],
        "models": [{"model": row["model"], "validation_accuracy": row["val"]["accuracy"],
                    "validation_macro_f1": row["val"]["macro_f1"], "test_accuracy": row["test"]["accuracy"],
                    "test_macro_f1": row["test"]["macro_f1"], "latency": row["latency"]} for row in metrics],
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
