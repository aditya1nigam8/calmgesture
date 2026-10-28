"""Controlled GRU ablations with validation-only selection across fixed seeds.

The held-out test is scored only for the selected configuration, after the
validation selection has been written to disk. Synthetic runs are checks only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from temporal_models import load_checkpoint, predict_probabilities
from train_temporal import train
from utils import dataset_provenance, evaluate, load_training_splits


CONFIGURATIONS = [
    {"name": "gru_baseline", "architecture": "gru", "model_config": {}},
    {"name": "gru_mean_state", "architecture": "gru", "model_config": {"pooling": "mean"}},
    {"name": "gru_hidden32", "architecture": "gru", "model_config": {"hidden_size": 32}},
    {"name": "gru_lr_small", "architecture": "gru", "model_config": {}, "learning_rate": 0.0003},
    {"name": "1dcnn_reference", "architecture": "1dcnn", "model_config": {}},
]


def aggregate(rows):
    """Variation over initialization seeds, not a confidence interval."""
    result = {}
    for key in ("accuracy", "macro_f1"):
        values = np.array([row[key] for row in rows])
        result[key] = {"mean": float(values.mean()),
                       "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None}
    return result


def select_configuration(rows):
    # Deliberately accepts only validation aggregates, never test results.
    return max(rows, key=lambda row: (row["validation"]["macro_f1"]["mean"],
                                     row["validation"]["accuracy"]["mean"]))["name"]


def sensitivity_probes(directory, validation):
    """Unlabeled perturbations: sensitivity only, not live recognition accuracy."""
    model, checkpoint = load_checkpoint(directory / "model.pt")
    original = predict_probabilities(model, checkpoint, validation).argmax(1)
    probes = {}
    for label, values in {
        "reversed": validation[:, ::-1].copy(),
        "static_first_frame": np.repeat(validation[:, :1], validation.shape[1], axis=1),
    }.items():
        predictions = predict_probabilities(model, checkpoint, values).argmax(1)
        probes[label] = {"prediction_change_rate": float(np.mean(predictions != original))}
    (directory / "validation_sensitivity.json").write_text(json.dumps(probes, indent=2) + "\n")


def run(data_dir, output_dir, seeds=(42, 43, 44), epochs=50, patience=10,
        configurations=None, allow_synthetic=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct seeds")
    provenance = dataset_provenance(data_dir)
    if provenance["dataset"] != "20BN-Jester v1" and not allow_synthetic:
        raise ValueError("Real Jester data required; --allow_synthetic is for pipeline checks only")
    _, validation, test, _, _, ytest, classes = load_training_splits(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    configs = CONFIGURATIONS if configurations is None else configurations
    protocol = {"data": provenance, "seeds": list(seeds), "epochs": epochs,
                "patience": patience, "configurations": configs,
                "batch_size": 32, "default_learning_rate": 0.001,
                "optimizer": "Adam", "loss": "cross_entropy", "gradient_clip_norm": 1.0,
                "checkpoint_selection": "minimum validation loss",
                "selection": "Mean validation macro-F1 across seeds, then mean validation accuracy; first listed breaks ties",
                "scope": "isolated clips; not continuous-input or clinical evaluation"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = []
    for config in configs:
        metrics = []
        for seed in seeds:
            directory = output / config["name"] / f"seed_{seed}"
            metric = train(config["architecture"], data_dir, directory, epochs=epochs,
                           patience=patience, seed=seed, model_config=config["model_config"],
                           learning_rate=config.get("learning_rate", 0.001), evaluate_test=False)
            if metric["data"] != provenance or metric["classes"] != classes:
                raise ValueError("Dataset metadata changed during the experiment")
            metrics.append(metric)
            sensitivity_probes(directory, validation)
        summaries.append({"name": config["name"],
                          "validation": aggregate([m["val"] for m in metrics]),
                          "parameters": metrics[0]["parameters"],
                          "best_epochs": [m["best_epoch"] for m in metrics]})
    selected = select_configuration(summaries)
    selection = {**protocol, "models": summaries, "selected_configuration": selected}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    # Selection is frozen before test predictions. Do not select a best seed.
    test_rows = []
    names = np.asarray(classes)
    for seed in seeds:
        directory = output / selected / f"seed_{seed}"
        model, checkpoint = load_checkpoint(directory / "model.pt")
        probabilities = predict_probabilities(model, checkpoint, test)
        predictions = names[probabilities.argmax(1)]
        test_rows.append({"seed": seed, **evaluate(ytest, predictions, classes)})
        np.savez_compressed(directory / "selected_test_predictions.npz", y_true=ytest,
                            y_pred=predictions, probabilities=probabilities, classes=names)
    result = {"selected_configuration": selected, "data": provenance,
              "per_seed": test_rows, "test": aggregate(test_rows),
              "note": "Seed standard deviation is not uncertainty over users or clips; sensitivity probes are not live validation."}
    (output / "selected_test.json").write_text(json.dumps(result, indent=2) + "\n")
    return selection, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/processed_jester")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--allow_synthetic", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
