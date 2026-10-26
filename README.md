# CalmGesture (ICT 4442 Deep Learning mini project)

CalmGesture: a gesture-to-sound sensory regulation tool for autism support. Webcam hand gestures are turned into MediaPipe landmarks, classified, and mapped to predefined calming sounds. This is an exploratory, non-clinical prototype; it makes no diagnostic or therapeutic claims.

Pipeline: MediaPipe landmarks -> normalise -> (30, 63) sequences -> MLP / CNN (1D-CNN and GRU to come).

## Status

- Done: preprocessing code, MLP, NumPy 2D CNN, shared split and metrics.
- To do: 1D-CNN, GRU, model comparison, real-time app.
- All metrics currently in `results/` come from **synthetic data**. They only show the code runs and must not be reported as project results.
- `src/preprocess.py` has not yet been run on real videos. Test it on a few clips first; it uses the legacy `mp.solutions.hands` API, which newer MediaPipe versions may not include.

## Setup

```
pip install mediapipe opencv-python-headless scikit-learn numpy
```

## Usage

Real data (`data/raw_videos/<class>/*.mp4` -> `data/processed/<class>/*.npy`):

```
python src/preprocess.py --input_dir data/raw_videos --output_dir data/processed
python src/train_mlp.py --data_dir data/processed --output_json results/mlp_real_metrics.json
python src/train_cnn.py --data_dir data/processed --output_json results/cnn_real_metrics.json
python src/cnn_channel_ablation.py --data_dir data/processed --output_json results/cnn_channel_ablation_real.json --seeds 0 1 2
```

NumPy CNN checks (finite-difference gradients, checkpoint round-trip):

```
python -m pytest tests     # or: python tests/test_cnn_gradients.py
```

Pipeline check only (synthetic landmarks, not real gesture data; writes to `data/processed_synthetic` and `results/`):

```
python src/synth_smoketest_data.py
python src/train_mlp.py
python src/train_cnn.py
```

## Layout

| Path | Purpose |
|---|---|
| `src/preprocess.py` | Video -> MediaPipe landmarks -> wrist-relative, scale-normalised (30, 63) `.npy` |
| `src/utils.py` | Loading, MLP features, stratified 70/10/20 split (seed 42), metrics |
| `src/train_mlp.py` | MLP (sklearn) |
| `src/cnn_numpy.py`, `src/train_cnn.py` | From-scratch NumPy 2D CNN (train-only normalisation, early stopping on val loss, `.npz` checkpoint in `models/`) |
| `src/cnn_channel_ablation.py` | CNN input-channel ablation: mean vs std vs mean+std, LR chosen on validation only |
| `tests/test_cnn_gradients.py` | Finite-difference gradient checks for the NumPy CNN |
| `src/synth_smoketest_data.py` | Synthetic clips for pipeline checks only |
| `results/` | Metrics JSON |
| `build_report.js` | Generates the interim report `.docx` (`npm install docx`) |

All models share one split and one metric set (accuracy, macro-F1, confusion matrix).
