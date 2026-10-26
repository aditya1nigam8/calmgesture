# CalmGesture (ICT 4442 Deep Learning mini project)

CalmGesture: a gesture-to-sound sensory regulation tool for autism support. Webcam hand gestures are turned into MediaPipe landmarks, classified, and mapped to predefined calming sounds. This is an exploratory, non-clinical prototype; it makes no diagnostic or therapeutic claims.

Pipeline: MediaPipe landmarks -> normalise -> (30, 63) sequences -> MLP / CNN / 1D-CNN / GRU.

## Status

- Implemented: MLP, NumPy 2D CNN, PyTorch 1D-CNN and GRU, Jester subset selection, frame-folder preprocessing,
  shared split loading, detection-quality logging, and regression checks.
- Real-data preprocessing and the first four-model comparison are complete.
- Local camera-to-sound app implemented with the selected 1D-CNN.
- To do: live webcam/audio testing, final error analysis/report, and a documented
  decision on the interim report's proposed CNN framework/architecture changes.
- The interim files `results/mlp_metrics.json` and `results/cnn_metrics.json`
  contain **synthetic** checks. Real results are separately identified below.
- `src/preprocess.py` has not yet been run on real videos. Test it on a few clips first; it uses the legacy `mp.solutions.hands` API, which newer MediaPipe versions may not include.

## Run the camera-to-sound app

From the project folder, with the downloaded landmark model and trained checkpoint available:

```bash
source .venv/bin/activate
pip install -r requirements-app.txt
python src/app.py
```

Open http://localhost:8765 in your browser, press **Start exploring**, and allow
camera access. Keep one whole hand visible and repeat a gesture for about three
seconds. Use the volume slider or mute; **End session** releases the camera.
Switching away from the tab also stops the session. Stop the server with Ctrl+C.
On this Mac, MediaPipe requires normal desktop graphics access.

A fresh clone does not include `data/models/hand_landmarker.task` or
`runs/jester_v1/1dcnn/model.pt`. Complete the data/preprocessing/training setup
below first; model files and datasets are intentionally excluded from Git.

The browser sends JPEG frames only to the local Python server. Frames are not
saved. The server extracts landmarks and applies the same motion features used
for training. A rolling 36-frame window at a target 12 fps is resampled to 30
frames. Three consecutive predictions scoring at least 0.75 activate a tone;
background, uncertainty, and missing hands silence it. These provisional
thresholds need live validation. The browser synthesizes four soft sine tones;
these are not clinically validated calming sounds.

Automated checks cover feature parity, prediction stability, tracking loss,
hand changes, and timestamp gaps. Saved-clip replay checks server integration;
it is not a webcam accuracy evaluation. Initial replay of clips 340 and 887
repeatedly reset the window before classification. Repeating one visible-hand
frame 48 times reached inference, but incorrectly activated shaking (a stationary
hand false positive). Motion rejection and tracking tolerance need improvement
before this can be described as reliable. Browser audio remains unverified.
Processing time shown in the interface
excludes the movement collection window, browser transport, and audio playback.

## First real-data experiment (7 October 2026)

All 3,000 selected Jester clips were downloaded (107,026 frames). The training-only
quality filter retained 1,292/2,000 training clips. All 500 internal-validation
and 500 held-out clips were retained, including tracking failures.

| Model | Validation macro-F1 | Held-out accuracy | Held-out macro-F1 |
|---|---:|---:|---:|
| MLP | 75.20% | 76.60% | 76.80% |
| NumPy spatial CNN | 75.97% | 74.00% | 74.60% |
| 1D-CNN | **76.98%** | **79.80%** | **79.92%** |
| GRU | 72.67% | 76.00% | 75.84% |

1D-CNN is the candidate for integration, chosen by validation macro-F1 before
using test metrics for reporting. This is a single seeded five-class experiment,
not an official 27-class Jester benchmark or a live-webcam evaluation. Swiping is
the weakest class: only 96/400 swiping training clips passed the tracking filter;
the selected model correctly classified 55/100 held-out swiping clips.

See `results/jester_comparison.json` and `results/jester_data_quality.json`.
Detailed metrics, confusion matrices, predictions, and reloadable models are in
`runs/jester_v1/`. The selected weights are `runs/jester_v1/1dcnn/model.pt`.
Single-clip classifier latency is recorded separately from camera/landmark/audio
latency; the complete live system has not yet been timed.

## Setup

Baseline training (from the project folder):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the temporal models and the complete test suite:

```bash
python -m pip install -r requirements-training.txt
```

Jester preprocessing uses the current MediaPipe Tasks API with a pinned package:

```bash
python -m pip install -r requirements-video.txt
mkdir -p data/models
curl -L --fail https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o data/models/hand_landmarker.task
```

`requirements-lock.txt` records the exact installed environment. The older
`src/preprocess.py` video entry point still uses the legacy API; use
`src/preprocess_jester.py` for this project's selected dataset.

## Selected Jester dataset

Only these eight official labels are selected, grouped into five outputs:

| Output class | Official Jester labels | Intended interaction |
|---|---|---|
| `rolling` | Rolling Hand Forward; Rolling Hand Backward | Repetitive rolling movement |
| `shaking` | Shaking Hand | Repetitive hand shaking |
| `tapping` | Drumming Fingers | Finger drumming/tapping |
| `swiping` | Swiping Left; Swiping Right | Broad lateral movement |
| `background` | No gesture; Doing other things | No sound trigger |

Swiping is not relabeled as waving; Jester also does not provide the custom
open/close class from the original proposal. Those classes are excluded.
Direction variants share an output because direction is not needed for the
planned sound mapping. This is a five-class subset experiment, not the full
27-class Jester benchmark.

`config/jester_subset.json` controls the reproducible selection (seed 42).
Each output has 400 training, 100 internal validation, and 100 held-out clips:
3,000 total before landmark quality filtering. Constituent source labels are
balanced within each output. Training and internal validation come only from
official training clips; held-out testing uses a subset of official validation.
Participant independence is not guaranteed by this protocol.

1. Obtain the official annotation package from the
   [download page](https://www.qualcomm.com/developer/software/jester-dataset/downloads).
   Place `train.csv`, `validation.csv`, and `labels.csv` under
   `data/jester/annotations/`. These files are already present in this checkout.
2. Generate the fixed clip manifest (already generated in this checkout):

```bash
python src/prepare_jester.py --train_csv data/jester/annotations/train.csv --validation_csv data/jester/annotations/validation.csv
```

The selector refuses to overwrite an existing experiment; use `--output` for a
new selection. The companion summary records counts, configuration, and input
file hashes. It never silently substitutes classes or shrinks the requested set.

3. Download and selectively extract the selected clips:

```bash
python -u src/download_jester_subset.py
```

This reads the official three-part TGZ stream and saves only selected JPGs. It
still transfers about 23 GB over the network, but does not store the full archive.
Use a fresh output directory, or add `--resume` after an interrupted run. Resume
verifies existing frames byte-for-byte; the compressed network stream must still
restart from the beginning. Short network interruptions are retried in place.
A failed run leaves partial output for inspection; `extraction.json` reports
whether extraction completed. For an existing local tar/tar.gz,
use `src/extract_jester_subset.py --manifest data/jester/subset.csv --archive PATH
--output data/jester/frames` instead. Raw split TGZ parts can be concatenated in
order and piped to that command with `--archive -`.

4. Validate frames, check a small real-data pilot, then preprocess:

```bash
python src/preprocess_jester.py --manifest data/jester/subset.csv --frames_dir data/jester/frames --check_only
python src/check_jester_pilot.py
python -u src/preprocess_jester.py --manifest data/jester/subset.csv --frames_dir data/jester/frames
```

The new feature representation keeps `(30, 63)` but preserves hand travel by
subtracting the first valid wrist XY position and dividing by the clip's median
palm scale. X and Z are corrected for image aspect ratio. Missing detections
hold the last valid pose; initial gaps use the first valid pose. A background
clip with no detected hand becomes zeros. Foreground training clips below 50%
detected frames are excluded; every decision is recorded in `quality.jsonl`. This initial
threshold needs inspection on real clips. Exclusions can change class balance.
No-hand behavior must also be explicitly handled in the eventual live app.

MediaPipe 0.10.32 was verified on ten real training clips on this Mac; all produced
the expected `(30, 63)` arrays. Two of the eight foreground clips had detection
rates below the initial threshold. This tiny sample verifies execution, not
recognition performance. See `results/jester_training_preprocessing_pilot.json`.
The installed 1.0.1 release crashed in its native graph; 0.10.32 required normal
macOS graphics access even with the CPU delegate selected.

Quality filtering applies only to foreground training clips. All validation and
test clips remain, including tracking failures, so reported accuracy does not
silently exclude difficult examples. The first extraction feasibility probe
included ten official-validation clips; no model was trained or scored in that
probe. Subsequent pilot checks use training clips only.

The output includes `dataset.json`, feature version, model/manifest hashes,
accepted clip IDs and fixed split membership. Incomplete datasets are rejected
by the training loader. Outputs must go to a fresh directory to avoid stale data.

5. Train the existing models on those fixed real-data splits:

```bash
python src/train_mlp.py --data_dir data/processed_jester --output_json results/mlp_jester_metrics.json
python src/train_cnn.py --data_dir data/processed_jester --output_json results/cnn_jester_metrics.json
python src/train_temporal.py --model 1dcnn --data_dir data/processed_jester --output_dir runs/jester_1dcnn
python src/train_temporal.py --model gru --data_dir data/processed_jester --output_dir runs/jester_gru
```

Or run all four sequentially and produce a validation-selected comparison:

```bash
python src/train_all.py --output_dir runs/jester_v2
```

Use a fresh output directory for each experiment. MLP uses the shared validation
set for early stopping (rather than reserving another portion of training data).
The NumPy CNN also restores the epoch with the lowest validation loss. Its loss
gradient now averages over each batch exactly once, verified with finite
differences; interim synthetic CNN numbers predate this correction.
Both baselines save `.joblib` checkpoints alongside their metrics, containing
weights and preprocessing settings. Load only locally generated/trusted joblib
files. The temporal models save `model.pt` checkpoints.

## Temporal models

The 1D-CNN applies two temporal convolutions (63 -> 64 -> 64 channels), global
average pooling, dropout, and a classifier. The GRU uses 64 hidden units and
classifies from the final hidden state. Both consume full landmark sequences,
preserving temporal ordering instead of reducing inputs to mean/std features.

The shared trainer fits coordinate normalization on training clips only, uses
Adam and gradient clipping, and selects weights by minimum validation loss with
early stopping. It runs on CPU with a fixed seed. Each fresh experiment directory
contains `model.pt` (weights, scaler, labels, feature provenance), `metrics.json`,
and `predictions.npz`. Evaluation uses the saved-and-reloaded checkpoint.
Single-clip latency reports exclude webcam capture, landmark extraction, window
collection, and audio. Do not directly compare these latency figures with the
interim NumPy CNN's batch-throughput estimate. Current runs use batch-size-one
CPU measurements with warmup for all four models, including their respective
feature aggregation/normalization and probability output.

These implementations support the model assignments in the synopsis; they do
not constitute evidence of individual member contributions. Team members should
review and understand their assigned models before recording contribution claims.

## Synthetic pipeline checks

Pipeline check only (synthetic landmarks, not real gesture data; writes to `data/processed_synthetic` and `results/`):

```
python src/synth_smoketest_data.py
python src/train_mlp.py
python src/train_cnn.py
```

To verify the foundation without replacing the interim report's saved results:

```bash
python -m unittest discover -s tests -v
python src/synth_smoketest_data.py
python src/train_mlp.py --output_json results/smoke_check/mlp_metrics.json
python src/train_cnn.py --output_json results/smoke_check/cnn_metrics.json
python src/train_temporal.py --model 1dcnn --data_dir data/processed_synthetic --output_dir runs/synthetic_1dcnn --epochs 20
python src/train_temporal.py --model gru --data_dir data/processed_synthetic --output_dir runs/synthetic_gru --epochs 20
```

Temporal experiments refuse to overwrite existing directories; choose a new
`--output_dir` when repeating a run. Synthetic results are checks of execution,
not real gesture recognition evidence.

The CNN now fits normalization on training clips only. Historical CNN metrics
predate this correction and must not be used for the final comparison.
Synthetic data retains the historical random clip-stratified split. Real Jester
data uses the fixed manifest instead; the trainers do not reshuffle it.

## Completion milestones

1. Verify the baseline training environment and fix evaluation leakage.
2. Obtain a selected 20BN-Jester subset and its official labels; validate extraction,
   detection quality, and preservation of wrist movement. Custom recordings are
   excluded from the updated scope.
3. Train MLP, CNN, 1D-CNN, and GRU on a shared real-data evaluation protocol.
4. Compare accuracy, macro-F1, confusion matrices, and latency; save the selected
   model and preprocessing settings for inference.
5. Build the webcam-to-sound interface with stable predictions and audio controls.
6. Test across users and conditions, document errors, and complete the final report.

Dataset decision: use Jester only. Its official distribution contains numbered
folders of JPG frames sampled at 12 FPS. The website describes its test set as
unlabeled, but the official label package retrieved on 6 October 2026 includes
`test-answers.csv`. We do not use that file in this experiment; our declared
held-out set comes from official validation. Do not claim participant-disjoint
testing without IDs, or custom-webcam validation without performing it.
See the [official dataset](https://www.qualcomm.com/developer/software/jester-dataset)
and [downloads](https://www.qualcomm.com/developer/software/jester-dataset/downloads).
The synopsis's custom-dataset objective will need to be marked as a scope change
in the final report.

## Layout

| Path | Purpose |
|---|---|
| `config/jester_subset.json` | Relevant gesture mapping and selection limits |
| `src/prepare_jester.py` | Official annotations to a fixed subset manifest |
| `src/download_jester_subset.py`, `src/extract_jester_subset.py` | Selective streaming extraction |
| `src/preprocess_jester.py`, `src/check_jester_pilot.py` | Tasks API extraction and real-frame feasibility check |
| `src/preprocess.py` | Video -> MediaPipe landmarks -> wrist-relative, scale-normalised (30, 63) `.npy` |
| `src/utils.py` | Loading, MLP features, stratified 70/10/20 split (seed 42), metrics |
| `src/train_mlp.py` | MLP (sklearn) |
| `src/cnn_numpy.py`, `src/train_cnn.py` | From-scratch NumPy 2D CNN |
| `src/temporal_models.py`, `src/train_temporal.py` | PyTorch 1D-CNN/GRU, checkpoint export and evaluation |
| `src/baseline_inference.py`, `src/train_all.py` | Baseline model loading and sequential four-model comparison |
| `src/synth_smoketest_data.py` | Synthetic clips for pipeline checks only |
| `results/` | Metrics JSON |
| `build_report.js` | Generates the interim report `.docx` (`npm install docx`) |

All models share one split and one metric set (accuracy, macro-F1, confusion matrix).
