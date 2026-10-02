"""
preprocess.py
CalmGesture — Dataset Acquisition & Preprocessing Pipeline
=============================================================

Extracts per-frame hand-landmark features from real video clips (Jester
subset clips, or custom webcam recordings) using MediaPipe Hands, then
normalizes and windows them into fixed-length sequences ready for the
MLP / CNN / 1D-CNN / GRU models.

USAGE (on real data, once video files are available):

    python preprocess.py --input_dir data/raw_videos --output_dir data/processed

Expected input layout:

    data/raw_videos/
        rolling/
            clip001.mp4
            clip002.mp4
            ...
        shaking/
            ...
        waving/
        tapping/
        open_close/

Each sub-folder name is treated as the class label. The script writes one
.npy file per clip to <output_dir>/<label>/<clip_name>.npy containing an
array of shape (T, 63) — T frames x 21 landmarks x (x, y, z).

NOTE: This script is fully functional and is the pipeline the team will
run on the actual Jester subset and the custom-recorded webcam clips.
It has been smoke-tested on short self-recorded clips during development.
It has NOT yet been run end-to-end on the full Jester subset or the full
custom dataset as of this interim submission — see Section 5 (Risk/Plan)
of the interim report.
"""

import argparse
import os
import numpy as np
import cv2
import mediapipe as mp

N_LANDMARKS = 21
N_COORDS = 3  # x, y, z


def extract_landmarks_from_video(video_path, max_frames=60, min_detection_confidence=0.5):
    """Run MediaPipe Hands over a video file and return an (T, 63) array.

    Frames where no hand is detected are filled by repeating the last
    valid detection (or zeros if no hand has been seen yet), so that
    every clip yields a fixed-rate sequence rather than ragged output.
    """
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        last_valid = np.zeros(N_LANDMARKS * N_COORDS, dtype=np.float32)
        while cap.isOpened() and len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                coords = np.array(
                    [[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32
                ).flatten()
                last_valid = coords
            frames.append(last_valid.copy())
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((1, N_LANDMARKS * N_COORDS))


def normalize_sequence(seq):
    """Wrist-relative, scale-normalized landmark sequence.

    Subtracts the wrist (landmark 0) position per frame and divides by
    the distance between wrist and middle-finger MCP (landmark 9), so
    the representation is invariant to hand position and distance from
    the camera.
    """
    seq = seq.reshape(seq.shape[0], N_LANDMARKS, N_COORDS)
    wrist = seq[:, 0:1, :]
    rel = seq - wrist
    scale = np.linalg.norm(rel[:, 9, :2], axis=-1, keepdims=True) + 1e-6
    norm = rel / scale[:, None, :]
    return norm.reshape(seq.shape[0], -1)


def resample_to_fixed_length(seq, target_len=30):
    """Linearly resample a (T, 63) sequence to (target_len, 63)."""
    t_old = np.linspace(0, 1, seq.shape[0])
    t_new = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(t_new, t_old, seq[:, d])
    return out


def process_directory(input_dir, output_dir, target_len=30):
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    )
    manifest = []
    for label in classes:
        src_dir = os.path.join(input_dir, label)
        dst_dir = os.path.join(output_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue
            raw = extract_landmarks_from_video(os.path.join(src_dir, fname))
            norm = normalize_sequence(raw)
            fixed = resample_to_fixed_length(norm, target_len)
            out_path = os.path.join(dst_dir, fname.rsplit(".", 1)[0] + ".npy")
            np.save(out_path, fixed)
            manifest.append((out_path, label))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw_videos")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--target_len", type=int, default=30)
    args = parser.parse_args()
    m = process_directory(args.input_dir, args.output_dir, args.target_len)
    print(f"Processed {len(m)} clips into {args.output_dir}")
