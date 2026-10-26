"""Shared training features plus conservative state handling for live inference."""
from collections import deque
import time

import numpy as np
import torch

from preprocess_jester import FEATURE_VERSION, motion_features
from temporal_models import load_checkpoint, predict_probabilities


class GestureGate:
    def __init__(self, threshold=0.75, confirmations=3):
        self.threshold = threshold
        self.confirmations = confirmations
        self.reset()

    def reset(self):
        self.candidate = None
        self.count = 0
        self.active = None

    def update(self, label, confidence, valid=True):
        if not valid or label == 'background' or confidence < self.threshold:
            self.reset()
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
            self.active = None
        if self.count >= self.confirmations:
            self.active = label
        return self.active


class LivePredictor:
    def __init__(self, checkpoint_path, window_frames=36):
        torch.set_num_threads(4)
        self.model, self.checkpoint = load_checkpoint(checkpoint_path)
        if self.checkpoint.get('data', {}).get('feature_version') != FEATURE_VERSION:
            raise ValueError('Model uses incompatible features. Train on the prepared Jester dataset first.')
        self.classes = self.checkpoint['classes']
        self.window_frames = window_frames
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.window_frames)
        self.detections = deque(maxlen=self.window_frames)
        self.gate = GestureGate()
        self.last_timestamp = None
        self.missing = 0
        self.handedness = None

    def update(self, coordinates, found, timestamp_ms, handedness=None):
        if self.last_timestamp is not None:
            if timestamp_ms <= self.last_timestamp:
                raise ValueError('Frame timestamps must increase')
            if timestamp_ms - self.last_timestamp > 250:
                self.reset()
        if found and self.handedness and handedness and handedness != self.handedness:
            self.reset()
        self.last_timestamp = timestamp_ms
        if found:
            self.handedness = handedness
        self.missing = 0 if found else self.missing + 1
        if self.missing >= 3:
            self.frames.clear()
            self.detections.clear()
            self.gate.reset()
        self.frames.append(np.asarray(coordinates, dtype=np.float32).reshape(21, 3))
        self.detections.append(bool(found))
        response = {'active': None, 'prediction': None, 'confidence': 0.0,
                    'samples': len(self.frames), 'required_samples': self.window_frames,
                    'detection_rate': float(np.mean(self.detections)), 'classifier_ms': 0.0}
        if not found:
            self.gate.reset()
            response['message'] = 'Bring one hand into view'
            return response
        if len(self.frames) < self.window_frames:
            response['message'] = 'Keep moving gently while the window fills'
            return response
        if response['detection_rate'] < 0.5:
            self.gate.reset()
            response['message'] = 'Keep your whole hand visible'
            return response
        started = time.perf_counter()
        try:
            features = motion_features(np.stack(self.frames), self.detections, self.checkpoint['target_frames'])
        except ValueError:
            self.gate.reset()
            response['message'] = 'Move your hand a little closer'
            return response
        probabilities = predict_probabilities(self.model, self.checkpoint, features[None])[0]
        index = int(probabilities.argmax())
        label, score = self.classes[index], float(probabilities[index])
        active = self.gate.update(label, score)
        response.update(active=active, prediction=label, confidence=score,
                        classifier_ms=(time.perf_counter() - started) * 1000)
        response['message'] = ('A gentle sound follows your movement' if active else
                               'Resting quietly' if label == 'background' else 'Waiting for a clearer movement')
        return response
