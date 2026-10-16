"""Live safety gates and parity with the training feature pipeline."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from live_engine import GestureGate, LivePredictor
from preprocess_jester import FEATURE_VERSION, motion_features

class LiveTests(unittest.TestCase):
    def test_requires_consecutive_confident_predictions(self):
        gate = GestureGate()
        self.assertIsNone(gate.update('rolling', .9))
        self.assertIsNone(gate.update('rolling', .9))
        self.assertEqual(gate.update('rolling', .9), 'rolling')
        self.assertIsNone(gate.update('swiping', .9))
        self.assertIsNone(gate.update('swiping', .2))
        self.assertIsNone(gate.update('swiping', .9))
        self.assertIsNone(gate.update('background', .99))
        self.assertIsNone(gate.update('rolling', .99, valid=False))

    def predictor(self):
        checkpoint = {'data':{'feature_version':FEATURE_VERSION}, 'classes':['rolling','background'], 'target_frames':30}
        with patch('live_engine.load_checkpoint', return_value=(None,checkpoint)):
            return LivePredictor('unused')

    def test_training_live_feature_parity_and_missing_hand(self):
        predictor = self.predictor()
        raw = np.random.default_rng(42).normal(size=(38,21,3)).astype(np.float32)
        with patch('live_engine.predict_probabilities', return_value=np.array([[.95,.05]])) as classify:
            for i, frame in enumerate(raw):
                response = predictor.update(frame, True, i*83, 'Left')
            self.assertEqual(response['active'],'rolling')
            np.testing.assert_allclose(classify.call_args.args[2][0],motion_features(raw[-36:], [True]*36,30))
            self.assertIsNone(predictor.update(np.zeros((21,3)),False,38*83)['active'])
            predictor.update(np.zeros((21,3)),False,39*83)
            response = predictor.update(np.zeros((21,3)),False,40*83)
            self.assertEqual(response['samples'],1)

    def test_gap_hand_change_and_out_of_order(self):
        predictor = self.predictor()
        frame = np.ones((21,3))
        predictor.update(frame,True,1,'Left')
        predictor.update(frame,True,84,'Left')
        self.assertEqual(predictor.update(frame,True,500,'Left')['samples'],1)
        self.assertEqual(predictor.update(frame,True,583,'Right')['samples'],1)
        with self.assertRaises(ValueError): predictor.update(frame,True,583,'Right')

if __name__ == '__main__': unittest.main()
