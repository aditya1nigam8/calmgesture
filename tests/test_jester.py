"""Checks for subset integrity, selective extraction and motion features."""
import csv
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from download_jester_subset import ChunkReader
from extract_jester_subset import extract_subset
from prepare_jester import select_rows, read_annotations
from preprocess_jester import motion_features, frame_paths, accept_clip
from utils import load_training_splits


class JesterTests(unittest.TestCase):
    def test_quality_filter_keeps_difficult_evaluation_examples(self):
        self.assertFalse(accept_clip({"split": "train", "label": "rolling"}, 0.1, 0.5))
        self.assertTrue(accept_clip({"split": "train", "label": "background"}, 0, 0.5))
        for split in ["validation", "test"]:
            self.assertTrue(accept_clip({"split": split, "label": "rolling"}, 0, 0.5))

    def setUp(self):
        self.config = {"seed": 42, "validation_fraction": 0.2,
                       "train_pool_per_class": 10, "test_per_class": 4,
                       "classes": {"rolling": ["Forward", "Backward"], "background": ["Nothing"]}}
        self.train = [{"clip_id": str(i * 20 + j), "source_label": name}
                      for i, name in enumerate(["Forward", "Backward", "Nothing"])
                      for j in range(20)]
        self.validation = [dict(row, clip_id=str(int(row["clip_id"]) + 1000)) for row in self.train]

    def test_subset_is_reproducible_balanced_and_disjoint(self):
        rows = select_rows(self.train, self.validation, self.config)
        self.assertEqual(rows, select_rows(self.train[::-1], self.validation[::-1], self.config))
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({r["clip_id"] for r in rows}), 28)
        for label in self.config["classes"]:
            self.assertEqual(sum(r["label"] == label and r["split"] == "train" for r in rows), 8)
            self.assertEqual(sum(r["label"] == label and r["split"] == "validation" for r in rows), 2)
            self.assertEqual(sum(r["label"] == label and r["split"] == "test" for r in rows), 4)
        self.assertTrue(all(r["source_split"] == "validation" for r in rows if r["split"] == "test"))

    def test_overlap_and_insufficient_classes_fail(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_rows(self.train, self.train, self.config)
        with self.assertRaisesRegex(ValueError, "need"):
            select_rows(self.train[:2], self.validation, self.config)

    def test_annotation_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train.csv"
            path.write_text("../escape;Forward\n")
            with self.assertRaises(ValueError):
                read_annotations(path)

    def test_only_selected_safe_jpgs_are_extracted(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name in ["20bn/12/00001.jpg", "20bn/13/00001.jpg", "../../12/00002.jpg", "20bn/12/note.txt"]:
                info = tarfile.TarInfo(name)
                info.size = 3
                archive.addfile(info, io.BytesIO(b"jpg"))
            link = tarfile.TarInfo("20bn/12/00003.jpg")
            link.type = tarfile.SYMTYPE
            link.linkname = "/tmp/escape"
            archive.addfile(link)
        stream.seek(0)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "frames"
            counts = extract_subset(stream, [{"clip_id": "12"}], output)
            self.assertEqual(counts, {"12": 1})
            self.assertEqual([p.relative_to(output).as_posix() for p in output.rglob("*.jpg")], ["12/00001.jpg"])
            self.assertEqual(json.loads((output / "extraction.json").read_text())["status"], "complete")
            stream.seek(0)
            self.assertEqual(extract_subset(stream, [{"clip_id": "12"}], output, resume=True), counts)
            (output / "12/00001.jpg").write_bytes(b"bad")
            stream.seek(0)
            with self.assertRaisesRegex(ValueError, "differs"):
                extract_subset(stream, [{"clip_id": "12"}], output, resume=True)

    def test_chunk_reader_preserves_split_archive_bytes(self):
        with io.BufferedReader(ChunkReader([b"abc", b"defg", b"h"])) as stream:
            self.assertEqual(stream.read(5), b"abcde")
            self.assertEqual(stream.read(), b"fgh")

    def test_features_preserve_motion_and_remove_initial_position_and_scale(self):
        raw = np.zeros((4, 21, 3), dtype=np.float32)
        raw[:, 9, 1] = 0.2
        raw[:, :, 0] += np.arange(4)[:, None] * 0.1
        detected = np.ones(4, dtype=bool)
        features = motion_features(raw, detected, 4)
        np.testing.assert_allclose(features[:, 0], [0, 0.5, 1, 1.5], atol=1e-6)
        shifted = raw * 2
        shifted[:, :, :2] += [0.3, 0.4]
        np.testing.assert_allclose(features, motion_features(shifted, detected, 4), atol=1e-6)

    def test_missing_detections_are_filled_without_artificial_initial_motion(self):
        raw = np.zeros((4, 21, 3), dtype=np.float32)
        raw[1:, :, 0] = 0.4
        raw[1:, 9, 1] = 0.2
        features = motion_features(raw, [False, True, False, True], 4)
        np.testing.assert_allclose(features[0], features[1])
        np.testing.assert_allclose(features[1], features[2])
        np.testing.assert_array_equal(motion_features(raw, [False] * 4), np.zeros((30, 63)))

    def test_frames_sorted_numerically_and_empty_folder_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                frame_paths(folder)
            for name in ["10.jpg", "2.jpg", "1.jpg"]:
                (Path(folder) / name).touch()
            self.assertEqual([p.name for p in frame_paths(folder)], ["1.jpg", "2.jpg", "10.jpg"])

    def test_loader_keeps_declared_splits_and_rejects_duplicate_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clips = []
            for index, split in enumerate(["train", "validation", "test"]):
                path = root / f"{index}.npy"
                np.save(path, np.full((30, 63), index, dtype=np.float32))
                clips.append({"clip_id": str(index), "label": "rolling", "split": split,
                              "source_split": "validation" if split == "test" else "train", "path": path.name})
            metadata = {"status": "complete", "classes": ["rolling"], "target_frames": 30, "clips": clips}
            path = root / "dataset.json"
            path.write_text(json.dumps(metadata))
            parts = load_training_splits(root)
            for index, part in enumerate(parts[:3]):
                self.assertTrue(np.all(part == index))
            clips[2]["clip_id"] = "0"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_training_splits(root)


if __name__ == "__main__":
    unittest.main()
