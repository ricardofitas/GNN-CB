"""Integrity tests for group-separated evaluation and balanced fitting."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from gnn_cb.model import load_checkpoint
from gnn_cb.train_expanded import (add_training_augmentation, build_protocol, compression_selection,
                                   conservative_continuity_mask,
                                   evaluate_expanded, fit_affine_balanced,
                                   run, specimen_weights, train_selected,
                                   validate_admission, validate_dataset)


def synthetic_specimen(name: str, slope: float, frames: int = 7) -> dict:
    x0 = np.asarray([[0, 0], [1, 0.5], [2, 1]], dtype=np.float32)
    p = np.linspace(0, 1, frames, dtype=np.float32)
    y = np.zeros((frames, 3, 2), dtype=np.float32)
    y[:, :, 1] = slope * p[:, None] * x0[None, :, 1]
    return {"specimen_id": np.asarray(name), "physical_specimen_id": np.asarray(name),
            "x0": x0, "y": y, "mask": np.ones((frames, 3), dtype=bool),
            "progress": p, "frame": np.arange(frames), "node_ids": np.arange(3),
            "edge_index": np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]]),
            "initial_pos_px": x0 * 100, "scale_px": np.asarray(100.0)}


def arguments(output: str) -> argparse.Namespace:
    return argparse.Namespace(data=["unused"], output=output, seeds=[11], epochs=5,
                              patience=10, learning_rate=0.002, threads=1,
                              protocol_only=False, skip_final=True, specimens=None)


class ExpandedTrainingIntegrity(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.samples = [synthetic_specimen(f"S{i}", 0.05 * i) for i in range(5)]

    def test_folds_keep_physical_replicates_together(self):
        replica = copy.deepcopy(self.samples[0])
        replica["specimen_id"] = np.asarray("S0_retracked")
        protocol = build_protocol(self.samples + [replica], arguments("unused"))
        self.assertEqual(protocol["independent_physical_specimens"], 5)
        self.assertEqual(protocol["dataset_extractions"], 6)
        test_groups = []
        for fold in protocol["folds"]:
            train = set(fold["training_physical_specimens"])
            validation = set(fold["validation_physical_specimens"])
            test = {fold["held_out_physical_specimen"]}
            self.assertFalse(train & validation or train & test or validation & test)
            self.assertEqual(len(train | validation | test), 5)
            self.assertEqual(len(train), 3)
            test_groups.extend(test)
        self.assertEqual(set(test_groups), {f"S{i}" for i in range(5)})

    def test_replica_count_does_not_change_group_weight(self):
        replica = copy.deepcopy(self.samples[0])
        replica["specimen_id"] = np.asarray("S0_retracked")
        weights = specimen_weights(self.samples + [replica])
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(weights[0] + weights[-1], 0.2)
        first = fit_affine_balanced(self.samples)
        second = fit_affine_balanced(self.samples + [replica])
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)

    def test_training_augmentation_preserves_withheld_physical_groups(self):
        protocol = build_protocol(self.samples, arguments("unused"))
        repeated_test = copy.deepcopy(self.samples[0])
        repeated_test["specimen_id"] = np.asarray("S0_video_reextraction")
        extra = [repeated_test, synthetic_specimen("NewVideo", 0.7)]
        for data in extra:
            data["_metadata"] = {"training_admitted": True}
        augmented = add_training_augmentation(protocol, self.samples, extra, copy.deepcopy(protocol))
        self.assertEqual(augmented["evaluation_physical_specimens"], 5)
        self.assertEqual(augmented["independent_physical_specimens"], 6)
        for original, changed in zip(protocol["folds"], augmented["folds"]):
            self.assertEqual(original["held_out_physical_specimen"], changed["held_out_physical_specimen"])
            self.assertEqual(original["validation_physical_specimens"], changed["validation_physical_specimens"])
            withheld = {changed["held_out_physical_specimen"], *changed["validation_physical_specimens"]}
            self.assertFalse(withheld & set(changed["training_physical_specimens"]))
            self.assertIn("NewVideo", changed["augmentation_extractions"])
            if "S0" in withheld:
                self.assertNotIn("S0_video_reextraction", changed["augmentation_extractions"])
            else:
                self.assertIn("S0_video_reextraction", changed["augmentation_extractions"])

    def test_duplicate_frame_sampling_does_not_overweight_one_specimen(self):
        original = [synthetic_specimen("A", 0.1), synthetic_specimen("B", 0.4)]
        repeated = copy.deepcopy(original)
        for key in ("y", "mask", "progress"):
            repeated[0][key] = np.repeat(repeated[0][key], 3, axis=0)
        repeated[0]["frame"] = np.arange(len(repeated[0]["progress"]))
        torch.testing.assert_close(fit_affine_balanced(original), fit_affine_balanced(repeated),
                                   atol=1e-6, rtol=1e-6)

    def test_masked_values_cannot_change_fitting_or_epoch_selection(self):
        original = copy.deepcopy(self.samples)
        original[0]["mask"][2:, 0] = False
        poisoned = copy.deepcopy(original)
        poisoned[0]["y"][~poisoned[0]["mask"]] = np.nan
        validate_dataset(poisoned[0])
        first, history1, info1 = train_selected(original[:3], original[3:4], "gnn", 11, 5, 10, 0.002)
        second, history2, info2 = train_selected(poisoned[:3], poisoned[3:4], "gnn", 11, 5, 10, 0.002)
        self.assertEqual(history1, history2)
        self.assertEqual(info1["selected_epoch"], info2["selected_epoch"])
        for name, expected in first.state_dict().items():
            torch.testing.assert_close(second.state_dict()[name], expected, atol=0, rtol=0)

    def test_training_validation_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            train_selected(self.samples[:2], self.samples[1:3], "gnn", 11, 0, 5, 0.002)

    def test_quarantined_metadata_cannot_be_admitted(self):
        validate_admission({"admissibility": "admitted_with_masked_automated_tracking_labels"}, "A")
        for metadata in ({"admissibility": "quarantined_low_coverage"},
                         {"quality_status": "failed"}, {"training_admitted": False}):
            with self.assertRaises(ValueError):
                validate_admission(metadata, "A")

    def test_no_complete_tracks_or_empty_phase_subsets_are_reported_without_nan(self):
        sample = copy.deepcopy(self.samples[0])
        sample["mask"][0] = False
        sample["progress"] = np.linspace(0.81, 1, len(sample["frame"]), dtype=np.float32)
        evaluation = evaluate_expanded(sample, sample["y"])
        self.assertEqual(evaluation["complete_track_subset"]["nodes"], 0)
        self.assertIsNone(evaluation["complete_track_subset"]["rmse_normalized"])
        self.assertIsNone(evaluation["early_phase"]["rmse_px"])
        self.assertEqual(evaluation["compression_window"]["rmse_px"], 0)

    def test_continuity_sensitivity_never_revives_long_gap_tracks(self):
        sample = synthetic_specimen("A", 0.1, frames=12)
        sample["mask"][2:4, 0] = False  # Two missing frames are permitted.
        sample["mask"][6:9, 0] = False  # Three missing frames truncate forever.
        sensitive = conservative_continuity_mask(sample)
        self.assertTrue(sensitive[4:6, 0].all())
        self.assertFalse(sensitive[6:, 0].any())
        np.testing.assert_array_equal(sensitive[:, 1:], sample["mask"][:, 1:])
        self.assertTrue(sample["mask"][9:, 0].all(), "Sensitivity must not modify source masks")

    def test_heldout_labels_do_not_affect_fold_checkpoints(self):
        # Run actual orchestration twice. S0 labels change, while its own
        # fold must retain identical fit, selection and saved predictions.
        poisoned = copy.deepcopy(self.samples)
        poisoned[0]["y"] += 77
        with tempfile.TemporaryDirectory() as directory:
            for label, datasets in (("original", self.samples), ("poisoned", poisoned)):
                args = arguments(str(Path(directory) / label))
                with mock.patch("gnn_cb.train_expanded.load_expanded", return_value=datasets), \
                        contextlib.redirect_stdout(io.StringIO()):
                    run(args)
            for kind in ("gnn", "mlp"):
                original = Path(directory) / "original" / "test_S0"
                changed = Path(directory) / "poisoned" / "test_S0"
                m1, metadata1 = load_checkpoint(str(original / f"{kind}_seed11.pt"))
                m2, metadata2 = load_checkpoint(str(changed / f"{kind}_seed11.pt"))
                self.assertEqual(metadata1["selected_epoch"], metadata2["selected_epoch"])
                for name, expected in m1.state_dict().items():
                    torch.testing.assert_close(m2.state_dict()[name], expected, atol=0, rtol=0)
                p1 = np.load(original / f"{kind}_seed11_S0_predictions.npz")["prediction"]
                p2 = np.load(changed / f"{kind}_seed11_S0_predictions.npz")["prediction"]
                np.testing.assert_array_equal(p1, p2)


if __name__ == "__main__":
    unittest.main()
