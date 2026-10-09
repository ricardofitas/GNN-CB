"""Scientific-integrity and executable-model checks.

Run with ``python -m unittest discover -s tests -v``.  Tests of packaged data
and model behaviour always run.  The independent spreadsheet audit also runs
when original archives are available in ``GNN_CB_SOURCE_DIR``, ``data/raw``,
or a sibling ``Graphs_Analysis_Corrugated_Boards`` checkout.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import pandas as pd
import torch

from gnn_cb.data import ARCHIVES, RestrictedGraphUnpickler, load_dataset
from gnn_cb.model import DeformationModel, ModelConfig, affine_features, load_checkpoint
from gnn_cb.predict import predict
from gnn_cb.train import fit_affine, partition, run, train_one


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def source_directory() -> Path | None:
    candidates = []
    if os.environ.get("GNN_CB_SOURCE_DIR"):
        candidates.append(Path(os.environ["GNN_CB_SOURCE_DIR"]))
    candidates += [ROOT / "data" / "raw", ROOT.parent / "Graphs_Analysis_Corrugated_Boards"]
    for candidate in candidates:
        if all((candidate / item["filename"]).is_file() for item in ARCHIVES.values()):
            return candidate
    return None


class PreparedDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load_dataset(DATA)
        if not cls.samples:
            raise AssertionError("No packaged training data: run scripts/prepare_data.py")

    def test_unique_specimens_and_complete_frame_indices(self):
        ids = [str(sample["specimen_id"]) for sample in self.samples]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(ARCHIVES))
        for sample in self.samples:
            with self.subTest(specimen=str(sample["specimen_id"])):
                np.testing.assert_array_equal(sample["frame"], np.arange(len(sample["frame"])))
                self.assertEqual(sample["y"].shape, (*sample["mask"].shape, 2))
                self.assertEqual(sample["mask"].dtype, np.dtype(bool))
                self.assertTrue(np.isfinite(sample["y"]).all())
                self.assertTrue(sample["mask"][0].all())
                np.testing.assert_array_equal(sample["y"][0], 0)
                self.assertTrue((~sample["mask"]).any(), "Source tracking has missing observations")

    def test_edges_are_symmetric_unique_and_valid(self):
        for sample in self.samples:
            with self.subTest(specimen=str(sample["specimen_id"])):
                edges = sample["edge_index"]
                self.assertEqual(edges.shape[0], 2)
                self.assertTrue(np.issubdtype(edges.dtype, np.integer))
                self.assertGreaterEqual(int(edges.min()), 0)
                self.assertLess(int(edges.max()), len(sample["x0"]))
                pairs = set(map(tuple, edges.T.tolist()))
                self.assertEqual(len(pairs), edges.shape[1])
                self.assertTrue(all(u != v and (v, u) in pairs for u, v in pairs))
                self.assertEqual(set(edges.flatten().tolist()), set(range(len(sample["x0"]))))

    def test_geometry_normalization_preserves_pixel_units(self):
        for sample in self.samples:
            with self.subTest(specimen=str(sample["specimen_id"])):
                x0, scale = sample["x0"], float(sample["scale_px"])
                self.assertGreater(scale, 0)
                np.testing.assert_allclose(x0.min(axis=0), 0, atol=1e-7)
                self.assertAlmostEqual(float(np.ptp(x0[:, 1])), 1.0, places=6)
                # x is scaled by board height too; independent per-axis
                # scaling would distort corrugation aspect ratios.
                ratios = np.ptp(sample["initial_pos_px"], axis=0) / scale
                np.testing.assert_allclose(np.ptp(x0, axis=0), ratios, rtol=1e-6)
                np.testing.assert_allclose(
                    x0 * scale + sample["origin_px"], sample["initial_pos_px"], atol=1e-4)
                np.testing.assert_allclose(
                    sample["y"][sample["mask"]] * scale,
                    sample["displacement_px"][sample["mask"]], atol=1e-5)

    def test_phase_retains_actual_frame_index(self):
        for sample in self.samples:
            with self.subTest(specimen=str(sample["specimen_id"])):
                start, end = sample["compression_window"]
                self.assertAlmostEqual(float(sample["progress"][start]), 0.0)
                self.assertAlmostEqual(float(sample["progress"][end]), 1.0)
                self.assertLess(float(sample["progress"][0]), 0.0)
                np.testing.assert_allclose(
                    np.diff(sample["progress"]), 1 / float(end - start), atol=2e-7)

    def test_original_tracking_rows_exactly_define_mask_and_targets(self):
        source = source_directory()
        if source is None:
            self.skipTest("Original ZIP archives unavailable; set GNN_CB_SOURCE_DIR for source audit")
        for sample in self.samples:
            specimen = str(sample["specimen_id"])
            metadata = json.loads((DATA / f"{specimen}.json").read_text())
            with self.subTest(specimen=specimen), zipfile.ZipFile(
                source / ARCHIVES[specimen]["filename"]
            ) as archive:
                sheets = pd.read_excel(
                    io.BytesIO(archive.read(metadata["source_members"]["tracking"])), sheet_name=None)
                expected_mask = np.zeros_like(sample["mask"])
                expected_mask[0] = True
                internal_holes = 0
                for column, node_id in enumerate(sample["node_ids"]):
                    table = sheets[str(node_id)].sort_values("frame_number")
                    if table.empty:
                        continue
                    observed = table["frame_number"].to_numpy(dtype=int)
                    self.assertEqual(int(observed[0]), 1, "First reference must correspond to frame 0")
                    expected_mask[observed, column] = True
                    first_reference = table.iloc[0][["pre_col", "pre_row"]].to_numpy(dtype=float)
                    np.testing.assert_array_equal(first_reference, sample["initial_pos_px"][column])
                    observed_displacement = table[["new_col", "new_row"]].to_numpy() - first_reference
                    np.testing.assert_allclose(
                        sample["y"][observed, column] * float(sample["scale_px"]),
                        observed_displacement, atol=1e-5)
                    internal_holes += int(observed[-1] - observed[0] + 1 - len(observed))
                self.assertGreater(internal_holes, 0, "Audit must cover genuine tracking gaps")
                np.testing.assert_array_equal(sample["mask"], expected_mask)
                np.testing.assert_array_equal(sample["y"][~expected_mask], 0)

    def test_unapproved_pickle_global_is_rejected(self):
        # The byte stream names an unsafe global but contains no call opcode.
        with self.assertRaises(pickle.UnpicklingError):
            RestrictedGraphUnpickler(io.BytesIO(b"cos\nsystem\n.")).load()


class ModelProperties(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        torch.set_num_threads(1)
        sample = load_dataset(DATA)[0]
        self.x0 = torch.from_numpy(sample["x0"])
        self.edges = torch.from_numpy(sample["edge_index"])
        self.phase = torch.tensor([0.0, 0.31, 0.72, 1.0])
        self.coefficients = torch.randn(9, 2) * 0.01
        self.config = ModelConfig(hidden_dim=16, message_steps=2)
        self.model = DeformationModel(self.config, self.coefficients, residual_scale=0.2)
        # Activate graph-dependent residuals: the deliberate zero-initialized
        # output layer would otherwise make structural tests vacuous.
        torch.nn.init.normal_(self.model.decoder[-1].weight, std=0.08)
        torch.nn.init.normal_(self.model.decoder[-1].bias, std=0.03)
        self.model.eval()

    def test_node_permutation_equivariance_with_active_messages(self):
        permutation = torch.randperm(len(self.x0))
        inverse = torch.argsort(permutation)
        with torch.no_grad():
            original = self.model(self.x0, self.edges, self.phase)
            reordered = self.model(self.x0[permutation], inverse[self.edges], self.phase)
        torch.testing.assert_close(reordered, original[:, permutation], atol=2e-6, rtol=2e-5)

    def test_translation_invariance(self):
        with torch.no_grad():
            original = self.model(self.x0, self.edges, self.phase)
            translated = self.model(self.x0 + torch.tensor([3.25, -4.5]), self.edges, self.phase)
        torch.testing.assert_close(original, translated, atol=2e-6, rtol=2e-5)

    def test_message_topology_matters_and_no_edges_are_finite(self):
        with torch.no_grad():
            connected = self.model(self.x0, self.edges, self.phase)
            disconnected = self.model(self.x0, torch.empty((2, 0), dtype=torch.long), self.phase)
            singleton = self.model(self.x0[:1], torch.empty((2, 0), dtype=torch.long), self.phase)
        self.assertTrue(torch.isfinite(disconnected).all())
        self.assertTrue(torch.isfinite(singleton).all())
        self.assertEqual(tuple(singleton.shape), (len(self.phase), 1, 2))
        self.assertGreater(float((connected - disconnected).abs().max()), 1e-5)

    def test_gradient_reaches_message_encoder_and_decoder(self):
        self.model.train()
        prediction = self.model(self.x0, self.edges, self.phase)
        target = torch.randn_like(prediction)
        torch.square(prediction - target).mean().backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_initial_model_exactly_matches_affine_and_phase_zero_is_not_forced_zero(self):
        model = DeformationModel(self.config, self.coefficients)
        with torch.no_grad():
            prediction = model(self.x0, self.edges, self.phase)
            baseline = affine_features(self.x0, self.phase) @ self.coefficients
        torch.testing.assert_close(prediction, baseline, atol=0, rtol=0)
        self.assertGreater(float(prediction[0].abs().max()), 0.0)

    def test_safe_checkpoint_reload_preserves_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": asdict(self.config), "state_dict": self.model.state_dict()}, path)
            restored, metadata = load_checkpoint(str(path))
            with torch.no_grad():
                expected = self.model(self.x0, self.edges, self.phase)
                actual = restored(self.x0, self.edges, self.phase)
            self.assertEqual(metadata["config"], asdict(self.config))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_inference_requires_no_targets_masks_or_future_frames(self):
        sample = load_dataset(DATA)[0]
        required = ("x0", "edge_index", "initial_pos_px", "node_ids", "scale_px")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, graph = Path(directory) / "model.pt", Path(directory) / "graph.npz"
            torch.save({"config": asdict(self.config), "state_dict": self.model.state_dict()}, checkpoint)
            np.savez(graph, **{key: sample[key] for key in required})
            result = predict(checkpoint, graph, self.phase.tolist())
            with torch.no_grad():
                expected = self.model(self.x0, self.edges, self.phase).numpy()
            np.testing.assert_allclose(result["displacement_normalized"], expected, atol=0, rtol=0)
            np.testing.assert_allclose(
                result["position_px"], sample["initial_pos_px"][None] + expected * sample["scale_px"], atol=1e-5)
            # A user graph can carry unrelated corrupt future labels; none
            # should be read or affect predictions.
            np.savez(graph, **{key: sample[key] for key in required},
                     y=np.full((1, 1, 2), np.nan), mask=np.zeros((1, 1), dtype=bool))
            repeated = predict(checkpoint, graph, self.phase.tolist())
            np.testing.assert_array_equal(repeated["displacement_normalized"], result["displacement_normalized"])

    def test_inference_rejects_nonfinite_scale_and_invalid_graph_indices(self):
        sample = load_dataset(DATA)[0]
        required = ("x0", "edge_index", "initial_pos_px", "node_ids", "scale_px")
        cases = [
            {"scale_px": np.asarray(np.nan)},
            {"scale_px": np.asarray(np.inf)},
            {"edge_index": sample["edge_index"].astype(float) + 0.5},
            {"edge_index": np.asarray([[0], [len(sample["x0"])]])},
            {"node_ids": sample["node_ids"][:-1]},
            {"node_ids": np.zeros_like(sample["node_ids"])},
            {"x0": np.empty((0, 2)), "initial_pos_px": np.empty((0, 2)),
             "edge_index": np.empty((2, 0), dtype=int), "node_ids": np.empty(0, dtype=int)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, graph = Path(directory) / "model.pt", Path(directory) / "graph.npz"
            torch.save({"config": asdict(self.config), "state_dict": self.model.state_dict()}, checkpoint)
            for replacement in cases:
                with self.subTest(invalid_fields=tuple(replacement)):
                    fields = {key: sample[key] for key in required}
                    fields.update(replacement)
                    np.savez(graph, **fields)
                    with self.assertRaises(ValueError):
                        predict(checkpoint, graph, self.phase.tolist())


class TrainingIsolation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.samples = load_dataset(DATA)

    def test_chronological_partitions_are_disjoint_with_real_gap(self):
        for sample in self.samples:
            with self.subTest(specimen=str(sample["specimen_id"])):
                split = partition(sample["progress"])
                train = sample["frame"][split["train"]]
                validation = sample["frame"][split["validation"]]
                gap = sample["frame"][split["gap"]]
                self.assertGreater(len(gap), 0)
                self.assertLess(int(train.max()), int(gap.min()))
                self.assertLess(int(gap.max()), int(validation.min()))
                self.assertFalse((split["train"] & split["validation"]).any())
                np.testing.assert_array_equal(
                    split["train"] | split["validation"] | split["gap"], split["test"])

    def test_masked_nan_targets_cannot_change_fitting_or_gradients(self):
        original = self.samples[0]
        poisoned = {key: value.copy() for key, value in original.items()}
        poisoned["y"][~poisoned["mask"]] = np.nan
        selected = partition(original["progress"])["train"]
        self.assertTrue((~original["mask"][selected]).any())
        torch.testing.assert_close(
            fit_affine(original, selected), fit_affine(poisoned, selected), atol=0, rtol=0)
        first, first_history, first_info = train_one(original, "gnn", 91, 5, 20, 0.002)
        second, second_history, second_info = train_one(poisoned, "gnn", 91, 5, 20, 0.002)
        self.assertEqual(first_history, second_history)
        self.assertEqual(first_info["selected_epoch"], second_info["selected_epoch"])
        best = min(first_history, key=lambda row: row["validation_mse_normalized"])
        self.assertEqual(first_info["selected_epoch"], best["epoch"])
        for key, expected in first.state_dict().items():
            with self.subTest(parameter=key):
                self.assertTrue(torch.isfinite(second.state_dict()[key]).all())
                torch.testing.assert_close(second.state_dict()[key], expected, atol=0, rtol=0)

    def test_heldout_labels_cannot_change_training_or_epoch_selection(self):
        # Run the actual outer benchmark twice. Only the held-out specimen's
        # labels differ in fold one. Any cross-specimen fitting, normalization,
        # or early stopping leak would alter its checkpoint or training history.
        poisoned = [{key: value.copy() for key, value in sample.items()} for sample in self.samples]
        poisoned[1]["y"][poisoned[1]["mask"]] += 500.0
        with tempfile.TemporaryDirectory() as directory:
            summaries = []
            for name, samples in (("original", self.samples), ("poisoned", poisoned)):
                args = argparse.Namespace(
                    data=str(DATA), output=str(Path(directory) / name), threads=1,
                    seeds=[91], epochs=5, patience=20, learning_rate=0.002)
                with mock.patch("gnn_cb.train.load_dataset", return_value=samples), contextlib.redirect_stdout(io.StringIO()):
                    summaries.append(run(args))
            for summary in summaries:
                self.assertEqual(len(summary["folds"]), 2)
                for fold in summary["folds"]:
                    self.assertNotEqual(fold["training_specimen"], fold["held_out_specimen"])
                    self.assertFalse(set(fold["training_frame_ids"]) & set(fold["validation_frame_ids"]))
            first_fold = summaries[0]["folds"][0]
            stem = f"train_{first_fold['training_specimen']}__test_{first_fold['held_out_specimen']}"
            for kind in ("gnn", "mlp"):
                with self.subTest(model=kind):
                    original_path = Path(directory) / "original" / stem / f"{kind}_seed91.pt"
                    poisoned_path = Path(directory) / "poisoned" / stem / f"{kind}_seed91.pt"
                    original_model, original_meta = load_checkpoint(str(original_path))
                    poisoned_model, poisoned_meta = load_checkpoint(str(poisoned_path))
                    self.assertEqual(original_meta["selected_epoch"], poisoned_meta["selected_epoch"])
                    for key, expected in original_model.state_dict().items():
                        torch.testing.assert_close(poisoned_model.state_dict()[key], expected, atol=0, rtol=0)
                    for output in ("original", "poisoned"):
                        saved = np.load(Path(directory) / output / stem / f"{kind}_seed91_predictions.npz")
                        # Reloaded model must reproduce the exported predictions.
                        model = original_model if output == "original" else poisoned_model
                        heldout = self.samples[1]
                        with torch.no_grad():
                            prediction = model(torch.from_numpy(heldout["x0"]),
                                               torch.from_numpy(heldout["edge_index"]),
                                               torch.from_numpy(heldout["progress"]))
                        np.testing.assert_allclose(prediction.numpy(), saved["prediction"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
