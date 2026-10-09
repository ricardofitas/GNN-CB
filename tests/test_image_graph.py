"""Topology and tracking integrity tests for image-derived supervision."""
import tempfile
import gzip
import json
from pathlib import Path
import unittest

import cv2
import networkx as nx
import numpy as np

from gnn_cb.image_graph import ExtractionConfig, curate_sequence, extract_graph, gated_assignment, skeleton_graph, track_graphs


class ImageGraphTests(unittest.TestCase):
    def test_chain_pixels_contract_to_one_edge(self):
        skeleton = np.zeros((80, 100), dtype=bool)
        skeleton[40, 10:90] = True
        graph, _ = skeleton_graph(skeleton, ExtractionConfig())
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertGreater(next(iter(graph.edges(data=True)))[2]["length_px"], 70)

    def test_junction_is_structural_and_not_every_skeleton_pixel(self):
        skeleton = np.zeros((80, 100), dtype=np.uint8)
        cv2.line(skeleton, (10, 40), (90, 40), 1)
        cv2.line(skeleton, (50, 10), (50, 40), 1)
        graph, _ = skeleton_graph(skeleton.astype(bool), ExtractionConfig())
        self.assertEqual(sorted(dict(graph.degree()).values()), [1, 1, 1, 3])
        self.assertEqual(graph.number_of_edges(), 3)

    def test_gating_leaves_outliers_unmatched(self):
        ref = np.array([[0, 0], [20, 0], [40, 0]])
        obs = np.array([[1, 0], [21, 0], [100, 100]])
        matches, distance, ambiguous = gated_assignment(ref, obs, 5)
        np.testing.assert_array_equal(matches, [0, 1, -1])
        self.assertTrue(np.isinf(distance[2]))

    def test_ambiguous_identity_withheld(self):
        matches, _, ambiguous = gated_assignment(np.array([[0, 0]]), np.array([[-1, 0], [1, 0]]), 5)
        self.assertEqual(matches[0], -1)
        self.assertTrue(ambiguous[0])

    def test_observed_node_not_duplicated_across_tracks(self):
        matches, _, _ = gated_assignment(np.array([[0, 0], [3, 0]]), np.array([[0.5, 0]]), 5, ambiguity_margin=0)
        self.assertEqual(int(np.sum(matches >= 0)), 1)

    def test_disappearing_track_stays_masked_without_forward_fill(self):
        def g(nodes):
            return {"nodes_xy": np.array(nodes, dtype=float), "degree": np.ones(len(nodes)), "metadata": {"valid_geometry": True}}
        graphs = [g([[0, 0], [20, 0], [40, 0]]), g([[1, 0], [21, 0]]), g([[2, 0], [22, 0]])]
        tracks = track_graphs(graphs)
        self.assertFalse(tracks["mask"][1:, 2].any())
        np.testing.assert_array_equal(tracks["displacement_px"][1:, 2], 0)
        self.assertEqual(tracks["matched_node_indices"][1, 2], -1)

    def test_known_translation_is_recovered_in_image_coordinates(self):
        nodes = np.array([[0, 10], [30, 0], [60, 10]], dtype=float)
        graphs = [{"nodes_xy": nodes + [2 * i, i], "degree": np.ones(3),
                   "metadata": {"valid_geometry": True}} for i in range(4)]
        tracks = track_graphs(graphs)
        self.assertTrue(tracks["mask"].all())
        np.testing.assert_array_equal(tracks["displacement_px"][-1], np.tile([6, 3], (3, 1)))

    def test_bright_plate_component_removed_but_connected_liners_retained(self):
        image = np.zeros((130, 240), dtype=np.uint8)
        cv2.line(image, (5, 10), (230, 10), 180, 3)
        cv2.line(image, (5, 50), (230, 50), 180, 3)
        cv2.line(image, (5, 100), (230, 100), 180, 3)
        for x in range(20, 210, 40):
            cv2.line(image, (x, 50), (x + 20, 100), 180, 3)
            cv2.line(image, (x + 20, 100), (x + 40, 50), 180, 3)
        graph = extract_graph(image, ExtractionConfig(median_kernel=3))
        self.assertGreater(len(graph["nodes_xy"]), 8)
        self.assertGreater(float(graph["nodes_xy"][:, 1].min()), 40)
        self.assertTrue(graph["metadata"]["valid_geometry"])

    def test_portable_sequence_retains_every_graph_and_observed_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for frame in range(3):
                image = np.zeros((100, 220), dtype=np.uint8)
                offset = 2 * frame
                cv2.line(image, (20 + offset, 25), (185 + offset, 25), 180, 3)
                cv2.line(image, (20 + offset, 75), (185 + offset, 75), 180, 3)
                for x in range(25, 170, 40):
                    cv2.line(image, (x + offset, 25), (x + 20 + offset, 75), 180, 3)
                    cv2.line(image, (x + 20 + offset, 75), (x + 40 + offset, 25), 180, 3)
                path = root / f"frame_{frame:04d}.png"
                cv2.imwrite(str(path), image)
                paths.append(path)
            result = curate_sequence(paths, root / "out", "synthetic", compression_window=(0, 2), overlay_count=0)
            self.assertEqual(result["n_frames"], 3)
            with gzip.open(root / "out" / result["graph_archive"], "rt") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual([r["sequence_index"] for r in records], [0, 1, 2])
            self.assertEqual(len({r["sha256"] for r in records}), 3)
            with np.load(root / "out" / "synthetic.npz") as data:
                np.testing.assert_array_equal(data["progress"], [0, 0.5, 1])
                self.assertTrue(data["mask"].all())
                np.testing.assert_allclose(data["displacement_px"][2], np.tile([4, 0], (len(data["node_ids"]), 1)), atol=0.01)


if __name__ == "__main__":
    unittest.main()
