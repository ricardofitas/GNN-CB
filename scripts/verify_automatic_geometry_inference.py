"""Run a frozen model on initial graphs with physically absent target arrays.

Only x0, edge_index, initial_pos_px, node_ids and scale_px may be read. A temporary
geometry-only NPZ is constructed using those keys, then predictions are compared
bit-for-bit with the same frozen model reading the original NPZ under an access
guard. This checks executable inference, not predictive accuracy on the videos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from gnn_cb.predict import predict

GEOMETRY_KEYS = ("x0", "edge_index", "initial_pos_px", "node_ids", "scale_px")
PHASES = [0.0, 0.25, 0.5, 0.75, 1.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_one(source: Path, checkpoint: Path, geometry_dir: Path, output_dir: Path) -> dict:
    real_load = np.load
    reads = []

    class GeometryOnlyArchive:
        def __init__(self, path, *args, **kwargs):
            self.path = str(path)
            self.archive = real_load(path, *args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            if key not in GEOMETRY_KEYS:
                raise AssertionError(f"Inference attempted to read non-geometry field: {key}")
            reads.append({"filename": Path(self.path).name, "key": key})
            return self.archive[key]

    with GeometryOnlyArchive(source, allow_pickle=False) as archive:
        initial_geometry = {key: archive[key].copy() for key in GEOMETRY_KEYS}
    geometry_path = geometry_dir / f"{source.stem}_initial_geometry_only.npz"
    np.savez_compressed(geometry_path, **initial_geometry)
    with real_load(geometry_path, allow_pickle=False) as archive:
        if set(archive.files) != set(GEOMETRY_KEYS):
            raise AssertionError("Geometry-only archive contains unexpected fields")
    with patch("gnn_cb.predict.np.load", GeometryOnlyArchive):
        geometry_result = predict(checkpoint, geometry_path, PHASES)
        full_result = predict(checkpoint, source, PHASES)
    for key in geometry_result:
        if not np.array_equal(geometry_result[key], full_result[key]):
            raise AssertionError(f"Geometry-only inference differs for result {key}")
    prediction = geometry_result["displacement_normalized"]
    if prediction.shape != (len(PHASES), len(initial_geometry["x0"]), 2):
        raise AssertionError("Unexpected prediction dimensions")
    for key in ("displacement_normalized", "displacement_px", "position_px"):
        if not np.isfinite(geometry_result[key]).all():
            raise AssertionError(f"Nonfinite prediction in {key}")
    target = output_dir / f"{source.stem}_geometry_only_predictions.npz"
    np.savez_compressed(target, **geometry_result,
                        source_recording=np.asarray(source.stem),
                        input_semantics=np.asarray("initial geometry plus user-specified hypothetical phase"))
    return {
        "recording": source.stem, "source_npz_sha256": sha256(source),
        "geometry_only_npz_sha256": sha256(geometry_path),
        "prediction_file": target.name, "prediction_sha256": sha256(target),
        "nodes": len(initial_geometry["x0"]),
        "directed_edges": int(initial_geometry["edge_index"].shape[1]),
        "undirected_edges": len({tuple(sorted(pair)) for pair in initial_geometry["edge_index"].T}),
        "scale_px": float(initial_geometry["scale_px"]), "requested_phases": PHASES,
        "predicted_node_phase_pairs": int(prediction.shape[0] * prediction.shape[1]),
        "all_predictions_finite": True, "full_npz_and_geometry_only_npz_bitwise_equivalent": True,
        "allowed_input_fields": list(GEOMETRY_KEYS), "observed_array_reads": reads,
        "targets_masks_observed_positions_and_recorded_phase_read": False,
        "geometry_archive_contains_targets": False,
        "accuracy_claim": None,
        "interpretation": ("Geometric domain stress test only; no validated compression endpoint and no accuracy claim."
                           if source.stem in {"Probe1_2", "Probe1_10"} else
                           "Geometric domain stress test over the full requested phase range; partial pre-collapse supervision is qualified separately, but complete-collapse tracking coverage is insufficient. No accuracy claim here."
                           if source.stem == "Probe1_8" else
                           "Functional inference on an unseen initial graph; phases are requested, not measured video endpoints. Accuracy requires a separate admissible evaluation."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/expanded/five_specimens/final/model.pt"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("data/automatic/tracks"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/expanded/automatic_geometry_inference"))
    parser.add_argument("--specimens", nargs="+", default=["Probe1_2", "Probe1_8", "Probe1_9", "Probe1_10"])
    args = parser.parse_args()
    for specimen in args.specimens:
        if not (args.tracks_dir / f"{specimen}.npz").is_file():
            raise FileNotFoundError(f"Initial graph not ready: {specimen}")
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnn_initial_geometry_") as temporary:
        rows = [verify_one(args.tracks_dir / f"{specimen}.npz", args.checkpoint,
                           Path(temporary), args.output_dir) for specimen in args.specimens]
    audit = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
             "verification_type": "initial-geometry-only executable inference; not accuracy evaluation",
             "phases": PHASES, "recordings": rows,
             "total_initial_nodes": sum(row["nodes"] for row in rows),
             "total_predicted_node_phase_pairs": sum(row["predicted_node_phase_pairs"] for row in rows),
             "target_access_guard": "Reading any field other than the five initial-geometry fields raises an error.",
             "phase_semantics": "Hypothetical protocol phase supplied by caller. No current/future image, tracked displacement, observation mask, compression window, or source progress array is used. Phase 1 is not claimed to be a true video endpoint; phase 0 is not necessarily camera frame0."}
    (args.output_dir / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"recordings": len(rows), "total_initial_nodes": audit["total_initial_nodes"],
                      "node_phase_predictions": audit["total_predicted_node_phase_pairs"],
                      "all_geometry_only_equivalence_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
