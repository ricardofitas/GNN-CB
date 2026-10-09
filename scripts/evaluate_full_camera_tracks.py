"""Frozen five-specimen model versus all eligible automatic camera labels.

No fitting, checkpoint selection, phase re-scaling or frame subsampling occurs.
Scores measure agreement with automated image pseudo-targets, not independent
physical displacement ground truth. Probe1_8 keeps its full phase denominator
while withholding the unqualified late-collapse observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gnn_cb.model import affine_features, load_checkpoint
from gnn_cb.train import metrics

SPECIMENS = {"Probe1_8": {"window": [136, 1474], "last_eligible_frame": 1246},
             "Probe1_9": {"window": [1006, 2033], "last_eligible_frame": 2033}}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/expanded/five_specimens/final/model.pt"))
    parser.add_argument("--tracks-dir", type=Path, default=Path("data/automatic/tracks"))
    parser.add_argument("--curation-audit", type=Path, default=Path("reports/automatic_image_curation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/expanded/automatic_external"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 256:
        parser.error("--batch-size must be between 1 and 256")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = digest(args.checkpoint)
    model, metadata = load_checkpoint(str(args.checkpoint))
    trained = set(metadata["training_specimens"]) | set(metadata["training_physical_specimens"])
    source_audit = json.loads(args.curation_audit.read_text())
    admissions = {row["specimen_id"]: row for row in source_audit["specimens"]}
    rows = []
    for specimen, specification in SPECIMENS.items():
        admission = admissions[specimen]
        if not admission["training_admission"] or admission["compression_window"] != specification["window"]:
            raise ValueError(f"Source quality admission changed for {specimen}; review protocol before evaluating")
        path = args.tracks_dir / f"{specimen}.npz"
        with np.load(path, allow_pickle=False) as archive:
            geometry = {key: archive[key].copy() for key in ["x0", "edge_index", "initial_pos_px", "node_ids", "scale_px"]}
            physical = str(archive["physical_specimen_id"])
            original_frame = archive["frame"].copy()
            # Targets are read only for scoring; no model or geometry feature uses them.
            targets = archive["y"].copy()
            source_mask = archive["mask"].copy()
            frame_valid = archive["frame_valid"].copy()
        if specimen in trained or physical in trained:
            raise ValueError("This checkpoint already fitted an evaluation specimen")
        if not np.array_equal(original_frame, np.arange(2048)):
            raise ValueError("Expected the complete, unsampled 2,048-frame camera track")
        start, end = specification["window"]
        selection = (original_frame >= start) & (original_frame <= end)
        frame = original_frame[selection]
        phase = ((frame - start) / float(end - start)).astype(np.float32)
        coverage = source_mask.mean(axis=1)
        qualified_frame = (frame_valid & (coverage >= 0.65)
                           & (original_frame <= specification["last_eligible_frame"]))
        mask = source_mask[selection] & qualified_frame[selection, None]
        target = targets[selection]
        if not np.isfinite(target[mask]).all():
            raise ValueError("Nonfinite qualified pseudo-target")
        x0 = torch.as_tensor(geometry["x0"], dtype=torch.float32)
        edge_index = torch.as_tensor(geometry["edge_index"], dtype=torch.long)
        chunks, affine_chunks = [], []
        with torch.no_grad():
            for offset in range(0, len(phase), args.batch_size):
                phase_tensor = torch.from_numpy(phase[offset:offset + args.batch_size])
                chunks.append(model(x0, edge_index, phase_tensor).numpy())
                affine_chunks.append((affine_features(x0, phase_tensor) @ model.affine_coefficients).numpy())
        prediction = np.concatenate(chunks, axis=0)
        affine_prediction = np.concatenate(affine_chunks, axis=0)
        if not np.isfinite(prediction).all() or prediction.shape != target.shape:
            raise ValueError("Invalid model prediction")
        scale = float(geometry["scale_px"])
        values = metrics(prediction, target, mask, scale)
        baselines = {"zero_displacement": metrics(np.zeros_like(prediction), target, mask, scale),
                     "checkpoint_affine_only": metrics(affine_prediction, target, mask, scale)}
        supported = mask.any(axis=1)
        framewise = []
        for i in np.flatnonzero(supported):
            framewise.append({"frame": int(frame[i]), "phase": float(phase[i]),
                              **metrics(prediction[i:i + 1], target[i:i + 1], mask[i:i + 1], scale)})
        output = args.output / f"{specimen}_full_loading_predictions.npz"
        np.savez_compressed(output, prediction=prediction,
                            checkpoint_affine_coefficients=model.affine_coefficients.detach().cpu().numpy(),
                            target=target, mask=mask,
                            frame=frame, progress=phase, physical_specimen_id=np.asarray(physical),
                            node_ids=geometry["node_ids"], x0=geometry["x0"],
                            initial_pos_px=geometry["initial_pos_px"], scale_px=geometry["scale_px"],
                            edge_index=geometry["edge_index"], compression_window=np.asarray([start, end]))
        rows.append({
            "specimen_id": specimen, "physical_specimen_id": physical,
            "source_npz_sha256": digest(path), "original_camera_frames": len(original_frame),
            "loading_window": [start, end], "loading_window_frames": len(frame),
            "supervised_loading_frames": int(supported.sum()), "observed_node_frames": int(mask.sum()),
            "nodes": len(x0), "scale_px": scale,
            "qualified_original_frame_range": [int(frame[supported][0]), int(frame[supported][-1])],
            "qualified_phase_range": [float(phase[supported][0]), float(phase[supported][-1])],
            "phase_definition": f"(original_video_frame - {start}) / ({end} - {start}); never renormalized to eligible subset",
            "mask_policy": "original unambiguous node observation AND frame_valid AND at least65% original-node coverage AND within prequalified original-frame cutoff",
            "last_eligible_original_frame": specification["last_eligible_frame"],
            "admission_scope": admission["training_admission_scope"],
            "pseudo_target_agreement": values, "baselines": baselines,
            "per_frame_pseudo_target_agreement": framewise,
            "prediction_file": output.name, "prediction_file_bytes": output.stat().st_size,
            "prediction_sha256": digest(output),
        })
    if digest(args.checkpoint) != checkpoint_hash:
        raise AssertionError("Checkpoint changed during evaluation")
    summary = {
        "evaluation": "frozen five-specimen checkpoint agreement with all eligible unsampled automatic image labels",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_training_specimens": metadata["training_specimens"],
        "checkpoint_training_physical_specimens": metadata["training_physical_specimens"],
        "fitting_performed": False, "selection_performed": False, "checkpoint_unchanged": True,
        "evaluation_frame_stride": 1, "inference_batch_size": args.batch_size,
        "source_admission_audit": str(args.curation_audit),
        "specimens": rows,
        "totals": {"physical_specimens": len(rows),
                   "predicted_loading_frames": sum(row["loading_window_frames"] for row in rows),
                   "supervised_loading_frames": sum(row["supervised_loading_frames"] for row in rows),
                   "observed_node_frames": sum(row["observed_node_frames"] for row in rows)},
        "macro_pseudo_target_agreement": {
            "rmse_normalized": float(np.mean([row["pseudo_target_agreement"]["rmse_normalized"] for row in rows])),
            "rmse_px": float(np.mean([row["pseudo_target_agreement"]["rmse_px"] for row in rows])),
            "weighting": "equal weight per physical specimen; each specimen aggregates all qualified coordinate observations"},
        "macro_baselines": {kind: {
            "rmse_normalized": float(np.mean([row["baselines"][kind]["rmse_normalized"] for row in rows])),
            "rmse_px": float(np.mean([row["baselines"][kind]["rmse_px"] for row in rows]))}
            for kind in ["zero_displacement", "checkpoint_affine_only"]},
        "baseline_provenance": "Zero displacement and unchanged affine_coefficients buffer from the same frozen five-specimen checkpoint. No baseline fitting on camera data.",
        "interpretation": "External to checkpoint fitting, but targets and phase are retrospective automated image outputs from the same recordings. This measures agreement with pseudo-targets, not independently calibrated physical accuracy. Frames/nodes are correlated and do not create extra independent specimens. Probe1_8 complete collapse remains unsupported; Probe1_2/10 are excluded because their full loading endpoints were not observed.",
        "comparison_to_augmentation": "This evaluation uses only the frozen final five-specimen model. Separately augmented models may fit these camera recordings and must not reuse this score as their held-out accuracy.",
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"totals": summary["totals"], "macro_pseudo_target_agreement": summary["macro_pseudo_target_agreement"],
                      "macro_baselines": summary["macro_baselines"],
                      "specimens": [{"specimen": row["specimen_id"], "frames": row["supervised_loading_frames"],
                                     "observations": row["observed_node_frames"], "metrics": row["pseudo_target_agreement"],
                                     "baselines": row["baselines"]} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
