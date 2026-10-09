#!/usr/bin/env python
"""Audit camera curation, annotate image-only loading phases, and export samples.

Run after curate_images.py, with the decoded PNG folders under --images.
Only complete loading/rebound recordings with sufficient correspondence are
eligible for this *exploratory* phase-conditioned augmentation. Endpoint and
initial graph contact sheets must also be reviewed before model use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter


SPECIMENS = {"Probe1_2": (118, 192), "Probe1_8": (115, 189),
             "Probe1_9": (150, 190), "Probe1_10": (149, 189)}


def profile_height(paths, initial_top, initial_bottom):
    rows = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        profile = image[:, 40:-40].mean(axis=1)
        low, high = initial_top - 15, initial_bottom - 10
        band = profile[low:high]
        threshold = max(20.0, 0.4 * float(band.max()))
        hits = np.flatnonzero(np.convolve((band > threshold).astype(float), np.ones(3), "valid") == 3)
        top = float(low + hits[0]) if len(hits) else float("nan")
        bottom = float(initial_bottom - 8 + np.argmax(profile[initial_bottom - 8:initial_bottom + 9]))
        rows.append([top, bottom, bottom - top])
    return np.asarray(rows)


def loading_annotation(height):
    smooth = median_filter(height, size=21, mode="nearest")
    baseline = float(np.median(smooth[:100]))
    sustained = np.convolve((smooth <= baseline - 2).astype(int), np.ones(50, dtype=int), "valid")
    candidates = np.flatnonzero(sustained == 50)
    start = int(candidates[0]) if len(candidates) else None
    minimum = float(np.min(smooth))
    plateau = np.flatnonzero(smooth <= minimum + 0.5)
    end = int(plateau[-1])
    rebound = float(np.median(smooth[-10:]) - minimum)
    endpoint_observed = rebound >= 2 and end < len(smooth) - 5
    return {"start": start, "end": end if endpoint_observed else None,
            "last_minimum_height_frame": end, "minimum_height_plateau": [int(plateau[0]), end],
            "initial_height_proxy_px": baseline, "minimum_height_proxy_px": minimum,
            "rebound_px": rebound, "right_censored": not endpoint_observed,
            "method": "Median row intensity over central width; top bright-material front requires three consecutive rows above max(20,40% band maximum), lower liner is local row-intensity maximum. Median-filter21frames. Start=first50frame run at least2px below initial100frame median height; end=last minimum-height plateau frame before>=2px visible rebound. Initial row search bands and endpoint contact sheets visually reviewed.",
            "limitations": "Retrospective image-only phase convention, not load, strain, physical timestamp calibration, or original author annotation. Onset carries pixel/threshold uncertainty; max-compression plateau end is defined operationally."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/automatic"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--only", nargs="*", choices=list(SPECIMENS))
    args = parser.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    figures = args.reports / "figures"
    figures.mkdir(exist_ok=True)
    audit = json.loads((args.reports / "video_source_audit.json").read_text())
    source = {"repository": audit["source_repository"], "commit": audit["source_commit"],
              "visibility": "private", "source_audit": "reports/video_source_audit.json"}
    result = {"schema_version": 1, "source": source, "specimens": [],
              "admission_policy": "Exploratory augmentation only: observed full loading/rebound defines phase, connected initial graph, visual initial/mid/endpoint review. Keep only observations from frames with>=65% initial-node coverage; partial loading-window supervision is allowed with true full-window phase retained. Every10th camera frame plus exact endpoints; an endpoint below coverage gate carries an all-false target mask. All versions grouped by physical specimen. Original5specimen test/validation sets must remain fixed."}
    chosen = args.only or list(SPECIMENS)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 1.2 * len(chosen)), squeeze=False)
    curvefig, curveaxes = plt.subplots(len(chosen), 2, figsize=(12, 2.65 * len(chosen)), squeeze=False)
    for row, sp in enumerate(chosen):
        metadata_path = args.data / f"{sp}.json"
        meta = json.loads(metadata_path.read_text())
        physical = "ML2023_" + sp
        meta["lineage_id"] = physical
        meta["source"] = {**meta["source"], **source}
        with np.load(args.data / "tracks" / f"{sp}.npz", allow_pickle=False) as archive:
            tracks = {k: archive[k] for k in archive.files}
        tracks["lineage_id"] = np.asarray(physical)
        tracks["physical_specimen_id"] = np.asarray(physical)
        np.savez_compressed(args.data / "tracks" / f"{sp}.npz", **tracks)
        paths = sorted((args.images / sp).glob("image*.png"))
        bounds = SPECIMENS[sp]
        trace = profile_height(paths, *bounds)
        annotation = loading_annotation(trace[:, 2])
        (args.data / "height_profiles").mkdir(exist_ok=True)
        np.savez_compressed(args.data / "height_profiles" / f"{sp}.npz", top_px=trace[:, 0], bottom_px=trace[:, 1], height_proxy_px=trace[:, 2])
        coverage = tracks["mask"].mean(axis=1)
        connected = meta["frames"][0]["quality"]["n_components"] == 1
        has_window = annotation["start"] is not None and annotation["end"] is not None
        start, end = (annotation["start"], annotation["end"]) if has_window else (None, None)
        window_coverage = coverage[start:end + 1] if has_window else np.empty(0)
        usable_window = np.flatnonzero(window_coverage >= 0.65) + start if has_window else np.empty(0, dtype=int)
        admitted = has_window and connected and len(usable_window) >= 20
        partial = admitted and not bool(np.all(window_coverage >= 0.65))
        scope = "partial_loading_window" if partial else "full_loading_window" if admitted else "withheld"
        reason = ("Connected initial material and observed full loading/rebound define phase. Only pre-collapse frames meeting65% coverage are supervised; all lower-coverage frames and ambiguous identities masked. Full-collapse behavior remains unsupported; phase is not renormalized to the truncated supervision interval." if partial else
                  "Complete image-observed loading/rebound; connected initial material; every loading frame meets65% coverage; visual review completed." if admitted else
                  "Final loading endpoint not observed; right-censored recording retained as graphs/tracks only." if annotation["right_censored"] else
                  "Image-observed loading exists but some loading frames lose more than35% of initial-node identities; retained for geometry/stress inspection only.")
        sampled_count = 0
        if admitted:
            selected = np.unique(np.r_[np.arange(start, end + 1, 10), start, end]).astype(int)
            sampled_count = len(selected)
            framekeys = {"y", "mask", "frame", "positions_px", "displacement_px", "matched_node_indices", "matching_distance_px", "ambiguous", "frame_valid"}
            sample = {key: (value[selected] if key in framekeys else value) for key, value in tracks.items()}
            sample["mask"] = sample["mask"] & (coverage[selected] >= 0.65)[:, None]
            sample["progress"] = ((selected - start) / (end - start)).astype(np.float32)
            sample["compression_window"] = np.asarray([start, end], dtype=np.int64)
            sample["annotation_source"] = np.asarray("retrospective_image_front_loading_window")
            (args.data / "training").mkdir(exist_ok=True)
            np.savez_compressed(args.data / "training" / f"{sp}.npz", **sample)
            video = next(item for item in audit["videos"] if item["specimen"] == sp)
            training_metadata = {
                "specimen_id": sp, "physical_specimen_id": physical, "lineage_id": physical,
                "admissibility": "admitted_partial_loading" if partial else "admitted_full_loading",
                "approved_for_training": True, "training_admitted": True,
                "admission_scope": scope, "admission_reason": reason,
                "source": source, "parent_video_sha256": video["sha256"],
                "parent_decoded_pixel_sha256": video["decoded_bgr_uint8_pixel_sha256"],
                "parent_frame_count": len(paths), "compression_window": [start, end],
                "phase_definition": "(original_video_frame - loading_start)/(loading_end-loading_start); phase not renormalized after masking",
                "annotation": annotation, "sampling": "Every10th raw loading-window frame plus exact endpoints; original frame indices retained.",
                "n_frames": len(selected), "supervised_frames": int(sample["mask"].any(axis=1).sum()),
                "observed_node_frames": int(sample["mask"].sum()),
                "mask_policy": "Original unambiguous gated node matches AND at least65% initial-node coverage for the frame. Endpoint below gate remains all-false; no imputation.",
                "split_policy": "Exploratory fit-only augmentation of fixed original5specimen held-out benchmark; physical lineage cannot overlap test/validation.",
            }
            (args.data / "training" / f"{sp}.json").write_text(json.dumps(training_metadata, indent=2) + "\n")
        meta["loading_annotation"] = annotation
        meta["training_admission"] = admitted
        meta["training_admission_scope"] = scope
        meta["training_admission_reason"] = reason
        meta["sampled_training_frames"] = sampled_count
        meta["compression_window"] = [start, end] if has_window else None
        meta["training_npz_written"] = admitted
        metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
        selectedframes = [0, 1023, 2047]
        for col, frame in enumerate(selectedframes):
            available = sorted((args.data / "overlays").glob(f"{sp}_*.png"))
            nearest = min(available, key=lambda p: abs(int(p.stem.rsplit("_", 1)[1]) - frame))
            image = cv2.cvtColor(cv2.imread(str(nearest)), cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(image[90:225])
            axes[row, col].set_title(f"{sp} · frame {int(nearest.stem.rsplit('_', 1)[1])}", fontsize=11)
            axes[row, col].axis("off")
        curveaxes[row, 0].plot(np.arange(len(trace)), trace[:, 2], color="#234e78", linewidth=1)
        curveaxes[row, 0].set_ylabel(sp + "\nHeight proxy (px)")
        curveaxes[row, 1].plot(np.arange(len(coverage)), coverage, color="#3d8361", linewidth=1)
        curveaxes[row, 1].axhline(0.65, color="#b54a3a", linestyle="--", linewidth=1)
        curveaxes[row, 1].set_ylim(0, 1.04)
        curveaxes[row, 1].set_ylabel("Observed initial-node fraction")
        if has_window:
            for ax in curveaxes[row]:
                ax.axvspan(start, end, color="#e4bd63", alpha=0.2)
        for ax in curveaxes[row]:
            ax.set_xlabel("Original video frame")
            ax.grid(alpha=0.2)
        summary = {"specimen_id": sp, "physical_specimen_id": physical, "n_frames": meta["n_frames"],
                   "n_initial_nodes": meta["n_initial_nodes"], "n_initial_edges": meta["n_initial_edges"],
                   "observed_node_frames": meta["observed_node_frames"], "mean_tracking_coverage": float(coverage.mean()),
                   "minimum_tracking_coverage": float(coverage.min()), "valid_frames": int(tracks["frame_valid"].sum()),
                   "initial_graph_connected": connected, "loading_annotation": annotation,
                   "compression_window": [start, end] if has_window else None, "right_censored": annotation["right_censored"],
                   "loading_window_mean_coverage": float(window_coverage.mean()) if has_window else None,
                   "loading_window_minimum_coverage": float(window_coverage.min()) if has_window else None,
                   "training_admission": admitted, "training_admission_reason": reason,
                   "training_admission_scope": scope,
                   "usable_loading_frames": int(len(usable_window)),
                   "usable_phase_extent": [float((usable_window[0] - start) / (end - start)), float((usable_window[-1] - start) / (end - start))] if len(usable_window) else None,
                   "sampled_training_frames": sampled_count,
                   "sampled_supervised_frames": int(sample["mask"].any(axis=1).sum()) if admitted else 0,
                   "sampled_observed_node_frames": int(sample["mask"].sum()) if admitted else 0,
                   "graph_archive_bytes": (args.data / meta["graph_archive"]).stat().st_size}
        result["specimens"].append(summary)
        print(json.dumps(summary), flush=True)
    result["total_processed_frames"] = sum(s["n_frames"] for s in result["specimens"])
    result["admitted_training_specimens"] = [s["specimen_id"] for s in result["specimens"] if s["training_admission"]]
    result["total_sampled_training_frames"] = sum(s["sampled_training_frames"] for s in result["specimens"])
    if set(chosen) == set(SPECIMENS):
        summaries = []
        for sp in chosen:
            item = json.loads((args.data / f"{sp}.json").read_text())
            summaries.append({key: value for key, value in item.items() if key != "frames"})
        (args.data / "curation_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.reports / "automatic_image_curation.json").write_text(json.dumps(result, indent=2) + "\n")
    fig.tight_layout()
    fig.savefig(figures / "automatic_video_graphs.png", dpi=170, bbox_inches="tight")
    curvefig.tight_layout()
    curvefig.savefig(figures / "automatic_video_coverage.png", dpi=160, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
