"""Import coherent, checksum-audited image graph archives for three experiments.

The original graph pickle and tracking workbook must come from the same source
folder.  A graph frame can contain a carried-forward coordinate after tracking
failed, so only explicit workbook observations become supervised targets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

from .data import RestrictedGraphUnpickler, load_specimen

SOURCE_REPOSITORY = "https://github.com/ricardofitas/uncertain_cb"
SOURCE_COMMIT = "1df856a4f2c71378744d4a73e3cd958f6f6d7a71"
SPECIFICATIONS = {
    "Exp3": {"rate": 10, "physical_specimen_id": "Exp3_D12", "raw_alias": "experiments/2024_08_05_D12",
             "graph_blob": "819cc3c6cd7b00c0578013c08e16c9c412025632", "track_blob": "38f244f803238676ee0f285e8ea3ca2af8c67842"},
    "Exp4": {"rate": 10, "physical_specimen_id": "Exp4_D14", "raw_alias": "experiments/2024_08_05_D14",
             "graph_blob": "0700f6fc61af412c60519a3f292f69859de7d048", "track_blob": "2047315c73639085cabd2eea41db70df2d0c6561"},
    "Exp5": {"rate": 3, "physical_specimen_id": "Exp5_ID03", "raw_alias": "experiments/2024_12_03_ID_03",
             "graph_blob": "59f0382ecf71b7a8262714f2f852c88d57f74103", "track_blob": "a26e07bfa5e57ebf16dcecdd75621a6a1c59b8dc"},
}


class ExpandedGraphUnpickler(RestrictedGraphUnpickler):
    """Same restricted data classes, including the NumPy 2 module spelling."""

    allowed = {**RestrictedGraphUnpickler.allowed,
               "numpy._core.multiarray": {"_reconstruct", "scalar"}}


def fingerprint(path: Path, expected_blob: str | None = None) -> dict:
    payload = path.read_bytes()
    git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if expected_blob is not None and git_blob != expected_blob:
        raise ValueError(f"Pinned Git blob checksum mismatch: {path.name}")
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob_sha1": git_blob}


def prepare_expanded_specimen(source_root: str | Path, output_dir: str | Path,
                              specimen: str, source_tree: dict) -> dict:
    source_root, output_dir = Path(source_root), Path(output_dir)
    spec = SPECIFICATIONS[specimen]
    suffix = f"{specimen}_ske_rate{spec['rate']}"
    relative = Path(f"data_{specimen}") / suffix
    paths = {"graph": relative / f"graphs_{suffix}.pkl",
             "tracking": relative / f"Track_data_{suffix}.xlsx",
             "annotations": relative / f"img_analysis_{suffix}.xlsx"}
    entries = {entry["path"]: entry for entry in source_tree["tree"]}
    provenance = {}
    for name, relative_path in paths.items():
        entry = entries[str(relative_path)]
        expected = spec.get({"graph": "graph_blob", "tracking": "track_blob"}.get(name, ""), entry["sha"])
        provenance[name] = {"path": f"Taieb/{relative_path}",
                            **fingerprint(source_root / relative_path, expected)}
    # The pickle globals are restricted AND its content is pinned before loading.
    with (source_root / paths["graph"]).open("rb") as stream:
        graphs = ExpandedGraphUnpickler(stream).load()
    if not isinstance(graphs, list) or not graphs or not all(isinstance(g, nx.Graph) for g in graphs):
        raise ValueError("Expected a nonempty list of NetworkX graphs")
    initial_graph = graphs[0]
    node_ids = np.asarray(sorted(initial_graph.nodes), dtype=np.int64)
    node_index = {int(node): index for index, node in enumerate(node_ids)}
    initial_px = np.asarray([initial_graph.nodes[int(n)]["o"][::-1] for n in node_ids], dtype=float)
    if not np.isfinite(initial_px).all():
        raise ValueError("Nonfinite initial coordinates")
    if len(np.unique(initial_px, axis=0)) != len(initial_px):
        raise ValueError("Ambiguous duplicate coordinates in initial graph")
    origin_px = initial_px.min(axis=0)
    scale_px = float(np.ptp(initial_px[:, 1]))
    if scale_px <= 0:
        raise ValueError("Invalid initial graph height")
    edges = sorted({tuple(sorted((node_index[int(u)], node_index[int(v)])))
                    for u, v in initial_graph.edges if u != v})
    edge_index = np.asarray(sorted(set(edges + [(v, u) for u, v in edges])), dtype=np.int64).T
    frame = np.arange(len(graphs), dtype=np.int64)
    displacement_px = np.zeros((len(frame), len(node_ids), 2), dtype=np.float32)
    source_mask = np.zeros((len(frame), len(node_ids)), dtype=bool)
    source_mask[0] = True
    workbook = pd.ExcelFile(source_root / paths["tracking"])
    excluded_later_nodes = []
    first_observations = {}
    initial_mismatches, graph_mismatches, chain_mismatches = [], [], []
    steps, gap_count = [], 0
    for sheet in workbook.sheet_names:
        table = pd.read_excel(workbook, sheet_name=sheet)
        if table.empty:
            continue
        node = int(sheet)
        if node not in node_index:
            excluded_later_nodes.append({"node_id": node, "observations": len(table)})
            continue
        j = node_index[node]
        if table.frame_number.duplicated().any():
            raise ValueError(f"Duplicate frame index in node {node}")
        table = table.sort_values("frame_number")
        raw_frames = table.frame_number.to_numpy(dtype=float)
        observed_frames = raw_frames.astype(int)
        if (not np.array_equal(raw_frames, observed_frames) or observed_frames.min() < 1
                or observed_frames.max() >= len(frame)):
            raise ValueError(f"Invalid frame index for node {node}")
        pre = table[["pre_col", "pre_row"]].to_numpy(dtype=float)
        xy = table[["new_col", "new_row"]].to_numpy(dtype=float)
        if not np.isfinite(pre).all() or not np.isfinite(xy).all():
            raise ValueError(f"Nonfinite track for node {node}")
        if not np.array_equal(pre[0], initial_px[j]):
            initial_mismatches.append(node)
        if len(pre) > 1 and not np.array_equal(pre[1:], xy[:-1]):
            chain_mismatches.append(node)
        for f, point in zip(observed_frames, xy):
            if node not in graphs[f] or not np.array_equal(graphs[f].nodes[node]["o"][::-1], point):
                graph_mismatches.append([node, int(f)])
        first_observations[str(node)] = int(observed_frames[0])
        displacement_px[observed_frames, j] = xy - initial_px[j]
        source_mask[observed_frames, j] = True
        gap_count += int(np.sum(np.diff(observed_frames) > 1))
        steps.extend(table.min_distance.to_numpy(dtype=float).tolist())
    if initial_mismatches or graph_mismatches or chain_mismatches:
        raise ValueError(f"Source alignment failed: initial={initial_mismatches}; "
                         f"graph={graph_mismatches[:5]}; chain={chain_mismatches}")
    # Several source nodes may match one skeleton junction. Do not count that as
    # independent ground truth for either node; mask every member of a collision.
    collision_mask = np.zeros_like(source_mask)
    for f in frame:
        indices = np.flatnonzero(source_mask[f])
        xy = initial_px[indices] + displacement_px[f, indices]
        _, inverse, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
        collision_mask[f, indices] = counts[inverse] > 1
    mask = source_mask & ~collision_mask
    displacement_px[~mask] = 0.0
    annotations = pd.read_excel(source_root / paths["annotations"]).iloc[0]
    start, end = int(annotations.compression_start), int(annotations.compression_end)
    if not 0 <= start < end < len(frame):
        raise ValueError("Invalid compression window")
    compression = (frame >= start) & (frame <= end)
    progress = ((frame - start) / float(end - start)).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{specimen}.npz", specimen_id=np.asarray(specimen),
        physical_specimen_id=np.asarray(spec["physical_specimen_id"]), experiment=np.asarray(specimen),
        x0=((initial_px - origin_px) / scale_px).astype(np.float32),
        y=displacement_px / scale_px, mask=mask, source_mask=source_mask,
        collision_mask=collision_mask, edge_index=edge_index, progress=progress,
        frame=frame, node_ids=node_ids, initial_pos_px=initial_px.astype(np.float32),
        displacement_px=displacement_px, origin_px=origin_px.astype(np.float32),
        scale_px=np.asarray(scale_px, dtype=np.float32),
        compression_window=np.asarray([start, end], dtype=np.int64))
    frame_relative = Path(f"data_{specimen}/Frames_{specimen}_rate{spec['rate']}")
    frame_entries = [entry for path, entry in entries.items()
                     if path.startswith(str(frame_relative) + "/frame_") and path.endswith(".jpg")]
    if {Path(entry["path"]).name for entry in frame_entries} != {f"frame_{f:04d}.jpg" for f in frame}:
        raise ValueError("Raw image inventory does not exactly match graph frame indices")
    available_frames = sorted((source_root / frame_relative).glob("frame_*.jpg"))
    available_info = []
    for image_path in available_frames:
        rel = str(image_path.relative_to(source_root))
        available_info.append({"path": f"Taieb/{rel}", **fingerprint(image_path, entries[rel]["sha"])})
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    image_records, selected_missing = [], []
    for f in sorted({0, start, (start + end) // 2, end}):
        image_path = source_root / frame_relative / f"frame_{f:04d}.jpg"
        if not image_path.exists():
            selected_missing.append(f)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        target = image_dir / f"{specimen}_frame{f:04d}.jpg"
        shutil.copyfile(image_path, target)
        image_records.append({"frame": f, "path": str(target.relative_to(output_dir)),
                              "width": width, "height": height,
                              **fingerprint(image_path)})
    metadata = {
        "specimen_id": specimen, "physical_specimen_id": spec["physical_specimen_id"],
        "admissibility": "admitted_with_masked_automated_tracking_labels",
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "source_taieb_tree": source_tree["sha"], "source_files": provenance,
        "raw_experiment_alias": spec["raw_alias"],
        "alias_policy": "Taieb/Sample_videos aliases and experiment paths are one physical specimen, not extra folds.",
        "processing_rate": spec["rate"], "n_frames": len(frame), "n_nodes": len(node_ids),
        "n_undirected_edges": len(edges), "n_directed_edges": int(edge_index.shape[1]),
        "compression_window": [start, end], "compression_frames": int(compression.sum()),
        "source_observed_node_frames": int(source_mask.sum()), "observed_node_frames": int(mask.sum()),
        "source_compression_observed_node_frames": int(source_mask[compression].sum()),
        "compression_observed_node_frames": int(mask[compression].sum()),
        "compression_coverage": float(mask[compression].mean()),
        "collision_observations_removed": int(collision_mask.sum()),
        "compression_collision_observations_removed": int(collision_mask[compression].sum()),
        "initial_coordinate_mismatches": initial_mismatches,
        "graph_coordinate_mismatches": graph_mismatches, "tracking_chain_mismatches": chain_mismatches,
        "excluded_later_born_nodes": excluded_later_nodes,
        "first_observation_by_initial_node": first_observations, "tracking_gap_count": gap_count,
        "tracking_step_percentiles_px": dict(zip(["p50", "p95", "p99", "max"], np.percentile(steps, [50, 95, 99, 100]).tolist())),
        "scale_px": scale_px, "origin_px": origin_px.tolist(),
        "raw_frames_in_source_tree": len(frame_entries), "raw_frames_downloaded_and_hash_verified": len(available_info),
        "raw_frame_fingerprints": available_info, "selected_source_images": image_records,
        "selected_source_images_not_yet_downloaded": selected_missing,
        "mask_policy": "Only explicit workbook matches; all duplicate-coordinate matches masked. Missing labels remain mask=False with zero placeholders. No interpolation or graph carry-forward as labels.",
        "reference_policy": "Every included node exists in graph0, its first pre-coordinate equals graph0, its track is continuous in observed-event order, and every observed coordinate equals the same node in the same archived graph frame. Late first observations retain the true graph0 reference.",
        "target": "Image-tracked displacement relative to frame0 / initial graph height, in image x-right/y-down coordinates.",
        "progress_limitations": "Author's offline compression-window annotation; not measured strain. Exp3/4/5 all use start=20; these annotations were not independently force/time calibrated.",
        "label_limitations": "Automated skeleton matching is a noisy image-derived label, not DIC/FEM ground truth; exact consistency does not establish physical node identity after buckling.",
        "force_labels": "Not imported: no verified synchronization to an independent mechanical force series.",
    }
    (output_dir / f"{specimen}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare_expanded(source_root: str | Path, base_dir: str | Path, output_dir: str | Path,
                     tree_path: str | Path, audit_path: str | Path) -> dict:
    source_root, base_dir, output_dir = Path(source_root), Path(base_dir), Path(output_dir)
    if (source_root / "Taieb").is_dir():
        source_root = source_root / "Taieb"
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = json.loads(Path(tree_path).read_text())
    # Ship only the source entries needed to reproduce this import. These are
    # plain Git paths/hashes, never temporary download URLs or credentials.
    used_prefixes = []
    for specimen, specification in SPECIFICATIONS.items():
        used_prefixes.extend([f"data_{specimen}/{specimen}_ske_rate{specification['rate']}/",
                              f"data_{specimen}/Frames_{specimen}_rate{specification['rate']}/"])
    compact_tree = {"sha": tree["sha"], "source_commit": SOURCE_COMMIT,
                    "tree": [{k: entry[k] for k in ("path", "mode", "type", "sha", "size") if k in entry}
                             for entry in tree["tree"]
                             if any(entry["path"].startswith(prefix) for prefix in used_prefixes)
                             and (entry["path"].endswith(".jpg") or Path(entry["path"]).name.startswith(("graphs_", "Track_data_", "img_analysis_")))]}
    (output_dir / "source_tree.json").write_text(json.dumps(compact_tree, indent=2) + "\n")
    specimens = []
    for path in sorted(base_dir.glob("*.npz")):
        target = output_dir / path.name
        shutil.copyfile(path, target)
        if fingerprint(path)["sha256"] != fingerprint(target)["sha256"]:
            raise ValueError("Original NPZ did not copy unchanged")
        metadata = json.loads(path.with_suffix(".json").read_text())
        shutil.copyfile(path.with_suffix(".json"), target.with_suffix(".json"))
        specimens.append({**metadata, "physical_specimen_id": metadata["specimen_id"],
                          "baseline_npz_unchanged": True, "npz_sha256": fingerprint(target)["sha256"]})
    if (base_dir / "images").exists():
        shutil.copytree(base_dir / "images", output_dir / "images", dirs_exist_ok=True)
    for specimen in SPECIFICATIONS:
        specimens.append(prepare_expanded_specimen(source_root, output_dir, specimen, tree))
    manifest = {
        "schema_version": 2, "independent_specimens": len(specimens),
        "specimens": specimens,
        "totals": {key: sum(item[key] for item in specimens) for key in
                   ["n_frames", "n_nodes", "n_undirected_edges", "compression_frames", "compression_observed_node_frames"]},
        "split_rule": "Entire physical specimen is one group; aliases, rates and re-extractions never count as independent experiments.",
        "exclusions": [
            "Taieb/Exp5 rate10 duplicates the physical specimen represented by rate3; use only rate3.",
            "Later CSV exports have different processing versions; use only the archived original graph+workbook pair.",
            "Graph-carried coordinates without an explicit workbook observation are excluded from labels.",
            "Duplicate-coordinate source matches are excluded in the three newly imported datasets.",
        ],
        "new_data_qualification": "Three additional physical specimens recovered from existing private repository artifacts. Re-extraction from images is a separate sensitivity analysis; this importer validates and curates the original image extraction.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Load through the existing public schema and check the masks/geometry.
    for path in sorted(output_dir.glob("*.npz")):
        data = load_specimen(path)
        assert np.isfinite(data["x0"]).all() and np.isfinite(data["y"][data["mask"]]).all()
        assert data["edge_index"].max() < len(data["x0"])
    return manifest
