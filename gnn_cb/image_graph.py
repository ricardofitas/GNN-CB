"""Inspectable camera image -> material graph -> masked nodal trajectories.

The extraction follows the existing Graph-CB methodology (blur, threshold,
component filtering, skeletonisation and contraction), without importing its
monolithic interactive script. Junction clusters and endpoints are nodes;
degree-two skeleton pixels are retained as edge polylines, not model nodes.
Automatic tracks are observations with uncertainty, never DIC ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import gzip
import json
import os
from pathlib import Path
import re

import cv2
import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ExtractionConfig:
    threshold: float = 20.0
    threshold_mode: str = "fixed"  # fixed reproduces source preprocessing
    gaussian_kernel: int = 7
    gaussian_sigma: float = 1.0
    median_kernel: int = 5
    min_component_pixels: int = 150
    min_component_height: int = 9
    min_spur_length: float = 8.0
    border_tolerance: float = 12.0
    max_track_distance: float = 10.0
    max_gap: int = 2
    ambiguity_margin: float = 1.5
    min_frame_coverage: float = 0.65
    roi: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1; image coordinates


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError("Expected grayscale or BGR image")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all():
            raise ValueError("Nonfinite image pixels")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def preprocess(image: np.ndarray, config: ExtractionConfig) -> tuple[np.ndarray, dict]:
    """Segment bright material while excluding isolated loading plate lines.

    Component height is measured *before* selecting a specimen. Thin isolated
    horizontal components are omitted, but liners connected to a corrugation
    remain. No closing, gap filling or guessed material bridges are applied.
    """
    gray = _gray(image)
    if config.threshold_mode not in {"fixed", "otsu"}:
        raise ValueError("threshold_mode must be fixed or otsu")
    if config.gaussian_kernel < 1 or config.gaussian_kernel % 2 != 1:
        raise ValueError("gaussian_kernel must be a positive odd integer")
    blurred = cv2.GaussianBlur(gray, (config.gaussian_kernel,) * 2, config.gaussian_sigma)
    if config.threshold_mode == "otsu":
        threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold = config.threshold
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    if config.roi is not None:
        x0, y0, x1, y1 = config.roi
        if not 0 <= x0 < x1 <= gray.shape[1] or not 0 <= y0 < y1 <= gray.shape[0]:
            raise ValueError("ROI falls outside image")
        roi_mask = np.zeros_like(binary)
        roi_mask[y0:y1, x0:x1] = 255
        binary &= roi_mask
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    retained, rejected = [], []
    for label in range(1, nlabels):
        x, y, w, h, area = map(int, stats[label])
        record = {"label": label, "bbox_xywh": [x, y, w, h], "area": area}
        # An intact corrugation spans at least 9 px vertically. Small broken
        # pieces are deliberately withheld and reported instead of stitched.
        if area >= config.min_component_pixels and h >= config.min_component_height:
            retained.append(record)
        else:
            rejected.append(record)
    clean = np.isin(labels, [c["label"] for c in retained]).astype(np.uint8) * 255
    if config.median_kernel > 1:
        if config.median_kernel % 2 != 1:
            raise ValueError("median_kernel must be odd")
        clean = cv2.medianBlur(clean, config.median_kernel)
    info = {
        "threshold_used": float(threshold), "retained_components": retained,
        "rejected_component_count": len(rejected),
        "retained_foreground_pixels": int(np.count_nonzero(clean)),
        "image_shape": list(gray.shape),
        "foreground_fraction": float(np.mean(clean > 0)),
    }
    return clean > 0, info


def _ordered_points(graph: nx.MultiGraph, u: int, v: int, data: dict) -> np.ndarray:
    points = data["points_xy"]
    return points if np.linalg.norm(points[0] - graph.nodes[u]["xy"]) <= np.linalg.norm(points[-1] - graph.nodes[u]["xy"]) else points[::-1]


def _contract_degree_two(graph: nx.MultiGraph) -> None:
    while True:
        removable = [n for n in graph if graph.degree(n) == 2 and len(set(graph.neighbors(n))) == 2]
        if not removable:
            break
        node = removable[0]
        u, v = list(graph.neighbors(node))
        a = next(iter(graph[u][node].values()))
        b = next(iter(graph[node][v].values()))
        pts = np.vstack([_ordered_points(graph, u, node, a), _ordered_points(graph, node, v, b)])
        graph.remove_node(node)
        graph.add_edge(u, v, points_xy=pts, length_px=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))


def skeleton_graph(skeleton: np.ndarray, config: ExtractionConfig) -> tuple[nx.MultiGraph, dict]:
    """Collapse the material skeleton into a graph with physical junctions.

    Eight-connected clusters of non-degree-two pixels form one junction. A
    component made solely of degree-two pixels is an isolated ring; three
    evenly spaced anchors retain that cycle instead of deleting it.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    neighbors_count = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    key_pixels = skeleton & (neighbors_count != 2)
    key_labels, nkeys = ndimage.label(key_pixels, np.ones((3, 3), dtype=np.uint8))
    graph = nx.MultiGraph()
    for label in range(1, nkeys + 1):
        yx = np.argwhere(key_labels == label)
        graph.add_node(label - 1, xy=yx[:, ::-1].mean(axis=0), pixels_yx=yx)
    pixels = set(map(tuple, np.argwhere(skeleton).tolist()))
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

    def adjacent(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in offsets if (p[0] + dy, p[1] + dx) in pixels]

    visited = set()
    for node, data in list(graph.nodes(data=True)):
        for raw in data["pixels_yx"]:
            start = tuple(raw)
            for neighbor in adjacent(start):
                pair = frozenset((start, neighbor))
                if pair in visited or key_labels[neighbor] == node + 1:
                    continue
                path = [start, neighbor]
                visited.add(pair)
                prev, current = start, neighbor
                while key_labels[current] == 0:
                    candidates = [p for p in adjacent(current) if p != prev]
                    if len(candidates) != 1:
                        break
                    nxt = candidates[0]
                    pair = frozenset((current, nxt))
                    if pair in visited:
                        break
                    visited.add(pair)
                    path.append(nxt)
                    prev, current = current, nxt
                end_label = key_labels[current]
                if end_label > 0 and int(end_label) - 1 != node:
                    end = int(end_label) - 1
                    pts = np.vstack([graph.nodes[node]["xy"], np.asarray(path)[:, ::-1], graph.nodes[end]["xy"]])
                    graph.add_edge(node, end, points_xy=pts, length_px=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))

    # Record isolated rings explicitly. They are unusual for board specimens
    # but valid skeleton material, and are flagged for review below.
    components, ncomponents = ndimage.label(skeleton, np.ones((3, 3), dtype=np.uint8))
    ring_count = 0
    for component in range(1, ncomponents + 1):
        coords = np.argwhere(components == component)
        if not len(coords) or np.any(key_labels[components == component]):
            continue
        start = tuple(coords[0])
        route = [start]
        prev, curr = None, start
        while True:
            candidates = [p for p in adjacent(curr) if p != prev]
            if not candidates:
                break
            nxt = candidates[0]
            if nxt == start:
                break
            route.append(nxt)
            prev, curr = curr, nxt
            if len(route) > len(coords):
                break
        if len(route) < 3:
            continue
        ring_count += 1
        indices = [0, len(route) // 3, 2 * len(route) // 3]
        ids = []
        for index in indices:
            node = max(graph.nodes, default=-1) + 1
            graph.add_node(node, xy=np.asarray(route[index][::-1], dtype=float), pixels_yx=np.asarray([route[index]]), ring_anchor=True)
            ids.append(node)
        extended = route + [route[0]]
        for i in range(3):
            stop = indices[i + 1] if i < 2 else len(route)
            pts = np.asarray(extended[indices[i]:stop + 1])[:, ::-1].astype(float)
            graph.add_edge(ids[i], ids[(i + 1) % 3], points_xy=pts, length_px=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
    removed_spurs = 0
    xs = [data["xy"][0] for _, data in graph.nodes(data=True)]
    minx, maxx = (min(xs), max(xs)) if xs else (0, 0)
    while True:
        remove = []
        for n in graph:
            if graph.degree(n) != 1:
                continue
            x = graph.nodes[n]["xy"][0]
            if x <= minx + config.border_tolerance or x >= maxx - config.border_tolerance:
                continue
            edge = next(iter(graph.edges(n, data=True)))[2]
            if edge["length_px"] < config.min_spur_length:
                remove.append(n)
        if not remove:
            break
        removed_spurs += len(remove)
        graph.remove_nodes_from(remove)
        _contract_degree_two(graph)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    # Stable indexing is geometric, unrelated to reference labels or targets.
    order = sorted(graph, key=lambda n: (graph.nodes[n]["xy"][0], graph.nodes[n]["xy"][1]))
    graph = nx.relabel_nodes(graph, {old: new for new, old in enumerate(order)}, copy=True)
    return graph, {"pruned_short_spurs": removed_spurs, "isolated_ring_count": ring_count}


def extract_graph(image: np.ndarray, config: ExtractionConfig | None = None) -> dict:
    config = config or ExtractionConfig()
    binary, metadata = preprocess(image, config)
    skeleton = skeletonize(binary)
    graph, graph_info = skeleton_graph(skeleton, config)
    nodes = np.asarray([graph.nodes[n]["xy"] for n in sorted(graph)], dtype=np.float32).reshape(-1, 2)
    degree = np.asarray([graph.degree(n) for n in sorted(graph)], dtype=np.int64)
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({"source": int(u), "target": int(v), "length_px": data["length_px"], "points_xy": data["points_xy"].tolist()})
    flags = []
    if len(nodes) < 4:
        flags.append("too_few_structural_nodes")
    if len(nodes) and np.ptp(nodes[:, 1]) < config.min_component_height:
        flags.append("insufficient_graph_height")
    if metadata["foreground_fraction"] > 0.5:
        flags.append("excessive_foreground")
    if nx.number_connected_components(graph) > 1:
        flags.append("disconnected_material")
    if graph_info["isolated_ring_count"]:
        flags.append("isolated_ring_requires_review")
    multiedges = len(edges) - len({tuple(sorted((e["source"], e["target"]))) for e in edges})
    if multiedges:
        flags.append("parallel_material_paths")
    metadata.update(graph_info)
    metadata.update(n_nodes=len(nodes), n_edges=len(edges), n_components=nx.number_connected_components(graph),
                    cycle_rank=len(edges) - len(nodes) + nx.number_connected_components(graph), quality_flags=flags,
                    valid_geometry=not any(flag in flags for flag in ["too_few_structural_nodes", "insufficient_graph_height", "excessive_foreground"]))
    return {"nodes_xy": nodes, "degree": degree, "edges": edges, "metadata": metadata, "binary": binary, "skeleton": skeleton}


def gated_assignment(reference: np.ndarray, observed: np.ndarray, max_distance: float,
                     reference_degree: np.ndarray | None = None, observed_degree: np.ndarray | None = None,
                     ambiguity_margin: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one assignment with dummies: out-of-gate matches stay absent.

    Ambiguous local alternatives (within ambiguity_margin pixels of the best
    distance) are withheld. Degree is a matching cue, never a hard label.
    """
    reference, observed = np.asarray(reference).reshape(-1, 2), np.asarray(observed).reshape(-1, 2)
    assigned = np.full(len(reference), -1, dtype=np.int64)
    distances = np.full(len(reference), np.inf)
    ambiguous = np.zeros(len(reference), dtype=bool)
    if not len(reference) or not len(observed):
        return assigned, distances, ambiguous
    dist = cdist(reference, observed)
    cost = dist.copy()
    if reference_degree is not None and observed_degree is not None:
        cost += np.minimum(np.abs(np.asarray(reference_degree)[:, None] - np.asarray(observed_degree)[None]), 2) * 1.0
    allowed = dist <= max_distance
    cost[~allowed] = 1e6
    # A private unmatched slot for each reference prevents forced assignment.
    augmented = np.hstack([cost, np.full((len(reference), len(reference)), max_distance + 3.0)])
    rows, cols = linear_sum_assignment(augmented)
    for row, col in zip(rows, cols):
        if col >= len(observed) or not allowed[row, col]:
            continue
        alternatives = dist[row][allowed[row]]
        if len(alternatives) > 1:
            sorted_dist = np.sort(alternatives)
            ambiguous[row] = sorted_dist[1] - sorted_dist[0] < ambiguity_margin
        # A competing reference at essentially the same location also makes
        # the physical identity uncertain, even if Hungarian picks a winner.
        competitors = np.sort(dist[:, col][allowed[:, col]])
        if len(competitors) > 1 and competitors[1] - competitors[0] < ambiguity_margin:
            ambiguous[row] = True
        if not ambiguous[row]:
            assigned[row] = int(col)
            distances[row] = float(dist[row, col])
    return assigned, distances, ambiguous


def track_graphs(graphs: list[dict], config: ExtractionConfig | None = None) -> dict:
    config = config or ExtractionConfig()
    if not graphs or not graphs[0]["metadata"]["valid_geometry"]:
        raise ValueError("First frame needs valid material geometry")
    initial = graphs[0]["nodes_xy"].copy()
    nframes, nnodes = len(graphs), len(initial)
    positions = np.zeros((nframes, nnodes, 2), dtype=np.float32)
    mask = np.zeros((nframes, nnodes), dtype=bool)
    match_indices = np.full((nframes, nnodes), -1, dtype=np.int64)
    distance = np.full((nframes, nnodes), np.nan, dtype=np.float32)
    ambiguity = np.zeros((nframes, nnodes), dtype=bool)
    positions[0], mask[0], match_indices[0], distance[0] = initial, True, np.arange(nnodes), 0
    last = initial.copy()
    last_seen = np.zeros(nnodes, dtype=int)
    last_degree = graphs[0]["degree"].copy()
    frame_valid = np.ones(nframes, dtype=bool)
    for frame, graph in enumerate(graphs[1:], start=1):
        active = np.flatnonzero(frame - last_seen <= config.max_gap + 1)
        if not graph["metadata"]["valid_geometry"]:
            frame_valid[frame] = False
            continue
        assignments, distances, ambiguous = gated_assignment(last[active], graph["nodes_xy"], config.max_track_distance,
            last_degree[active], graph["degree"], config.ambiguity_margin)
        ambiguity[frame, active] = ambiguous
        matched = assignments >= 0
        ids, obs = active[matched], assignments[matched]
        positions[frame, ids] = graph["nodes_xy"][obs]
        mask[frame, ids] = True
        match_indices[frame, ids], distance[frame, ids] = obs, distances[matched]
        last[ids], last_seen[ids], last_degree[ids] = positions[frame, ids], frame, graph["degree"][obs]
        frame_valid[frame] = float(mask[frame].mean()) >= config.min_frame_coverage
    displacement = positions - initial[None]
    displacement[~mask] = 0
    return {"positions_px": positions, "displacement_px": displacement, "mask": mask,
            "matched_node_indices": match_indices, "matching_distance_px": distance,
            "ambiguous": ambiguity, "frame_valid": frame_valid}


def save_overlay(image: np.ndarray, graph: dict, destination: str | Path, tracked_nodes: np.ndarray | None = None) -> None:
    image = _gray(image)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for edge in graph["edges"]:
        pts = np.round(edge["points_xy"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (100, 210, 80), 1, cv2.LINE_AA)
    for i, xy in enumerate(graph["nodes_xy"]):
        x, y = np.round(xy).astype(int)
        cv2.circle(canvas, (x, y), 3, (30, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60, 230, 255), 1, cv2.LINE_AA)
    if tracked_nodes is not None:
        for xy in tracked_nodes:
            cv2.circle(canvas, tuple(np.round(xy).astype(int)), 5, (255, 170, 20), 1, cv2.LINE_AA)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise IOError(f"Cannot write overlay {destination}")


def natural_key(path: Path) -> list:
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", str(path))]


def curate_sequence(paths: list[Path], output: Path, specimen_id: str, config: ExtractionConfig | None = None,
                    compression_window: tuple[int, int] | None = None, source: dict | None = None,
                    lineage_id: str | None = None, overlay_count: int = 5) -> dict:
    """Save auditable graph JSON and masked tracks, optionally a training NPZ.

    A compression window is never guessed from an image count. Without an
    independently supplied window, tracks are exported under ``tracks/`` and
    remain ineligible for the phase-conditioned model's automatic loader.
    """
    config = config or ExtractionConfig()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise ValueError("No image paths")
    graphs, frames = [], []
    selected = set(np.linspace(0, len(paths) - 1, min(overlay_count, len(paths)), dtype=int).tolist())
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        graph = extract_graph(image, config)
        record = {"sequence_index": i, "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "nodes_xy": graph["nodes_xy"].tolist(), "degree": graph["degree"].tolist(),
                  "edges": graph["edges"], "quality": graph["metadata"]}
        frames.append(record)
        # Large raster buffers are unnecessary after extraction.
        graph.pop("binary")
        graph.pop("skeleton")
        graphs.append(graph)
        if i in selected:
            save_overlay(image, graph, output / "overlays" / f"{specimen_id}_{i:05d}.png")
    tracks = track_graphs(graphs, config)
    initial = graphs[0]["nodes_xy"]
    origin = initial.min(axis=0)
    scale = float(np.ptp(initial[:, 1]))
    if scale <= 0:
        raise ValueError("Zero initial graph height")
    edges = {(edge["source"], edge["target"]) for edge in graphs[0]["edges"]}
    pairs = sorted(edges | {(v, u) for u, v in edges})
    edge_index = np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
    common = {"specimen_id": np.asarray(specimen_id), "lineage_id": np.asarray(lineage_id or specimen_id),
              "experiment": np.asarray("automatic_image_extraction"), "x0": (initial - origin) / scale,
              "y": tracks["displacement_px"] / scale, "mask": tracks["mask"],
              "edge_index": edge_index, "frame": np.arange(len(paths)), "node_ids": np.arange(len(initial)),
              "initial_pos_px": initial, "origin_px": origin, "scale_px": np.asarray(scale, dtype=np.float32),
              **tracks}
    (output / "tracks").mkdir(exist_ok=True)
    np.savez_compressed(output / "tracks" / f"{specimen_id}.npz", **common)
    training_written = False
    if compression_window is not None:
        start, end = compression_window
        if not 0 <= start < end < len(paths):
            raise ValueError("Compression window must use valid sequence indices")
        validmask = tracks["mask"] & tracks["frame_valid"][:, None]
        np.savez_compressed(output / f"{specimen_id}.npz", **{**common, "mask": validmask},
            progress=((np.arange(len(paths)) - start) / (end - start)).astype(np.float32),
            compression_window=np.asarray([start, end]))
        training_written = True
    (output / "graphs").mkdir(exist_ok=True)
    graph_archive = output / "graphs" / f"{specimen_id}.jsonl.gz"
    temporary_archive = graph_archive.with_suffix(".gz.tmp")
    with gzip.open(temporary_archive, "wt", encoding="utf-8") as stream:
        for record in frames:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    # Validate the entire compressed stream before making it discoverable by
    # downstream tooling. A partial file must never masquerade as 2,048 graphs.
    verified = 0
    with gzip.open(temporary_archive, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["sequence_index"] != verified or record["sha256"] != frames[verified]["sha256"]:
                raise ValueError("Graph archive frame identity verification failed")
            verified += 1
    if verified != len(paths):
        raise ValueError("Graph archive frame count verification failed")
    os.replace(temporary_archive, graph_archive)
    metadata = {"specimen_id": specimen_id, "lineage_id": lineage_id or specimen_id, "source": source or {},
                "config": asdict(config), "n_frames": len(paths), "n_initial_nodes": len(initial),
                "n_initial_edges": len(graphs[0]["edges"]), "observed_node_frames": int(tracks["mask"].sum()),
                "tracking_coverage": float(tracks["mask"].mean()), "valid_frames": int(tracks["frame_valid"].sum()),
                "ambiguous_node_frames": int(tracks["ambiguous"].sum()), "compression_window": compression_window,
                "training_npz_written": training_written,
                "target_provenance": "Automatic skeleton-junction correspondences from camera images, not independent DIC/FEM ground truth.",
                "mask_policy": "Only gated one-to-one, unambiguous observed matches are true. Missing targets are zero placeholders. Low-coverage frames masked in training NPZ.",
                "split_policy": "All versions, crops and frame subsets of one physical experiment must share lineage_id and one dataset split.",
                "parallel_edge_policy": "Polylines retained in JSON; phase surrogate consumes one adjacency per node pair.",
                "coordinate_convention": "x=column/right, y=row/down, pixels in original image coordinates",
                "graph_archive": str(graph_archive.relative_to(output)),
                "frames": [{key: value for key, value in frame.items() if key not in {"nodes_xy", "degree", "edges"}} for frame in frames]}
    (output / f"{specimen_id}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {key: val for key, val in metadata.items() if key != "frames"}
