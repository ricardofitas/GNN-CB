"""Audited import of image-derived corrugated-board graphs and trajectories.

The original archives bundle images, graph topology, tracking tables, and
compression-window annotations from the same processing run.  We deliberately
do not join them to later CSV exports, whose node labelling changed.

Coordinates use image convention: x points right and y points down.  Targets
are observed displacements relative to frame 0, not FEM ground truth.  Missing
observations are masked; they are never interpolated or forward filled.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path
from urllib.request import urlopen
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image

SOURCE_REPOSITORY = "https://github.com/ricardofitas/Graphs_Analysis_Corrugated_Boards"
SOURCE_COMMIT = "16eed74ff025e0e0f40f4f5d81c2b71af58b93b5"
ARCHIVES = {
    "Probe2_7": {
        "experiment": "Ex1",
        "filename": "data_Probe2_7.zip",
        "sha256": "8294aba04b87e8d634370307156a359984641c121340d33ecb0bb41d04985490",
    },
    "Probe1_22": {
        "experiment": "Ex2",
        "filename": "data_Probe1_22.zip",
        "sha256": "01f820fc253a477812a7612d61bdfd3840b48cccdda61e88cc239a2f2ef32c72",
    },
}


class RestrictedGraphUnpickler(pickle.Unpickler):
    """Reject every global except the reviewed NetworkX/NumPy data classes.

    This is used only after verifying the pinned archive SHA-256.  Pickle is
    not a general interchange format and arbitrary user pickles must not be
    passed to this importer.
    """

    allowed = {
        "networkx.classes.graph": {"Graph"},
        "networkx.classes.reportviews": {"NodeView", "EdgeView", "DegreeView"},
        "networkx.classes.coreviews": {"AdjacencyView"},
        "numpy": {"ndarray", "dtype"},
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
    }

    def find_class(self, module, name):
        if name not in self.allowed.get(module, set()):
            raise pickle.UnpicklingError(f"Unapproved pickle global: {module}.{name}")
        return super().find_class(module, name)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    names = [n for n in archive.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one member ending {suffix!r}, found {names}")
    return names[0]


def download_sources(destination: str | Path) -> Path:
    """Download only the two checksum-pinned public source archives."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for specification in ARCHIVES.values():
        target = destination / specification["filename"]
        if not target.exists():
            url = (
                "https://raw.githubusercontent.com/ricardofitas/"
                f"Graphs_Analysis_Corrugated_Boards/{SOURCE_COMMIT}/{target.name}"
            )
            with urlopen(url, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != specification["sha256"]:
                raise ValueError(f"Downloaded archive checksum mismatch: {target.name}")
            target.write_bytes(payload)
        if _sha256(target) != specification["sha256"]:
            raise ValueError(f"Source archive checksum mismatch: {target}")
    return destination


def prepare_specimen(source_dir: str | Path, output_dir: str | Path, specimen: str) -> dict:
    """Create one portable NPZ plus provenance from a coherent source archive."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    specification = ARCHIVES[specimen]
    source = source_dir / specification["filename"]
    if _sha256(source) != specification["sha256"]:
        raise ValueError(f"Source archive checksum mismatch: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        graph_member = _find_member(archive, f"graphs_{specimen}_ske_rate10.pkl")
        graphs = RestrictedGraphUnpickler(io.BytesIO(archive.read(graph_member))).load()
        if not isinstance(graphs, list) or not graphs or not all(isinstance(g, nx.Graph) for g in graphs):
            raise ValueError("Expected a nonempty list of NetworkX graphs")
        initial_graph = graphs[0]
        node_ids = np.asarray(sorted(initial_graph.nodes), dtype=np.int64)
        node_index = {int(node): index for index, node in enumerate(node_ids)}
        # NetworkX's 'o' attribute is (row, column), hence the explicit reversal.
        initial_px = np.asarray([initial_graph.nodes[int(n)]["o"][::-1] for n in node_ids], dtype=np.float64)
        if not np.isfinite(initial_px).all():
            raise ValueError("Nonfinite initial geometry")
        origin_px = initial_px.min(axis=0)
        scale_px = float(np.ptp(initial_px[:, 1]))
        if scale_px <= 0:
            raise ValueError("Initial graph has zero height")
        edges = sorted({(node_index[int(u)], node_index[int(v)]) for u, v in initial_graph.edges if u != v})
        edge_index = np.asarray(sorted(set(edges + [(v, u) for u, v in edges])), dtype=np.int64).T
        frame = np.arange(len(graphs), dtype=np.int64)
        displacement_px = np.zeros((len(frame), len(node_ids), 2), dtype=np.float32)
        mask = np.zeros((len(frame), len(node_ids)), dtype=bool)
        mask[0] = True
        tracking_member = _find_member(archive, f"Track_data_{specimen}_ske_rate10.xlsx")
        tracking = pd.ExcelFile(io.BytesIO(archive.read(tracking_member)))
        gap_count = 0
        initial_mismatches = []
        step_sizes = []
        for node in node_ids:
            index = node_index[int(node)]
            table = pd.read_excel(tracking, sheet_name=str(node))
            if table.empty:
                continue
            if table.frame_number.duplicated().any():
                raise ValueError(f"Duplicate frame in track {node}")
            table = table.sort_values("frame_number")
            first = table.iloc[0][["pre_col", "pre_row"]].to_numpy(dtype=float)
            if not np.array_equal(first, initial_px[index]):
                initial_mismatches.append(int(node))
            observed_frames = table.frame_number.to_numpy(dtype=int)
            if observed_frames.min() < 1 or observed_frames.max() >= len(frame):
                raise ValueError(f"Track frame out of range for node {node}")
            positions = table[["new_col", "new_row"]].to_numpy(dtype=float)
            if not np.isfinite(positions).all():
                raise ValueError(f"Nonfinite tracking coordinates for node {node}")
            displacement_px[observed_frames, index] = positions - initial_px[index]
            mask[observed_frames, index] = True
            gap_count += int(np.sum(np.diff(observed_frames) > 1))
            step_sizes.extend(table.min_distance.to_numpy(dtype=float).tolist())
        if initial_mismatches:
            raise ValueError(f"Graph/track initial coordinate mismatches: {initial_mismatches}")
        annotations_member = _find_member(archive, f"img_analysis_{specimen}_ske_rate10.xlsx")
        annotations = pd.read_excel(io.BytesIO(archive.read(annotations_member))).iloc[0]
        start, end = int(annotations.compression_start), int(annotations.compression_end)
        if not 0 <= start < end < len(frame):
            raise ValueError("Invalid compression annotation")
        progress = ((frame - start) / float(end - start)).astype(np.float32)
        compression = (frame >= start) & (frame <= end)
        x0 = ((initial_px - origin_px) / scale_px).astype(np.float32)
        y = displacement_px / scale_px
        np.savez_compressed(
            output_dir / f"{specimen}.npz",
            specimen_id=np.asarray(specimen),
            experiment=np.asarray(specification["experiment"]),
            x0=x0,
            y=y,
            mask=mask,
            edge_index=edge_index,
            progress=progress,
            frame=frame,
            node_ids=node_ids,
            initial_pos_px=initial_px.astype(np.float32),
            displacement_px=displacement_px,
            origin_px=origin_px.astype(np.float32),
            scale_px=np.asarray(scale_px, dtype=np.float32),
            compression_window=np.asarray([start, end], dtype=np.int64),
        )
        # Exact source camera images at selected frames, retained without overlay.
        images = []
        for number in sorted({0, start, (start + end) // 2, end}):
            member = _find_member(archive, f"Frames_{specimen}_rate10/frame_{number:04d}.jpg")
            target = image_dir / f"{specimen}_frame{number:04d}.jpg"
            payload = archive.read(member)
            with Image.open(io.BytesIO(payload)) as image:
                if image.size != (1024, 256):
                    raise ValueError(f"Unexpected camera image size: {image.size}")
            target.write_bytes(payload)
            images.append(str(target.relative_to(output_dir)))
        overlay_member = _find_member(archive, "Graph_Frame_0000.jpg")
        overlay_target = image_dir / f"{specimen}_source_graph_frame0000.jpg"
        overlay_target.write_bytes(archive.read(overlay_member))
    duplicate_observations = 0
    for number in frame:
        positions = (initial_px + displacement_px[number])[mask[number]]
        duplicate_observations += len(positions) - len(np.unique(positions, axis=0))
    metadata = {
        "specimen_id": specimen,
        "experiment": specification["experiment"],
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_archive": specification["filename"],
        "source_archive_sha256": specification["sha256"],
        "source_members": {"graph": graph_member, "tracking": tracking_member, "annotations": annotations_member},
        "n_frames": len(frame),
        "n_nodes": len(node_ids),
        "n_undirected_edges": len(edges),
        "n_directed_edges": int(edge_index.shape[1]),
        "compression_window": [start, end],
        "compression_frames": int(compression.sum()),
        "observed_node_frames": int(mask.sum()),
        "compression_observed_node_frames": int(mask[compression].sum()),
        "compression_coverage": float(mask[compression].mean()),
        "tracking_gap_count": gap_count,
        "duplicate_observed_coordinates": int(duplicate_observations),
        "initial_coordinate_mismatches": initial_mismatches,
        "tracking_step_percentiles_px": dict(zip(["p50", "p95", "p99", "max"], np.percentile(step_sizes, [50, 95, 99, 100]).tolist())),
        "scale_px": scale_px,
        "origin_px": origin_px.tolist(),
        "source_images": images,
        "source_graph_overlay": str(overlay_target.relative_to(output_dir)),
        "image_registration": "Archived camera frames are already 1024x256 crops. Graph0 stores (row,column) in that camera crop; the supplied graph overlay is a separately rendered/cropped plot and must not be used as an unscaled image-coordinate background.",
        "target": "Image-tracked nodal displacement relative to frame 0, divided by initial graph height",
        "coordinate_convention": "x=column/right; y=row/down; source coordinates in pixels",
        "progress_definition": "(frame - compression_start)/(compression_end - compression_start)",
        "progress_limitations": "Author's offline compression-window annotation; this phase is not mechanical strain or an independently measured timestamp.",
        "mask_policy": "Missing observations are zero placeholders with mask=False; no interpolation, smoothing, or forward fill.",
        "label_limitations": "Automated image tracking can lose correspondence, merge nodes, and retain pixel quantization; not displacement ground truth from DIC or FEM.",
        "force_labels": "Not exported: only displacement targets are supported by this importer.",
    }
    (output_dir / f"{specimen}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare_dataset(source_dir: str | Path, output_dir: str | Path) -> dict:
    specimens = [prepare_specimen(source_dir, output_dir, specimen) for specimen in ARCHIVES]
    manifest = {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "independent_specimens": len(specimens),
        "specimens": specimens,
        "split_rule": "Keep each complete specimen in one fold. Never random-split correlated frames or individual nodes.",
        "excluded_data": [
            "Graph-CB newer CSVs: different node labels / tracking versions, not joined to original image graphs.",
            "Graph-CB data_st CSVs: duplicates of existing specimens, not independent measurements.",
            "Exp3: incomplete and late-start trajectories; no corresponding full raw image/topology archive.",
            "Exp4 and Exp5: missing initial-coordinate files and original images/topology; cannot reconstruct a geometry graph reliably.",
        ],
    }
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_specimen(path: str | Path) -> dict[str, np.ndarray]:
    """Read a prepared NPZ without enabling pickle."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_dataset(directory: str | Path) -> list[dict[str, np.ndarray]]:
    return [load_specimen(path) for path in sorted(Path(directory).glob("*.npz"))]
