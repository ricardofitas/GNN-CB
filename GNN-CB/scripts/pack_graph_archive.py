#!/usr/bin/env python3
"""Losslessly pack per-frame material graphs into numeric ragged NumPy arrays.

Run from the repository root:
    python scripts/pack_graph_archive.py

Only a .jsonl.gz whose sibling specimen metadata JSON exists is considered
closed. Original compressed JSON files are preserved. Every numeric array and
all per-frame metadata are reconstructed and compared against the source before
an NPZ is published. Float64 preserves the exact binary64 values parsed by JSON;
no coordinate rounding, downcasting or polyline simplification is performed.

Layout: frame_node_offsets slices nodes_xy/degree; frame_edge_offsets slices
edges_uv/edge_length_px; edge_point_offsets slices edge_points_xy. Edge endpoints
use local per-frame node indices. Per-frame non-graph fields are retained as
UTF-8 JSON bytes using frame_metadata_offsets, allowing full-record round trips.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


FORMAT_VERSION = "ragged_material_graph_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def arrays_for(source: Path) -> tuple[dict[str, np.ndarray], str]:
    node_chunks, degree_chunks, edge_chunks, length_chunks, point_chunks = [], [], [], [], []
    node_offsets, edge_offsets, point_offsets, metadata_offsets = [0], [0], [0], [0]
    metadata = bytearray()
    semantic_digest = hashlib.sha256()
    for record in records(source):
        semantic_digest.update(canonical(record) + b"\n")
        nodes = np.asarray(record["nodes_xy"], dtype=np.float64).reshape(-1, 2)
        degree = np.asarray(record["degree"], dtype=np.int64)
        if not np.isfinite(nodes).all() or len(nodes) != len(degree):
            raise ValueError("Invalid graph coordinates/degrees")
        if degree.tolist() != record["degree"]:
            raise ValueError("Degree values must be exact integers")
        edges, lengths = [], []
        for edge in record["edges"]:
            if set(edge) != {"source", "target", "length_px", "points_xy"}:
                raise ValueError(f"Unexpected edge fields: {sorted(edge)}")
            u, v = edge["source"], edge["target"]
            if int(u) != u or int(v) != v or not (0 <= u < len(nodes) and 0 <= v < len(nodes)):
                raise ValueError("Invalid local edge endpoints")
            points = np.asarray(edge["points_xy"], dtype=np.float64).reshape(-1, 2)
            length = float(edge["length_px"])
            if not np.isfinite(points).all() or not np.isfinite(length):
                raise ValueError("Nonfinite edge data")
            edges.append((u, v))
            lengths.append(length)
            point_chunks.append(points)
            point_offsets.append(point_offsets[-1] + len(points))
        node_chunks.append(nodes)
        degree_chunks.append(degree)
        edge_chunks.append(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        length_chunks.append(np.asarray(lengths, dtype=np.float64))
        node_offsets.append(node_offsets[-1] + len(nodes))
        edge_offsets.append(edge_offsets[-1] + len(edges))
        fields = {key: value for key, value in record.items() if key not in {"nodes_xy", "degree", "edges"}}
        metadata.extend(canonical(fields))
        metadata_offsets.append(len(metadata))
    if len(node_offsets) == 1:
        raise ValueError("Archive contains no frame records")
    join = lambda chunks, shape, dtype: np.concatenate(chunks) if chunks else np.empty(shape, dtype=dtype)
    arrays = {
        "format_version": np.asarray(FORMAT_VERSION),
        "frame_node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "frame_edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "edge_point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "nodes_xy": join(node_chunks, (0, 2), np.float64),
        "degree": join(degree_chunks, (0,), np.int64),
        "edges_uv": join(edge_chunks, (0, 2), np.int64),
        "edge_length_px": join(length_chunks, (0,), np.float64),
        "edge_points_xy": join(point_chunks, (0, 2), np.float64),
        "frame_metadata_offsets": np.asarray(metadata_offsets, dtype=np.int64),
        "frame_metadata_utf8": np.frombuffer(bytes(metadata), dtype=np.uint8),
    }
    return arrays, semantic_digest.hexdigest()


def unpack_frame(archive, index: int) -> dict:
    """Reconstruct an original graph record from an open np.load archive/dict."""
    a, b = archive["frame_metadata_offsets"][index:index + 2]
    record = json.loads(archive["frame_metadata_utf8"][a:b].tobytes().decode("utf-8"))
    a, b = archive["frame_node_offsets"][index:index + 2]
    record["nodes_xy"] = archive["nodes_xy"][a:b].tolist()
    record["degree"] = archive["degree"][a:b].tolist()
    first, last = archive["frame_edge_offsets"][index:index + 2]
    edges = []
    for j in range(int(first), int(last)):
        a, b = archive["edge_point_offsets"][j:j + 2]
        u, v = archive["edges_uv"][j]
        edges.append({"source": int(u), "target": int(v),
                      "length_px": float(archive["edge_length_px"][j]),
                      "points_xy": archive["edge_points_xy"][a:b].tolist()})
    record["edges"] = edges
    return record


def write_npz(destination: Path, arrays: dict[str, np.ndarray]) -> None:
    # NumPy's own NPY writer inside a standards-compatible ZIP, with maximum
    # deflate compression. The result loads with np.load(..., allow_pickle=False).
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name, value in arrays.items():
            with output.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)


def verify(source: Path, packed: Path, expected_semantic_sha256: str) -> dict:
    semantic_digest = hashlib.sha256()
    count = 0
    # Materialize arrays once; repeatedly indexing NpzFile would decompress each
    # member for each frame and dramatically increase verification cost.
    with np.load(packed, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    for index, original in enumerate(records(source)):
        restored = unpack_frame(arrays, index)
        for key in ("nodes_xy", "degree"):
            if not np.array_equal(np.asarray(original[key]), np.asarray(restored[key])):
                raise ValueError(f"Frame {index} differs: {key}")
        if len(original["edges"]) != len(restored["edges"]):
            raise ValueError(f"Frame {index} edge count differs")
        for edge_index, (before, after) in enumerate(zip(original["edges"], restored["edges"])):
            if before["source"] != after["source"] or before["target"] != after["target"]:
                raise ValueError(f"Frame {index} edge {edge_index} endpoints differ")
            if before["length_px"] != after["length_px"] or not np.array_equal(before["points_xy"], after["points_xy"]):
                raise ValueError(f"Frame {index} edge {edge_index} metric geometry differs")
        if original != restored:
            raise ValueError(f"Frame {index} metadata differs")
        # JSON float/integer spellings may differ while all numeric values are
        # exact, so hash the original validated record rather than its spelling.
        semantic_digest.update(canonical(original) + b"\n")
        count += 1
    if count + 1 != len(arrays["frame_node_offsets"]) or count + 1 != len(arrays["frame_edge_offsets"]):
        raise ValueError("Round-trip frame count differs")
    if semantic_digest.hexdigest() != expected_semantic_sha256:
        raise ValueError("Source semantic digest changed during packing")
    return {"frames": count, "nodes_total": len(arrays["nodes_xy"]),
            "edges_total": len(arrays["edges_uv"]), "polyline_points_total": len(arrays["edge_points_xy"]),
            "numeric_coordinate_dtype": str(arrays["nodes_xy"].dtype),
            "numeric_polyline_dtype": str(arrays["edge_points_xy"].dtype),
            "all_records_equal": True, "numeric_arrays_exact": True,
            "semantic_source_sha256": expected_semantic_sha256}


def pack(source: Path, metadata_path: Path) -> dict:
    before = sha256(source)
    metadata = json.loads(metadata_path.read_text())
    destination = source.with_name(source.name.removesuffix(".jsonl.gz") + ".npz")
    temporary = destination.with_suffix(".npz.tmp")
    arrays, semantic_digest = arrays_for(source)
    frame_count = len(arrays["frame_node_offsets"]) - 1
    if metadata.get("n_frames") != frame_count:
        raise ValueError(f"Metadata/archive count mismatch for {source.name}")
    arrays["source_jsonl_gz_sha256"] = np.asarray(before)
    arrays["source_semantic_sha256"] = np.asarray(semantic_digest)
    write_npz(temporary, arrays)
    del arrays
    details = verify(source, temporary, semantic_digest)
    if sha256(source) != before:
        raise ValueError("Source archive changed during packing")
    temporary.replace(destination)
    return {"source_path": str(source), "source_bytes": source.stat().st_size,
            "source_sha256": before, "packed_path": str(destination),
            "packed_bytes": destination.stat().st_size, "packed_sha256": sha256(destination),
            "packed_to_source_ratio": destination.stat().st_size / source.stat().st_size,
            "metadata_path": str(metadata_path), "format_version": FORMAT_VERSION,
            "source_preserved": True, **details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/automatic/graphs"))
    parser.add_argument("--report", type=Path, default=Path("reports/graph_archive_verification.json"))
    parser.add_argument("--specimens", nargs="+")
    args = parser.parse_args()
    previous = json.loads(args.report.read_text()) if args.report.exists() else {"archives": []}
    reports = {row["source_path"]: row for row in previous["archives"]}
    for source in sorted(args.directory.glob("*.jsonl.gz")):
        specimen = source.name.removesuffix(".jsonl.gz")
        if args.specimens and specimen not in args.specimens:
            continue
        metadata = args.directory.parent / (specimen + ".json")
        if not metadata.exists():
            print(json.dumps({"skipped": str(source), "reason": "metadata absent; writer may still be open"}), flush=True)
            continue
        row = pack(source, metadata)
        reports[str(source)] = row
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"format_version": FORMAT_VERSION,
            "packing_policy": "Lossless binary64 coordinates and polylines; exact full-record round-trip validation; original archives retained.",
            "archives": list(reports.values())}, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
