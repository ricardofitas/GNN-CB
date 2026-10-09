"""Predict graph-node deformation from initial geometry and supplied protocol phase.

This is a conditional surrogate, not an image detector or an autonomous forecast.
Only x0, edge_index, initial_pos_px, node_ids and scale_px are read from the graph.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint


def predict(checkpoint: str | Path, graph: str | Path, phases: list[float]) -> dict:
    phase = np.asarray(phases, dtype=np.float32)
    if phase.ndim != 1 or len(phase) == 0 or not np.isfinite(phase).all():
        raise ValueError("Supply a nonempty finite list of protocol phases.")
    if ((phase < 0) | (phase > 1)).any():
        raise ValueError("Protocol phase must lie in [0,1]; extrapolation is unsupported.")
    with np.load(graph, allow_pickle=False) as archive:
        # Inference intentionally never accesses target y, tracked displacement,
        # observed masks, future images or future graph topology.
        x0 = archive["x0"].copy()
        edge_index = archive["edge_index"].copy()
        initial = archive["initial_pos_px"].copy()
        node_ids = archive["node_ids"].copy()
        scale = float(archive["scale_px"])
    if x0.ndim != 2 or x0.shape[1] != 2 or len(x0) == 0 or not np.isfinite(x0).all():
        raise ValueError("x0 must contain finite [N,2] normalized initial coordinates.")
    if (initial.shape != x0.shape or not np.isfinite(initial).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Initial pixel coordinates and positive scale_px are required.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E].")
    if not np.issubdtype(edge_index.dtype, np.integer):
        raise ValueError("edge_index must have integer dtype.")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(x0)):
        raise ValueError("edge_index refers to an absent node.")
    if node_ids.ndim != 1 or len(node_ids) != len(x0) or len(np.unique(node_ids)) != len(x0):
        raise ValueError("node_ids must contain one unique identifier per initial node.")
    model, metadata = load_checkpoint(str(checkpoint))
    with torch.no_grad():
        prediction = model(torch.as_tensor(x0, dtype=torch.float32),
                           torch.as_tensor(edge_index, dtype=torch.long),
                           torch.from_numpy(phase)).numpy()
    return {"phase": phase, "node_ids": node_ids, "initial_pos_px": initial,
            "displacement_normalized": prediction,
            "displacement_px": prediction * scale,
            "position_px": initial[None] + prediction * scale,
            "edge_index": edge_index, "scale_px": np.asarray(scale),
            "model_kind": np.asarray(metadata["config"]["kind"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph", required=True, help="NPZ containing initial geometry and edges")
    parser.add_argument("--phase", "--progress", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--output", default="results/prediction.npz")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = predict(args.checkpoint, args.graph, args.phase)
    output = Path(args.output)
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "node_id", "initial_x_px", "initial_y_px",
                         "pred_dx_px", "pred_dy_px", "pred_x_px", "pred_y_px"])
        for i, phase in enumerate(result["phase"]):
            for j, node_id in enumerate(result["node_ids"]):
                writer.writerow([float(phase), int(node_id), *map(float, result["initial_pos_px"][j]),
                                 *map(float, result["displacement_px"][i, j]),
                                 *map(float, result["position_px"][i, j])])
    print(json.dumps({"output": str(output), "phases": len(result["phase"]),
                      "nodes": len(result["node_ids"]), "model_kind": str(result["model_kind"])}))


if __name__ == "__main__":
    main()
