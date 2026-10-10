"""Export the original held-out GNN checkpoints for genuine browser inference.

Images and observations are display-only. The JavaScript model accepts only the
initial graph and phase. No new fitting or benchmark selection takes place.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gnn_cb.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web_demo" / "data"
SOURCE_COMMIT = "50f888bcddffe3284dae77cfba0769cbd54a62fb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(array, dtype="<f4"):
    a = np.asarray(array, dtype=dtype)
    return {"shape": list(a.shape), "dtype": dtype,
            "base64": base64.b64encode(a.tobytes(order="C")).decode("ascii")}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "source_commit": SOURCE_COMMIT,
                "seed": 11, "kind": "original_five_specimen_loso_gnn",
                "inference_inputs": ["initial normalized coordinates", "initial edges", "protocol phase"],
                "normalization": "initial graph height; image x-right/y-down",
                "images_are_model_inputs": False, "cases": []}
    reference = []
    for path in sorted((ROOT / "data/expanded").glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            source = {k: d[k].copy() for k in d.files}
        name = path.stem
        physical = str(source.get("physical_specimen_id", source["specimen_id"]))
        checkpoint = ROOT / f"results/expanded/five_specimens/test_{physical}/gnn_seed11.pt"
        model, meta = load_checkpoint(str(checkpoint))
        assert meta["held_out_physical_specimen"] == physical
        assert physical not in meta["training_physical_specimens"]
        assert physical not in meta["validation_physical_specimens"]
        metadata = json.loads(path.with_suffix(".json").read_text())
        phases = source["progress"]
        selected = np.flatnonzero((phases >= 0) & (phases <= 1))
        image_paths = metadata.get("source_images", [])
        if not image_paths:
            image_paths = [x["path"] for x in metadata["selected_source_images"]]
        images = []
        for image_path in image_paths:
            p = ROOT / "data/expanded" / image_path
            frame = int(p.stem.split("frame")[-1])
            with Image.open(p) as image:
                width, height = image.size
            images.append({"frame": frame, "phase": float(phases[frame]),
                           "width": width, "height": height, "sha256": sha(p),
                           "source_path": str(p.relative_to(ROOT)),
                           "url": f"https://raw.githubusercontent.com/ricardofitas/GNN-CB/{SOURCE_COMMIT}/{p.relative_to(ROOT)}"})
        case = {"schema_version": 1, "id": name, "physical_id": physical,
                "source_commit": SOURCE_COMMIT, "dataset_sha256": sha(path),
                "checkpoint_path": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha(checkpoint), "checkpoint_metadata": meta,
                "model": {"config": model.metadata()["config"],
                          "weights": {k: encoded(v.numpy()) for k, v in model.state_dict().items()}},
                "graph": {"x0": source["x0"].tolist(),
                          "edge_index": source["edge_index"].tolist(),
                          "initial_pos_px": source["initial_pos_px"].tolist(),
                          "node_ids": source["node_ids"].tolist(),
                          "scale_px": float(source["scale_px"])},
                "observations": {"frames": source["frame"][selected].tolist(),
                                 "phases": phases[selected].tolist(),
                                 "displacements": encoded(source["y"][selected]),
                                 "mask": encoded(source["mask"][selected], "u1")},
                "images": images}
        # Metadata has no tensors; model parameters are exported separately.
        case["checkpoint_metadata"] = {k: v for k, v in meta.items() if k != "state_dict"}
        out = OUT / f"{name}.json"
        out.write_text(json.dumps(case, separators=(",", ":"), allow_nan=False) + "\n")
        manifest["cases"].append({"id": name, "physical_id": physical,
                                  "path": f"data/{name}.json", "sha256": sha(out),
                                  "nodes": len(source["x0"]),
                                  "edges": source["edge_index"].shape[1] // 2})
        # All archived compression phases plus non-archived phases prove this is
        # the continuous network, not interpolation between saved predictions.
        probe_phases = np.unique(np.r_[phases[selected], [.1234567, .3812345, .8765432]]).astype("float32")
        with torch.no_grad():
            y = model(torch.from_numpy(source["x0"]), torch.from_numpy(source["edge_index"]),
                      torch.from_numpy(probe_phases)).numpy()
        reference.append({"id": name, "phases": probe_phases.tolist(), "prediction": y.tolist()})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Reference is test evidence, never consumed by the UI/inference module.
    cache = ROOT / "results/diagnosis/cache/web_demo"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "pytorch_reference.json").write_text(json.dumps(reference, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(manifest["cases"]), "seed": 11,
                      "held_out_verified": True, "reference": str(cache / "pytorch_reference.json")}))


if __name__ == "__main__":
    main()
