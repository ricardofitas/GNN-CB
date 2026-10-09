#!/usr/bin/env python
"""Curate image folders or manifest sequences into structural graphs and tracks.

Examples:
  python scripts/curate_images.py --images /data/Probe3 --specimen Probe3 --output data/curated
  python scripts/curate_images.py --manifest data/image_sequences.json --output data/curated

Manifest: {"sequences": [{"specimen_id": "Probe3", "image_directory": "...",
"glob": "*.jpg", "lineage_id": "physical_test_3", "source": {"repository": "..."},
"compression_window": [12, 180], "config": {"threshold": 20}}]}.
Omit compression_window to withhold phase-labelled training exports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn_cb.image_graph import ExtractionConfig, curate_sequence, natural_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--specimen")
    parser.add_argument("--lineage-id")
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=20)
    parser.add_argument("--threshold-mode", choices=["fixed", "otsu"], default="fixed")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--compression-window", type=int, nargs=2)
    parser.add_argument("--max-track-distance", type=float, default=10)
    args = parser.parse_args()
    if args.manifest:
        sequences = json.loads(args.manifest.read_text())["sequences"]
        base = args.manifest.resolve().parent
    else:
        if not args.specimen:
            parser.error("--specimen is required with --images")
        sequences = [{"specimen_id": args.specimen, "image_directory": str(args.images.resolve()),
                      "glob": args.glob, "lineage_id": args.lineage_id or args.specimen,
                      "compression_window": args.compression_window,
                      "config": {"threshold": args.threshold, "threshold_mode": args.threshold_mode,
                                 "roi": args.roi, "max_track_distance": args.max_track_distance}}]
        base = Path.cwd()
    summaries = []
    for sequence in sequences:
        directory = Path(sequence["image_directory"])
        if not directory.is_absolute():
            directory = base / directory
        paths = sorted(directory.glob(sequence.get("glob", "*.jpg")), key=natural_key)
        if sequence.get("stride", 1) != 1:
            raise ValueError("Subsampling requires independently indexed compression labels; curate full sequences instead")
        config = ExtractionConfig(**sequence.get("config", {}))
        summary = curate_sequence(paths, args.output, sequence["specimen_id"], config,
            compression_window=sequence.get("compression_window"), source=sequence.get("source"),
            lineage_id=sequence.get("lineage_id"))
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    (args.output / "curation_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
