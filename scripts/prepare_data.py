#!/usr/bin/env python3
"""Prepare checksum-pinned, specimen-separated image-graph training data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn_cb.data import download_sources, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--download", action="store_true", help="Download the two pinned public source archives")
    args = parser.parse_args()
    if args.download:
        download_sources(args.source_dir)
    manifest = prepare_dataset(args.source_dir, args.output_dir)
    print(json.dumps({"independent_specimens": manifest["independent_specimens"], "output": str(args.output_dir), "specimens": [{k: row[k] for k in ("specimen_id", "n_nodes", "n_frames", "compression_coverage")} for row in manifest["specimens"]]}, indent=2))


if __name__ == "__main__":
    main()
