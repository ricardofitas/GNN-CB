"""Prepare the five-specimen audited dataset from locally restored sources.

Authenticated source restoration: check out ricardofitas/uncertain_cb at the
commit documented in gnn_cb/import_expanded.py, then supply the checkout root
or its Taieb directory. The bundled source_tree.json pins the required files.
"""
from __future__ import annotations

import argparse
import json

from gnn_cb.import_expanded import prepare_expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-tree", default="data/expanded/source_tree.json")
    parser.add_argument("--base-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/expanded")
    parser.add_argument("--audit", default="reports/expanded_data_audit.json")
    args = parser.parse_args()
    result = prepare_expanded(args.source_root, args.base_dir, args.output_dir, args.source_tree, args.audit)
    print(json.dumps({"independent_specimens": result["independent_specimens"], **result["totals"]}, indent=2))


if __name__ == "__main__":
    main()
