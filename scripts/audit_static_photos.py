#!/usr/bin/env python
"""Reproduce the 22-photo acquisition-domain audit without treating photos as tests.

Examples (run from the repository root):
  python scripts/audit_static_photos.py --images data/raw/static_photos --download
  python scripts/audit_static_photos.py --images /path/to/Opt_CB_deformed \
      --output-report reports/static_photo_curation_reproduced.json \
      --figure reports/figures/static_photo_curation_reproduced.jpg

The source manifest pins repository commits and Git blob SHAs. --download is
optional and fetches only the 22 pinned photos, without cloning the repository.
The quarantine decisions are the previously completed visual review of these
exact images; this script reproduces candidate extraction and its audit, not
an automatic adjudicator for arbitrary new photographs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnn_cb.image_graph import ExtractionConfig, extract_graph, save_overlay


# Freeze the settings actually used in the recorded audit. The extractor's
# camera-sequence default threshold later changed to 20, which is intentionally
# not inherited here.
AUDIT_CONFIG = dict(threshold=30.0, gaussian_kernel=7, gaussian_sigma=1.0,
                    median_kernel=5, min_component_pixels=150,
                    min_component_height=9, min_spur_length=8.0,
                    border_tolerance=12.0, max_track_distance=10.0, max_gap=2,
                    ambiguity_margin=1.5, min_frame_coverage=0.65, roi=None)
REVIEW_FLAGS = ["unpaired_static_photo_no_reference_frame",
                "multiple_strips_or_repeated_views", "out_of_plane_perspective",
                "nonuniform_lighting_and_background_texture",
                "automatic_skeleton_not_confirmed_material_topology"]
REJECTION_REASON = (
    "These are perspective/group photographs of damaged strips, not the registered "
    "side-on single-specimen compression frames used by the displacement task. "
    "Full-image threshold skeletons follow illumination, paper surfaces and "
    "background structures, so physical nodes/edges and material identity are not "
    "trustworthy. A nonempty graph is not acceptance."
)


def git_blob_sha(payload: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def locate_photo(directory: Path, record: dict, download: bool) -> Path:
    relative = Path(record["path"])
    candidates = [directory / relative, directory / relative.name]
    for candidate in candidates:
        if candidate.is_file():
            if git_blob_sha(candidate.read_bytes()) != record["sha"]:
                raise ValueError(f"Source Git blob SHA mismatch: {candidate}")
            return candidate
    if not download:
        raise FileNotFoundError(f"Missing pinned photo {relative.name}; supply its folder or use --download")
    url = "https://raw.githubusercontent.com/" + record["repository"] + "/" + record["commit_sha"] + "/" + quote(record["path"], safe="/")
    with urlopen(url, timeout=90) as response:
        payload = response.read()
    if git_blob_sha(payload) != record["sha"]:
        raise ValueError(f"Downloaded source Git blob SHA mismatch: {record['path']}")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / relative.name
    destination.write_bytes(payload)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=ROOT / "reports/public_image_inventory.json")
    parser.add_argument("--output-report", type=Path, default=ROOT / "reports/static_photo_curation.json")
    parser.add_argument("--figure", type=Path, default=ROOT / "reports/figures/static_photo_curation.jpg")
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned public JPEGs and verify Git blob SHA")
    args = parser.parse_args()
    photos = json.loads(args.inventory.read_text())["static_photos"]
    if len(photos) != 22 or any(p["repository"] != "ricardofitas/Opt_CB" or
                               p["commit_sha"] != "8a3941c0132e1b404b8d42ef48509d91145ece7e" for p in photos):
        raise ValueError("This replay applies only to the exact reviewed 22-photo Opt_CB corpus")
    records = []
    canvas = Image.new("RGB", (1600, 65 + math.ceil(len(photos) / 4) * 220), (247, 246, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "22 additional static photos: automated material-graph extraction audit (Otsu)", fill=(30, 40, 50))
    draw.text((20, 35), "All quarantined: perspective / multiple strips / no matched reference or displacement targets. Yellow nodes and green edges.", fill=(30, 40, 50))
    with tempfile.TemporaryDirectory(prefix="gnn_cb_static_audit_") as temporary:
        for index, photo in enumerate(photos):
            path = locate_photo(args.images, photo, args.download)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode photo {path}")
            height, width = image.shape[:2]
            scale = min(1.0, 800.0 / max(height, width))
            small = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            attempts = []
            for mode in ("fixed", "otsu"):
                config = ExtractionConfig(threshold_mode=mode, **AUDIT_CONFIG)
                graph = extract_graph(small, config)
                attempts.append({"threshold_mode": mode, "config": asdict(config),
                                 "metadata": graph["metadata"], "nodes_xy_scaled": graph["nodes_xy"].tolist(),
                                 "edges": [{"source": edge["source"], "target": edge["target"],
                                            "length_px_scaled": edge["length_px"]} for edge in graph["edges"]]})
                if mode == "otsu":
                    overlay = Path(temporary) / f"photo_{index:02d}_otsu_overlay.jpg"
                    save_overlay(small, graph, overlay)
                    with Image.open(overlay) as rendered:
                        thumbnail = rendered.convert("RGB")
                    thumbnail.thumbnail((341, 192))
                    x, y = (index % 4) * 400, 65 + (index // 4) * 220
                    canvas.paste(thumbnail, (x, y))
                    draw.text((x + 5, y + 198), f"Photo {index:02d}: QUARANTINED", fill=(150, 60, 40))
            records.append({"photo_index": index, "source_path": photo["path"],
                            "source_blob_sha": photo["sha"], "original_size_wh": [width, height],
                            "audit_image_scale": scale, "attempts": attempts,
                            "accepted_as_training_sequence": False, "accepted_as_physical_graph": False,
                            "disposition": "quarantine", "review_flags": REVIEW_FLAGS,
                            "independent_specimen_id": None})
            print(f"Photo {index + 1}/22: both thresholds extracted; recorded visual-review quarantine retained.", flush=True)
    result = {
        "source_repository": "ricardofitas/Opt_CB", "source_commit_sha": photos[0]["commit_sha"],
        "method": "Graph-CB-style blur, threshold, component filtering, skeletonization and contraction applied to each static photo at maximum 800 px. Two threshold policies: fixed 30 and Otsu. Candidate graphs quarantined after acquisition-domain review.",
        "summary": {"photos_processed": len(records), "threshold_attempts": 2 * len(records),
                    "candidate_graphs_extracted": sum(a["metadata"]["n_nodes"] > 0 for record in records for a in record["attempts"]),
                    "physical_graphs_accepted": 0, "training_sequences_accepted": 0,
                    "photos_quarantined": len(records), "independent_specimen_count": None},
        "reason": REJECTION_REASON, "records": records,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result, indent=2) + "\n")
    canvas.save(args.figure, quality=90)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
