#!/usr/bin/env python3
"""Restore exact graph NPZ archives from checked, content-hashed binary parts.

Uses only the Python standard library. Every part's size/SHA256 and the complete
NPZ's size/SHA256 must match before the destination is atomically replaced.
    python scripts/restore_graph_archives.py
    python scripts/restore_graph_archives.py --verify-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_archive(manifest_path: Path, archive: dict, output_dir: Path | None = None,
                    verify_only: bool = False) -> dict:
    manifest_path = manifest_path.resolve()
    filename = archive["archive_filename"]
    if Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("Invalid archive output filename")
    output_dir = output_dir or manifest_path.parent
    destination = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    stream = None
    if not verify_only:
        handle, temporary_name = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=output_dir)
        temporary = Path(temporary_name)
        stream = os.fdopen(handle, "wb")
    whole_digest, total = hashlib.sha256(), 0
    try:
        for expected_index, part in enumerate(archive["parts"], 1):
            if part["index"] != expected_index:
                raise ValueError("Part order is not contiguous")
            relative = Path(part["filename"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid relative part filename")
            source = manifest_path.parent / relative
            if source.stat().st_size != part["bytes"]:
                raise ValueError(f"Part size mismatch: {relative}")
            part_digest, part_bytes = hashlib.sha256(), 0
            with source.open("rb") as source_stream:
                for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                    part_digest.update(chunk)
                    whole_digest.update(chunk)
                    part_bytes += len(chunk)
                    total += len(chunk)
                    if stream is not None:
                        stream.write(chunk)
            if part_bytes != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
                raise ValueError(f"Part SHA256 mismatch: {relative}")
        if total != archive["archive_bytes"] or whole_digest.hexdigest() != archive["archive_sha256"]:
            raise ValueError(f"Whole-archive size/SHA256 mismatch: {filename}")
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            stream = None
            # No destination is touched before both the parts and whole archive
            # have passed their independent expected checksum checks.
            temporary.replace(destination)
            temporary = None
        elif destination.exists() and (destination.stat().st_size != total or
                                      sha256_file(destination) != archive["archive_sha256"]):
            raise ValueError(f"Existing restored archive differs: {destination}")
        return {"specimen_id": archive["specimen_id"], "archive": str(destination),
                "parts": len(archive["parts"]), "bytes": total,
                "sha256": whole_digest.hexdigest(), "all_part_hashes_verified": True,
                "whole_archive_hash_verified": True, "restored": not verify_only}
    finally:
        if stream is not None:
            stream.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    default_manifest = Path(__file__).resolve().parents[1] / "data/automatic/graphs/archive_parts_manifest.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--specimens", nargs="+")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("format_version") != "gnn_cb_binary_parts_v1":
        raise ValueError("Unknown parts manifest format")
    archives = [row for row in manifest["archives"] if not args.specimens or row["specimen_id"] in args.specimens]
    if args.specimens and set(args.specimens) - {row["specimen_id"] for row in archives}:
        raise ValueError("Selected specimen missing from parts manifest")
    results = []
    for archive in archives:
        result = restore_archive(args.manifest, archive, args.output_dir, args.verify_only)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest), "archives": results,
            "all_part_hashes_verified": all(row["all_part_hashes_verified"] for row in results),
            "all_whole_archive_hashes_verified": all(row["whole_archive_hash_verified"] for row in results)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
