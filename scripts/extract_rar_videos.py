"""Verify/extract pinned multi-volume RAR camera recordings; optionally decode.

Restore Data/Videos from ricardofitas/ML_Corrugated_Boards at the source_commit
in reports/video_source_audit.json. No credentials or network requests are used.

Example:
  python scripts/extract_rar_videos.py --source-dir /path/to/Data/Videos \
      --output-dir /path/to/decoded --decode

Needs libarchive-c and the libarchive shared library. --decode also needs
opencv-python-headless. Frames are lossless PNG files; all source hashes are
checked before extraction and all video hashes before decoding.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


def file_fingerprint(path: Path) -> dict:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    return {"bytes": size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def verify_file(path: Path, reference: dict) -> dict:
    actual = file_fingerprint(path)
    for key in ("bytes", "sha256", "git_blob_sha1"):
        if key in reference and actual[key] != reference[key]:
            raise ValueError(f"{key} mismatch for {path.name}")
    return actual


def safe_basename(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.part[0-9]{2}\.rar|\.avi)", value):
        raise ValueError(f"Unexpected source filename: {value!r}")
    return value


def extract_video(source_dir: Path, output_dir: Path, video: dict, parts: list[dict]) -> dict:
    from libarchive import ffi
    from libarchive.read import ArchiveRead, new_archive_read

    filename = safe_basename(video["filename"])
    destination = output_dir / filename
    if destination.exists():
        verify_file(destination, video)
        return {"filename": filename, "reused_verified_video": True, **file_fingerprint(destination)}
    paths = [source_dir / safe_basename(part["filename"]) for part in parts]
    library = ctypes.CDLL(ffi.libarchive_path)
    open_multi = library.archive_read_open_filenames
    open_multi.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t]
    open_multi.restype = ctypes.c_int
    encoded = [os.fsencode(path) for path in paths]
    filenames = (ctypes.c_char_p * (len(paths) + 1))(*encoded, None)
    temporary = destination.with_suffix(".avi.partial")
    extracted = 0
    try:
        with new_archive_read() as pointer:
            status = open_multi(pointer, filenames, 65536)
            if status != 0:
                raise RuntimeError(f"libarchive open failed with status {status}")
            for entry in ArchiveRead(pointer):
                # Never use an archive member pathname to form an output path.
                member = PurePosixPath(entry.pathname.replace("\\", "/"))
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError("Unsafe member path in source archive")
                if not entry.isfile or member.name != filename or extracted:
                    raise ValueError(f"Unexpected archive member: {entry.pathname!r}")
                if entry.size != video["bytes"]:
                    raise ValueError("Unexpected uncompressed video size")
                with temporary.open("wb") as stream:
                    written = 0
                    for block in entry.get_blocks():
                        written += len(block)
                        if written > video["bytes"]:
                            raise ValueError("Uncompressed video exceeds pinned size")
                        stream.write(block)
                verify_file(temporary, video)
                extracted += 1
        if extracted != 1:
            raise ValueError("Expected exactly one regular AVI member")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"filename": filename, "reused_verified_video": False, **file_fingerprint(destination)}


def decode_video(path: Path, reference: dict, save_frames: bool) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {path.name}")
    expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    output = path.parent / path.stem
    if save_frames:
        output.mkdir(exist_ok=True)
    pixel_hash = hashlib.sha256()
    count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape != (reference["height"], reference["width"], 3):
                raise ValueError(f"Unexpected decoded image dimensions in {path.name}")
            pixel_hash.update(frame.tobytes())
            if save_frames and not cv2.imwrite(str(output / f"image{count:04d}.png"), frame):
                raise RuntimeError(f"Failed writing frame {count} of {path.name}")
            count += 1
    finally:
        capture.release()
    if count != expected_count or count != reference["expected_frames"]:
        raise ValueError(f"Incomplete decode: {count}/{reference['expected_frames']} frames")
    if pixel_hash.hexdigest() != reference["decoded_bgr_uint8_pixel_sha256"]:
        raise ValueError("Decoded pixels differ from the pinned audit; check decoder/version before use")
    return {"specimen": path.stem, "decoded_frames": count, "fps_container": fps,
            "decoded_bgr_uint8_pixel_sha256": pixel_hash.hexdigest(),
            "opencv_version": cv2.__version__, "png_frames_saved": save_frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports/video_source_audit.json")
    parser.add_argument("--specimen", action="append", help="Limit to a pinned recording; may be repeated")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true", help="Verify archives and any existing AVIs without extracting")
    parser.add_argument("--decode", action="store_true", help="Decode every frame and save lossless PNGs")
    parser.add_argument("--decode-check", action="store_true", help="Decode/hash every frame without saving PNGs")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.verify_only and (args.decode or args.decode_check):
        parser.error("--verify-only cannot be combined with decoding")
    manifest = json.loads(args.manifest.read_text())
    videos = [video for video in manifest["videos"]
              if not args.specimen or video["specimen"] in args.specimen]
    if not videos or (args.specimen and set(args.specimen) - {v["specimen"] for v in videos}):
        parser.error("Unknown or empty specimen selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verified_parts = []
    grouped_parts = {}
    for video in videos:
        parts = sorted([p for p in manifest["archive_volumes"] if p["specimen"] == video["specimen"]],
                       key=lambda p: p["filename"])
        expected_names = [f"{video['specimen']}.part{i:02d}.rar" for i in range(1, video["archive_volumes"] + 1)]
        if [p["filename"] for p in parts] != expected_names:
            raise ValueError(f"Noncontiguous archive volumes for {video['specimen']}")
        for part in parts:
            result = verify_file(args.source_dir / safe_basename(part["filename"]), part)
            verified_parts.append({"filename": part["filename"], **result})
        grouped_parts[video["specimen"]] = parts
    results = []
    if args.verify_only:
        for video in videos:
            path = args.output_dir / safe_basename(video["filename"])
            results.append({"filename": path.name, "present": path.exists(),
                            **(verify_file(path, video) if path.exists() else {})})
    else:
        def process(video: dict) -> dict:
            result = extract_video(args.source_dir, args.output_dir, video, grouped_parts[video["specimen"]])
            if args.decode or args.decode_check:
                result["decode"] = decode_video(args.output_dir / video["filename"], video, args.decode)
            return result
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(process, videos))
    audit = {"source_repository": manifest["source_repository"], "source_commit": manifest["source_commit"],
             "archive_volumes": verified_parts, "videos": results,
             "verify_only": args.verify_only}
    (args.output_dir / "restoration_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"verified_archive_volumes": len(verified_parts), "recordings": len(results),
                      "decoded_frames": sum(r.get("decode", {}).get("decoded_frames", 0) for r in results)}, indent=2))


if __name__ == "__main__":
    main()
