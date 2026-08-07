"""Scan final_dataset for corrupt images and quarantine them (non-destructive move).

Run:
    .venv/bin/python scripts/_scan_corrupt.py
Moves any file that fails PIL verify() into data/_quarantine_corrupt_YYYYMMDD/
preserving the relative path, so the action is fully reversible.

Strategy: only call verify() (reads the file header, cheap, no pixel decode) so
oversized images never get loaded into memory. Size is read from the header
(im.size) without decoding pixels, purely for reporting.
"""
import os
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")
DATA = ROOT / "data/final_dataset"
QUAR = ROOT / "data/_quarantine_corrupt_20260804"
QUAR.mkdir(parents=True, exist_ok=True)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

bad = []
oversize = 0
total = 0
for dirpath, dirnames, filenames in os.walk(DATA):
    for fn in filenames:
        p = Path(dirpath) / fn
        if p.suffix.lower() not in EXTS:
            continue
        total += 1
        try:
            with Image.open(p) as im:
                w, h = im.size  # header only, no pixel decode
                if w * h > 50_000_000:
                    oversize += 1
                im.verify()  # validates structure; raises on corruption
        except Exception as e:  # noqa: BLE001 - any decode failure = corrupt
            bad.append((p, repr(e)))

print(f"SCANNED={total} CORRUPT={len(bad)} OVERSIZE(>50Mpx)={oversize}")
for p, e in bad:
    rel = p.relative_to(DATA)
    dst = QUAR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dst))
    print(f"MOVED {p} -> {dst}  ({e})")

# Remove now-empty directories left behind in final_dataset
emptied = 0
for dirpath, dirnames, filenames in os.walk(DATA, topdown=False):
    if not dirnames and not filenames:
        try:
            os.rmdir(dirpath)
            emptied += 1
        except OSError:
            pass
print(f"REMOVED_EMPTY_DIRS={emptied}")
print("DONE")
