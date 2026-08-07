"""Quarantine oversized-but-valid images (>50M pixels) out of final_dataset.

Why: oversized images cannot be safely decoded during training. Pillow either
raises DecompressionBomb or the process OOMs -- and an OOM is a SIGKILL from the
OS, NOT catchable by Python try/except. So they must be removed from the
training set before training. Corrupt leftovers (verify fails) are also moved,
just in case. Fully reversible: kept under data/_quarantine_corrupt_20260804/.
"""
import os
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")
DATA = ROOT / "data/final_dataset"
QUAR = ROOT / "data/_quarantine_corrupt_20260804"
OVER = QUAR / "oversize"
OVER.mkdir(parents=True, exist_ok=True)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
THRESH = 50_000_000  # pixels (width * height)

moved_over = 0
moved_corrupt = 0
for dirpath, dirnames, filenames in os.walk(DATA):
    for fn in filenames:
        p = Path(dirpath) / fn
        if p.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size  # header only, cheap
                im.verify()     # raises on corruption
            if w * h > THRESH:
                dst = OVER / p.relative_to(DATA)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst))
                moved_over += 1
        except Exception:  # noqa: BLE001 - corrupt leftover, move it too
            dst = QUAR / p.relative_to(DATA)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            moved_corrupt += 1

# remove now-empty directories left behind in final_dataset
emptied = 0
for dirpath, dirnames, filenames in os.walk(DATA, topdown=False):
    if not dirnames and not filenames:
        try:
            os.rmdir(dirpath)
            emptied += 1
        except OSError:
            pass
print(f"MOVED_OVERSIZE={moved_over} MOVED_CORRUPT={moved_corrupt} REMOVED_EMPTY_DIRS={emptied}")
print("DONE")
