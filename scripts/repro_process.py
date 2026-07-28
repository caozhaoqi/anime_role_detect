#!/usr/bin/env python3
"""隔离调用 process_single_image，捕获顶层异常，二分 use_deepdanbooru。"""
import os, sys, traceback, asyncio, faulthandler
faulthandler.enable()
from io import BytesIO

os.environ.setdefault("MODEL_SERVICE_URL", "http://127.0.0.1:8000")
os.environ.setdefault("USE_MODEL_SERVICE", "True")

TRACE = "/tmp/repro_proc.txt"
def log(*a):
    with open(TRACE, "a") as fh:
        fh.write(" ".join(str(x) for x in a) + "\n")

img = None
for cand in os.popen("find data -type f \\( -name '*.jpg' -o -name '*.png' \\) 2>/dev/null | grep -v _mislabeled | head -1").read().split():
    img = cand.strip(); break
log("IMG:", img)

from starlette.datastructures import UploadFile

async def run(use_dd):
    try:
        from src.services.processor.image_processor import process_single_image
        data = open(img, "rb").read()
        f = UploadFile(filename="test.jpg", file=BytesIO(data))
        log(f"\n=== use_deepdanbooru={use_dd} ===")
        log("调用 process_single_image ...")
        try:
            result = await process_single_image(
                f, "efficientnet_b3", False, False, True, True, use_dd
            )
            log("OK 返回 keys:", list(result.keys())[:15])
            log("role_info:", str(result.get("role_info"))[:200])
        except BaseException as e:
            log(f"!!! 顶层异常: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            return
    except BaseException as e:
        log(f"!!! import/setup 异常: {type(e).__name__}: {e}")
        log(traceback.format_exc())

async def main():
    await run(False)
    await run(True)

try:
    asyncio.run(main())
    log("\n[DONE]")
except BaseException as e:
    log(f"!!! asyncio.run 顶层: {type(e).__name__}: {e}")
    log(traceback.format_exc())
