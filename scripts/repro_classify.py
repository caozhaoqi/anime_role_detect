#!/usr/bin/env python3
"""隔离复现 /api/classify 崩溃：用 TestClient 进程内调用，捕获完整异常（含 BaseException）。"""
import os, sys, traceback

# 让 model-service HTTP 调用指向真实运行的服务
os.environ.setdefault("MODEL_SERVICE_URL", "http://127.0.0.1:8000")
os.environ.setdefault("USE_MODEL_SERVICE", "True")

try:
    from fastapi.testclient import TestClient
except Exception as e:
    print("TestClient 不可用:", e)
    sys.exit(2)

# 找一张图片
img = None
for cand in os.popen("find data -type f \\( -name '*.jpg' -o -name '*.png' \\) 2>/dev/null | grep -v _mislabeled | head -1").read().split():
    img = cand.strip(); break
print("测试图片:", img)

try:
    from src.api.app import app
    print("[OK] app 导入成功")
except Exception as e:
    print("[FAIL] app 导入失败:", repr(e))
    traceback.print_exc()
    sys.exit(3)

import io
TRACE = "/tmp/repro_trace.txt"
def log(*a):
    with open(TRACE, "a") as fh:
        fh.write(" ".join(str(x) for x in a) + "\n")

try:
    with TestClient(app) as client:
        with open(img, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            data = {"model_name": "efficientnet_b3", "use_deepdanbooru": "true"}
            log(">>> 调用 POST /api/classify ...")
            try:
                resp = client.post("/api/classify", files=files, data=data, timeout=90)
                log(f"<<< HTTP {resp.status_code}")
                log("BODY:", resp.text[:800])
            except BaseException as e:
                log(f"\n!!! 捕获到顶层异常: {type(e).__name__}: {e}")
                log(traceback.format_exc())
                sys.exit(1)
except BaseException as e:
    log(f"\n!!! TestClient 上下文顶层异常: {type(e).__name__}: {e}")
    log(traceback.format_exc())
    sys.exit(1)

log("\n[OK] 未崩溃，分类返回正常")
