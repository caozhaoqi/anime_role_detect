#!/usr/bin/env python3
"""全量 API 功能测试 - 针对 supervisord 拉起的 5 个在线服务。
使用 .venv 的 requests。覆盖 api-service/api-gateway/model-service/multimedia-service/search-service。
"""
import requests
import json
import os
import time
from datetime import datetime

BASE = {
    "api": "http://127.0.0.1:8001",
    "gateway": "http://127.0.0.1:8080",
    "model": "http://127.0.0.1:8000",
    "multimedia": "http://127.0.0.1:8002",
    "search": "http://127.0.0.1:8003",
}
IMG = None
for cand in os.popen("find data -type f \\( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \\) 2>/dev/null | grep -v _mislabeled | head -1").read().split():
    IMG = cand.strip()
    break

TOKEN = None
RESULTS = []  # (group, method, url, status, note)

sess = requests.Session()
sess.trust_env = False  # 避免代理干扰


def rec(group, method, url, status, note=""):
    RESULTS.append((group, method, url, status, note))
    tag = "PASS" if 200 <= status < 300 else ("4xx" if 400 <= status < 500 else "FAIL")
    print(f"  [{tag}] {group:11} {method:4} {url} -> {status}  {note}")


def call(group, method, base, path, **kw):
    url = base + path
    try:
        headers = kw.pop("headers", {})
        if TOKEN and "auth" in kw:
            headers["Authorization"] = f"Bearer {TOKEN}"
            kw.pop("auth")
        if "files" in kw:
            r = sess.request(method, url, headers=headers, timeout=90, **kw)
        else:
            r = sess.request(method, url, headers=headers, timeout=30, **kw)
        rec(group, method, url, r.status_code, (r.text[:120].replace("\n", " ") if r.status_code >= 400 else ""))
        return r
    except requests.exceptions.ConnectionError as e:
        rec(group, method, url, 0, f"连接失败: {str(e)[:80]}")
        return None
    except Exception as e:
        rec(group, method, url, 0, f"异常: {type(e).__name__} {str(e)[:80]}")
        return None


print("=" * 80)
print("全量 API 功能测试  ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"测试图片: {IMG}")
print("=" * 80)

# ---------- 1. 认证闭环 (api-service) ----------
print("\n### 1. 认证 API (api-service :8001) ###")
call("auth", "GET", BASE["api"], "/api/health")
call("auth", "GET", BASE["api"], "/live")
call("auth", "GET", BASE["api"], "/ready")
call("auth", "GET", BASE["api"], "/api/health/detailed")
call("auth", "GET", BASE["api"], "/api/monitoring")

# 注册新用户（避免与上次重复）
u = f"aptest_{int(time.time())%100000}"
pw = "Ap@12345"
r = call("auth", "POST", BASE["api"], "/api/auth/register",
         data={"username": u, "password": pw, "email": f"{u}@ex.com"})
if r and r.status_code == 200:
    try:
        TOKEN = r.json().get("data", {}).get("access_token") or r.json().get("access_token")
    except Exception:
        pass
    if not TOKEN:
        # 注册响应可能不含 token，用 login 拿
        pass

# 登录
r = call("auth", "POST", BASE["api"], "/api/auth/login", data={"username": u, "password": pw})
if r and r.status_code == 200:
    try:
        j = r.json()
        TOKEN = (j.get("data", {}) or {}).get("access_token") or j.get("access_token") or TOKEN
    except Exception:
        pass
print(f"  >> TOKEN 获取: {'成功' if TOKEN else '失败'}")

# 错误密码（期望 4xx）
call("auth", "POST", BASE["api"], "/api/auth/login", data={"username": u, "password": "wrong"})
# refresh（用 refresh_token 若有）
call("auth", "GET", BASE["api"], "/api/auth/me", auth=True)
call("auth", "GET", BASE["api"], "/api/admin/test", auth=True)

# ---------- 2. 模型管理 API (api-service) ----------
print("\n### 2. 模型管理 API (api-service) ###")
call("models", "GET", BASE["api"], "/api/models")
call("models", "GET", BASE["api"], "/api/model-versions")
call("models", "GET", BASE["api"], "/api/multi-model/config")

# ---------- 3. 分类/识别 API (api-service + model-service) ----------
print("\n### 3. 分类/识别 API ###")
if IMG:
    # model-service
    call("model", "POST", BASE["model"], "/api/classify",
         files={"file": open(IMG, "rb")}, data={"model_name": "efficientnet_b3"})
    call("model", "POST", BASE["model"], "/api/model/predict",
         files={"file": open(IMG, "rb")})
    call("model", "POST", BASE["model"], "/api/model/extract",
         files={"file": open(IMG, "rb")})
    call("model", "POST", BASE["model"], "/api/model/detect-yolo",
         files={"file": open(IMG, "rb")})
    call("model", "POST", BASE["model"], "/api/model/detect-multiple",
         files={"file": open(IMG, "rb")})
    # api-service 分类（走内部调用 model-service）
    call("classify", "POST", BASE["api"], "/api/classify",
         files={"file": open(IMG, "rb")}, data={"model_name": "efficientnet_b3"})
    call("classify", "POST", BASE["api"], "/api/classify/multi-role",
         files={"file": open(IMG, "rb")})
    r = call("classify", "POST", BASE["api"], "/api/batch_classify")
call("model", "GET", BASE["model"], "/api/health")
call("model", "GET", BASE["model"], "/live")
call("model", "GET", BASE["model"], "/ready")
call("model", "GET", BASE["model"], "/model_service")

# ---------- 4. tracing / 历史 / misc (api-service) ----------
print("\n### 4. Tracing / 历史 / 杂项 API (api-service) ###")
call("tracing", "GET", BASE["api"], "/api/tracing/stats")
call("tracing", "GET", BASE["api"], "/api/tracing/traces")
call("tracing", "GET", BASE["api"], "/api/tracing/health")
call("history", "GET", BASE["api"], "/api/history")
call("misc", "POST", BASE["api"], "/api/feedback", json={"content": "test feedback"})
call("misc", "GET", BASE["api"], "/api/config")
call("misc", "GET", BASE["api"], "/api/docs/info")

# ---------- 5. 清洗 API (api-service) ----------
print("\n### 5. 数据清洗 API (api-service) ###")
call("cleaning", "GET", BASE["api"], "/api/cleaning/config/default")
call("cleaning", "GET", BASE["api"], "/api/cleaning/tasks")
call("cleaning", "GET", BASE["api"], "/api/cleaning/progress")
call("cleaning", "GET", BASE["api"], "/api/cleaning/browse")

# ---------- 6. 搜索 API (api-service + search-service) ----------
print("\n### 6. 搜索 API ###")
call("search", "GET", BASE["api"], "/api/search/stats")
call("search", "GET", BASE["api"], "/api/search/health")
call("search-svc", "GET", BASE["search"], "/api/health")
call("search-svc", "GET", BASE["search"], "/api/search/stats")
call("search-svc", "GET", BASE["search"], "/api/search/queue-status")
if IMG:
    call("search-svc", "POST", BASE["search"], "/api/search/image",
         files={"file": open(IMG, "rb")})
    call("search-svc", "POST", BASE["search"], "/api/search/build-index")

# ---------- 7. 多媒体服务 (multimedia-service) ----------
print("\n### 7. 多媒体服务 API (multimedia :8002) ###")
call("mm", "GET", BASE["multimedia"], "/api/health")
call("mm", "GET", BASE["multimedia"], "/health")
call("mm", "GET", BASE["multimedia"], "/info")
call("mm", "GET", BASE["multimedia"], "/search/stats")
call("mm", "GET", BASE["multimedia"], "/video/stats")
if IMG:
    call("mm", "POST", BASE["multimedia"], "/search/image", files={"file": open(IMG, "rb")})

# ---------- 8. 视频识别 API ----------
print("\n### 8. 视频识别 API ###")
call("video", "POST", BASE["api"], "/api/video/recognize")  # 无视频，期望 4xx 校验
call("video", "GET", BASE["api"], "/api/video/recognize/status")
call("video-svc", "POST", BASE["multimedia"], "/video/extract")  # 无视频，期望 4xx
call("video-svc", "POST", BASE["search"], "/api/video/recognize")

# ---------- 9. 网关 API (api-gateway) ----------
print("\n### 9. API 网关 API (gateway :8080) ###")
call("gw", "GET", BASE["gateway"], "/")
call("gw", "GET", BASE["gateway"], "/api/health")
call("gw", "GET", BASE["gateway"], "/health")
call("gw", "GET", BASE["gateway"], "/api/services")
call("gw", "GET", BASE["gateway"], "/monitor/services")
call("gw", "GET", BASE["gateway"], "/monitor/tracing/stats")
call("gw", "GET", BASE["gateway"], "/monitor/cleaning/progress")
call("gw", "GET", BASE["gateway"], "/logs/stats")
call("gw", "GET", BASE["gateway"], "/logs/services")
# 代理：经网关访问 api-service 的 health
call("gw-proxy", "GET", BASE["gateway"], "/api/health")

# ---------- 汇总 ----------
print("\n" + "=" * 80)
print("汇总")
print("=" * 80)
from collections import Counter, defaultdict
by_group = defaultdict(list)
for g, m, u, s, n in RESULTS:
    by_group[g].append((m, u, s))
total = len(RESULTS)
pass_n = sum(1 for x in RESULTS if 200 <= x[3] < 300)
fail_n = sum(1 for x in RESULTS if x[3] == 0 or x[3] >= 500)
c4 = sum(1 for x in RESULTS if 400 <= x[3] < 500)
print(f"总计: {total}  2xx通过: {pass_n}  4xx(校验/鉴权预期): {c4}  失败(0/5xx): {fail_n}")
print()
for g in sorted(by_group):
    s = by_group[g]
    pg = sum(1 for x in s if 200 <= x[2] < 300)
    fg = sum(1 for x in s if x[2] == 0 or x[2] >= 500)
    print(f"  {g:11} {len(s):2} 项 | 通过 {pg:2} | 失败 {fg:2}")

# 失败明细
print("\n--- 失败项明细 (status 0 或 5xx) ---")
failed = [x for x in RESULTS if x[3] == 0 or x[3] >= 500]
if failed:
    for g, m, u, s, n in failed:
        print(f"  {g:11} {m:4} {u} -> {s}  {n}")
else:
    print("  无失败项")

# 保存 JSON
out = {
    "time": datetime.now().isoformat(),
    "total": total, "pass": pass_n, "client_4xx": c4, "fail": fail_n,
    "results": [{"group": g, "method": m, "url": u, "status": s, "note": n} for g, m, u, s, n in RESULTS],
}
with open("outputs/api_test_result.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n报告已保存: outputs/api_test_result.json")
