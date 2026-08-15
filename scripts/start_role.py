#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单仓多角色启动编排：本机资源受限（Mac MPS 统一内存）下，按需启停服务，
避免一次性全量拉起导致内存竞争 → OOM → supervisord 重启 → 正在轮询的作业被杀死。

用法：
    python scripts/start_role.py <role> [--stop-others] [--conf supervisord.conf]

角色（role）→ 程序组：
    web       仅网关 + 前端 + 健康检查（轻量，做 UI/联调）
    inference 模型/多媒体/搜索 + worker（识别链路，不含 t2i）
    t2i       仅 t2i-service（独立跑图生成/训练，独占内存，最稳）
    train     仅 t2i-service（与 t2i 同义，强调训练场景，内存最省）
    all       全部程序

说明：
    - 默认只 ``start`` 角色内程序，不动其它；``--stop-others`` 会先 ``stop all`` 再起角色组，
      确保内存彻底释放（推荐 train/t2i 前使用）。
    - 依赖 .venv 的 supervisorctl（与 supervisord.conf 同源鉴权）。
    - 纯编排脚本，不加载任何模型、不触发推理。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONF = os.path.join(PROJECT_ROOT, "supervisord.conf")
SUPERVISORCTL = os.path.join(PROJECT_ROOT, ".venv", "bin", "supervisorctl")

# 从 supervisord.conf 解析出的程序名（与文件保持一致，作为兜底）
KNOWN_PROGRAMS = [
    "model-service", "api-service", "api-gateway", "multimedia-service",
    "search-service", "t2i-service", "search-worker", "celery-worker",
    "inference-worker", "frontend", "health-check", "log-monitor",
    "resource-monitor",
]

ROLES = {
    "web": ["api-gateway", "frontend", "health-check"],
    "inference": [
        "model-service", "multimedia-service", "search-service",
        "inference-worker", "search-worker", "celery-worker",
    ],
    "t2i": ["t2i-service"],
    "train": ["t2i-service"],
    "all": list(KNOWN_PROGRAMS),
}


def _run(args: list) -> int:
    cmd = [SUPERVISORCTL, "-c", CONF] + args
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="单仓多角色启动编排")
    ap.add_argument("role", choices=list(ROLES.keys()), help="要启动的角色")
    ap.add_argument("--stop-others", action="store_true", help="先 stop all 再起角色组（彻底释放内存）")
    ap.add_argument("--conf", default=CONF, help="supervisord.conf 路径")
    args = ap.parse_args()

    if not os.path.exists(SUPERVISORCTL):
        print(f"[error] 找不到 supervisorctl: {SUPERVISORCTL}（请先创建 .venv）", file=sys.stderr)
        return 2
    if not os.path.exists(args.conf):
        print(f"[error] 找不到配置文件: {args.conf}", file=sys.stderr)
        return 2

    targets = ROLES[args.role]
    print(f"[role] {args.role} -> {targets}")

    if args.stop_others:
        print("[step] stop all（释放内存）")
        _run(["stop", "all"])
    else:
        # 先停掉角色组之外的程序，避免与本角色争内存
        others = [p for p in KNOWN_PROGRAMS if p not in targets]
        if others:
            print(f"[step] stop 非角色程序: {others}")
            _run(["stop", *others])

    print(f"[step] start {args.role} 程序组")
    rc = _run(["start", *targets])
    print(f"[done] role={args.role} 启动完成（退出码 {rc}）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
