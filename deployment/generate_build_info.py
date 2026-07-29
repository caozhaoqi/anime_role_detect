#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建期生成 src/core/version/_build_info.py，注入 BUILD_TIME 与 GIT_COMMIT。

本地开发环境无此文件时，src/core/version 会自动回退到实时 git 估算，
因此该脚本仅在镜像构建期需要（由 deployment/Dockerfile.base 调用）。
"""
import os
import subprocess
import datetime

TARGET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "core", "version", "_build_info.py",
)


def _git_commit() -> str:
    val = os.environ.get("GIT_COMMIT")
    if val:
        return val
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    build_time = (
        os.environ.get("BUILD_TIME")
        or os.environ.get("BUILD_DATE")
        or datetime.datetime.utcnow().isoformat() + "Z"
    )
    commit = _git_commit()
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write("# 自动生成，请勿手改；由 deployment/generate_build_info.py 在镜像构建期写入。\n")
        f.write(f"BUILD_TIME = {build_time!r}\n")
        f.write(f"GIT_COMMIT = {commit!r}\n")
    print(f"[build-info] 写入 {TARGET}: BUILD_TIME={build_time} GIT_COMMIT={commit}")


if __name__ == "__main__":
    main()
