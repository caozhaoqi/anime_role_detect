#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监视图片采集进度。
读取 collect_from_keywords.py 的最新日志，动态显示进展。

Usage:
    python3 scripts/data_collection/watch_collection.py
    python3 scripts/data_collection/watch_collection.py --log logs/keyword_collection_v3.log
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
LOG_DIR = PROJECT_ROOT / "logs"


def find_latest_log() -> Path:
    logs = sorted(LOG_DIR.glob("keyword_collection_*.log"))
    if not logs:
        print("未找到 keyword_collection_*.log 文件")
        sys.exit(1)
    return logs[-1]


def count_images(data_dir: Path) -> int:
    total = 0
    if not data_dir.exists():
        return 0
    for d in data_dir.iterdir():
        if d.is_dir():
            total += sum(1 for f in d.iterdir()
                         if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    return total


def count_roles(data_dir: Path) -> tuple:
    """返回 (有图片的角色数, 总角色目录数)"""
    total_dirs = 0
    nonempty = 0
    if not data_dir.exists():
        return 0, 0
    for d in data_dir.iterdir():
        if d.is_dir():
            total_dirs += 1
            n = sum(1 for f in d.iterdir()
                     if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
            if n > 0:
                nonempty += 1
    return nonempty, total_dirs


def parse_log(log_path: Path, history_lines: int) -> dict:
    """从日志中提取当前进度"""
    info = {
        "current_idx": 0,
        "total": "?",
        "current_name": "",
        "done_lines": [],
        "skip_exists_lines": [],
        "downloading": False,
        "last_update": "",
    }

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except FileNotFoundError:
        return info

    # 找最近的 [X/Y] 行
    progress_pattern = re.compile(r'\[(\d+)/(\d+)\]')
    matches = list(progress_pattern.finditer(content))
    if matches:
        last = matches[-1]
        info["current_idx"] = int(last.group(1))
        info["total"] = last.group(2)

    # 找当前正在处理的角色名
    lines = content.splitlines()
    for line in reversed(lines):
        m = re.match(r'\s*\[(\d+)/(\d+)\]\s+(.+?)\s+\(来源:', line)
        if m:
            info["current_name"] = m.group(3).strip()
            break

    # ✅ 成功 / 跳过行
    for line in lines:
        if re.match(r'\s*✅\s', line):
            info["done_lines"].append(line.strip())
        elif re.match(r'.*已有 \d+ ≥ \d+, 跳过', line):
            info["skip_exists_lines"].append(line.strip())

    # 尾部 history (下载中的最新行)
    tail_start = max(0, len(lines) - history_lines)
    info["tail"] = lines[tail_start:]

    # 最后更新时间
    if lines:
        info["last_update"] = lines[-1]

    return info


def main():
    parser = argparse.ArgumentParser(description="采集进度监视器")
    parser.add_argument("--log", type=str, default=None, help="日志文件路径")
    parser.add_argument("--interval", type=float, default=2.0, help="刷新间隔(秒)")
    parser.add_argument("--history", type=int, default=8, help="尾部显示行数")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else find_latest_log()
    print(f"监视日志: {log_path}")
    print(f"刷新间隔: {args.interval}s  |  按 Ctrl+C 退出\n")

    try:
        while True:
            info = parse_log(log_path, args.history)

            # 采集 total images
            img_count = count_images(FINAL_DATASET_DIR)
            nonempty, total_dirs = count_roles(FINAL_DATASET_DIR)

            # 清屏
            sys.stdout.write("\033[H\033[J")
            sys.stdout.flush()

            # ── 标题 ──
            print("╔══════════════════════════════════════════╗")
            print(f"║  图片采集进度监控")
            print(f"║  {info['current_idx']}/{info['total']} 角色  |  final_dataset: {img_count} 张 ({nonempty}/{total_dirs} 角色有图)")
            print("╚══════════════════════════════════════════╝")

            # ── 当前状态 ──
            if info["current_name"]:
                print(f"\n▶ 当前: {info['current_name']}")

            # ── 最近完成的角色 ──
            recent_done = info["done_lines"][-5:]
            if recent_done:
                print("\n▸ 最近完成:")
                for line in recent_done:
                    print(f"  {line}")

            # ── 最近跳过的 ──
            recent_skip = info["skip_exists_lines"][-3:]
            if recent_skip:
                for line in recent_skip:
                    print(f"  ○ {line}")

            # ── 尾部日志 ──
            tail = info.get("tail", [])
            download_lines = [l for l in tail
                              if any(kw in l for kw in ("下载", "尝试站点", "成功=", "失败", "搜索"))]
            if download_lines:
                print(f"\n▸ 实时日志:")
                for line in download_lines[-args.history:]:
                    # 截短过长行
                    display = line[:130] + "…" if len(line) > 130 else line
                    print(f"  {display}")

            # ── 训练状态 ──
            try:
                ps = subprocess.run(
                    ["ps", "aux"], capture_output=True, text=True, timeout=3
                )
                train_lines = [l for l in ps.stdout.splitlines()
                               if "train_adv" in l and "grep" not in l]
                if train_lines:
                    import shlex
                    parts = shlex.split(train_lines[0].strip())
                    # 找到 epoch 信息
                    epoch_info = ""
                    for i, p in enumerate(parts):
                        if p == "--epochs" and i + 1 < len(parts):
                            epoch_info = f" epoch={parts[i+1]}"
                            break
                    cpu = train_lines[0].split()[2]
                    runtime = train_lines[0].split()[10]
                    print(f"\n▸ 训练: 运行中 (CPU={cpu}%, 时间={runtime}{epoch_info})")
            except Exception:
                pass

            sys.stdout.flush()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n监控已退出")


if __name__ == "__main__":
    main()