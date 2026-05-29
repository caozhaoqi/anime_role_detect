#!/usr/bin/env python3
"""
统一日志查看器
Usage:
    python log_viewer.py                    # 查看统一日志（最后100行）
    python log_viewer.py --lines 50        # 查看最后50行
    python log_viewer.py --service model_service  # 查看模型服务日志
    python log_viewer.py --all             # 查看所有服务日志
    python log_viewer.py --info            # 显示日志系统信息
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from src.core.logging import get_log_info, get_unified_log, tail_unified_log
from src.core.logging.unified_logger import LOG_DIR

def main():
    parser = argparse.ArgumentParser(description='统一日志查看器')
    parser.add_argument('--lines', '-n', type=int, default=100, help='显示行数')
    parser.add_argument('--service', '-s', type=str, help='查看指定服务日志')
    parser.add_argument('--all', '-a', action='store_true', help='查看所有日志')
    parser.add_argument('--info', '-i', action='store_true', help='显示日志系统信息')
    args = parser.parse_args()

    print("=" * 70)
    print("  动漫角色识别系统 - 统一日志查看器")
    print("=" * 70)

    if args.info:
        print("\n📁 日志系统信息:")
        info = get_log_info()
        print(f"   日志目录: {info['log_dir']}")
        print(f"   统一日志: {info['unified_log']}")
        print("\n   子目录日志文件:")
        for subdir, files in info['files'].items():
            print(f"   [{subdir}]: {len(files)} 个文件")
        return

    if args.service:
        log_file = LOG_DIR / f"{args.service}.log"
        unified_file = LOG_DIR / "unified.log"
        if log_file.exists():
            lines = log_file.read_text(encoding='utf-8').split('\n')
            print(f"\n📋 [{args.service}] 日志 (共 {len(lines)} 行，显示最后 {args.lines} 行):")
            print("-" * 70)
            print('\n'.join(lines[-args.lines:]))
        elif unified_file.exists():
            print(f"\n📋 [{args.service}] 日志未找到，显示统一日志:")
            print(tail_unified_log(args.lines))
        else:
            print(f"\n❌ 日志文件不存在: {log_file}")
        return

    if args.all:
        print("\n📋 所有服务统一日志:")
        print("-" * 70)
        info = get_log_info()
        print(f"日志目录: {info['log_dir']}")
        print(f"\n统一日志 (最后 {args.lines} 行):")
        print(tail_unified_log(args.lines))
        return

    print(f"\n📋 统一日志 (最后 {args.lines} 行):")
    print("-" * 70)
    print(tail_unified_log(args.lines))

if __name__ == "__main__":
    main()
