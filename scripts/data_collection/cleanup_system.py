#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统清理工具 —— 释放磁盘空间，可独立运行或被 collector_runner.py 调用

用法:
    python3 scripts/data_collection/cleanup_system.py                    # 完整清理
    python3 scripts/data_collection/cleanup_system.py --dry-run          # 仅预览，不执行
    python3 scripts/data_collection/cleanup_system.py --aggressive       # 更强力清理
"""

import os
import re
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PACK_DIR = PROJECT_ROOT / "packs"
LOG_DIR = PROJECT_ROOT / "logs"


def run_cmd(cmd: list, timeout: int = 120) -> dict:
    """运行命令，返回 {"ok", "stdout", "stderr", "freed_mb"}"""
    result = {"ok": False, "stdout": "", "stderr": "", "freed_mb": 0}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        result["stdout"] = r.stdout.strip()
        result["stderr"] = r.stderr.strip()
        result["ok"] = r.returncode == 0
    except FileNotFoundError:
        result["stderr"] = "命令不存在"
    except subprocess.TimeoutExpired:
        result["stderr"] = "超时"
    except Exception as e:
        result["stderr"] = str(e)
    return result


def get_disk_free(path: str = "/") -> float:
    """获取某路径的可用空间（GB）"""
    try:
        import shutil
        _, _, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    except Exception:
        return 0


def clean_apt(dry_run: bool = False) -> dict:
    """清理 APT 缓存 + 自动移除无用依赖"""
    report = {"name": "APT 缓存 & 无用依赖", "freed_mb": 0, "ok": True, "detail": ""}

    # 统计清理前 cache 大小
    cache_size_before = 0
    try:
        r = subprocess.run(["du", "-sm", "/var/cache/apt/archives"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            cache_size_before = int(r.stdout.split()[0])
    except Exception:
        pass

    if dry_run:
        report["detail"] = f"apt 缓存: {cache_size_before}MB (将清理)"
        report["freed_mb"] = cache_size_before
        return report

    steps = []

    # apt clean
    r = run_cmd(["sudo", "-n", "apt", "clean", "-y"], timeout=60)
    if r["ok"]:
        steps.append("apt clean")
    else:
        steps.append(f"apt clean 失败 ({r['stderr'][:60]})")

    # apt autoremove
    r2 = run_cmd(["sudo", "-n", "apt", "autoremove", "-y"], timeout=120)
    if r2["ok"]:
        steps.append("autoremove")
        # 尝试从输出中提取释放空间
        for line in r2["stdout"].split("\n"):
            if "freed" in line.lower() or "释放" in line:
                report["detail"] = line.strip()
                break
    else:
        steps.append(f"autoremove 失败 ({r2['stderr'][:60]})")

    # 计算释放量
    cache_size_after = 0
    try:
        r3 = subprocess.run(["du", "-sm", "/var/cache/apt/archives"],
                            capture_output=True, text=True, timeout=10)
        if r3.returncode == 0:
            cache_size_after = int(r3.stdout.split()[0])
    except Exception:
        pass

    freed = cache_size_before - cache_size_after
    report["freed_mb"] = max(freed, 0)
    if not report["detail"]:
        report["detail"] = "清理完成" if steps[0].startswith("apt") else "; ".join(steps)
    report["ok"] = any("apt" in s for s in steps if "失败" not in s)
    return report


def clean_journal(dry_run: bool = False) -> dict:
    """清理 systemd journal 日志，保留 200MB"""
    report = {"name": "Systemd Journal", "freed_mb": 0, "ok": True, "detail": ""}

    # 统计清理前大小
    size_before = 0
    r = run_cmd(["journalctl", "--disk-usage"], timeout=10)
    if r["ok"]:
        for line in r["stdout"].split("\n"):
            import re
            m = re.search(r'(\d+(?:\.\d+)?)\s*(M|G)', line)
            if m:
                val = float(m.group(1))
                size_before = int(val * 1000) if m.group(2) == "G" else int(val)

    if dry_run:
        report["detail"] = f"journal 日志: {size_before}MB (将压缩至 200MB)"
        report["freed_mb"] = max(size_before - 200, 0)
        return report

    r = run_cmd(["sudo", "-n", "journalctl", "--vacuum-size=200M"], timeout=60)
    if r["ok"]:
        for line in r["stdout"].split("\n"):
            if "freed" in line.lower() or "Deleted" in line or "释放" in line:
                report["detail"] = line.strip()
                break
        if not report["detail"]:
            report["detail"] = "已压缩至 200MB"
        # 计算释放量
        size_after = 0
        r2 = run_cmd(["journalctl", "--disk-usage"], timeout=10)
        if r2["ok"]:
            for line in r2["stdout"].split("\n"):
                import re
                m = re.search(r'(\d+(?:\.\d+)?)\s*(M|G)', line)
                if m:
                    val = float(m.group(1))
                    size_after = int(val * 1000) if m.group(2) == "G" else int(val)
        report["freed_mb"] = max(size_before - size_after, 0)
    else:
        report["ok"] = False
        report["detail"] = f"失败: {r['stderr'][:80]}"
        report["freed_mb"] = 0
    return report


def clean_docker(dry_run: bool = False) -> dict:
    """清理 Docker 残留（无 Docker 则跳过）"""
    report = {"name": "Docker 残留", "freed_mb": 0, "ok": True, "detail": "未安装 Docker 或 Docker 未运行"}

    # 检查 docker 是否存在
    if not shutil.which("docker"):
        return report

    r = run_cmd(["docker", "info"], timeout=10)
    if not r["ok"]:
        report["detail"] = "Docker 未运行"
        return report

    if dry_run:
        report["detail"] = "Docker 残留 (将 prune)"
        report["ok"] = True
        return report

    # 统计清理前
    r = run_cmd(["docker", "system", "df"], timeout=15)
    size_before_str = ""
    if r["ok"]:
        for line in r["stdout"].split("\n"):
            if "Total" in line or "总计" in line:
                size_before_str = line.strip()

    r = run_cmd(["docker", "system", "prune", "-af", "--volumes"], timeout=120)
    if r["ok"]:
        for line in r["stdout"].split("\n"):
            if "freed" in line.lower() or "Total" in line or "释放" in line:
                report["detail"] = line.strip()
                break
            if "Space" in line:
                report["detail"] = line.strip()
        if not report["detail"]:
            report["detail"] = "已清理"
        # 解析释放空间
        import re
        m = re.search(r'(\d+(?:\.\d+)?)\s*(MB|GB)', r["stdout"])
        if m:
            val = float(m.group(1))
            report["freed_mb"] = int(val * 1000) if m.group(2) == "GB" else int(val)
        report["ok"] = True
    else:
        report["detail"] = f"清理失败: {r['stderr'][:80]}"
    return report


def clean_temp(dry_run: bool = False, aggressive: bool = False) -> dict:
    """清理 /tmp 和 /var/tmp"""
    report = {"name": "临时文件", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    for tmp_dir in ["/tmp", "/var/tmp"]:
        if not os.path.exists(tmp_dir):
            continue
        try:
            r = run_cmd(["du", "-sm", tmp_dir], timeout=10)
            size_before = int(r["stdout"].split()[0]) if r["ok"] else 0
        except Exception:
            size_before = 0

        if dry_run:
            if size_before > 0:
                details.append(f"{tmp_dir}: {size_before}MB")
            continue

        # 删除 7 天前的临时文件（默认）；aggressive 模式删 1 天前的
        days = "1" if aggressive else "7"
        r = run_cmd(["sudo", "-n", "find", tmp_dir, "-type", "f",
                      "-atime", f"+{days}", "-delete"], timeout=60)
        if r["ok"]:
            # 计算释放
            try:
                r2 = run_cmd(["du", "-sm", tmp_dir], timeout=10)
                size_after = int(r2["stdout"].split()[0]) if r2["ok"] else 0
                freed_tmp = max(size_before - size_after, 0)
                freed += freed_tmp
                if freed_tmp > 0:
                    details.append(f"{tmp_dir}: 释放 {freed_tmp}MB")
            except Exception:
                pass
        else:
            details.append(f"{tmp_dir}: 清理失败 ({r['stderr'][:40]})")

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无需清理"
    return report


def clean_project_temp(dry_run: bool = False) -> dict:
    """清理项目内的 __pycache__ / .pyc / logs 旧文件"""
    report = {"name": "项目缓存 & 旧日志", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    # __pycache__
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            if not dry_run:
                size = sum(f.stat().st_size for f in pycache.rglob("*") if f.is_file())
                shutil.rmtree(pycache, ignore_errors=True)
                freed += size
            else:
                try:
                    r = run_cmd(["du", "-sm", str(pycache)], timeout=5)
                    if r["ok"]:
                        freed += int(r["stdout"].split()[0])
                except Exception:
                    pass

    # logs 目录保留最近 5 个
    if LOG_DIR.exists():
        log_files = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(log_files) > 5:
            for f in log_files[5:]:
                if not dry_run:
                    freed += f.stat().st_size
                    f.unlink(missing_ok=True)
                else:
                    freed += f.stat().st_size
            details.append(f"日志: 清理 {len(log_files) - 5} 个旧文件")

    # 多余 packs（保险：最多保留 MAX_LOCAL_PACKS=2 个）
    max_packs = 2
    if PACK_DIR.exists():
        packs = sorted(PACK_DIR.glob("dataset_pack_*.zip"), key=lambda p: p.stat().st_mtime)
        if len(packs) > max_packs:
            for p in packs[:-max_packs]:
                if not dry_run:
                    freed += p.stat().st_size
                    p.unlink(missing_ok=True)
                else:
                    freed += p.stat().st_size
            details.append(f"旧包: 清理 {len(packs) - max_packs} 个")

    report["freed_mb"] = freed // (1024 * 1024)
    report["detail"] = "; ".join(details) if details else "无需清理"
    return report


# ══════════════════════════════════════════════
#  大文件分析 & 安全清理
# ══════════════════════════════════════════════

def analyze_disk_usage(top_n: int = 15) -> list:
    """
    扫描系统最大目录，返回 [(size_mb, path), ...]
    仅扫描 / 下 1 层和常用大目录
    """
    targets = ["/var", "/home", "/root", "/opt", "/tmp", "/usr/local"]
    entries = []

    for t in targets:
        if not os.path.exists(t):
            continue
        r = run_cmd(["du", "-sm", "--exclude=proc", "--exclude=sys", t], timeout=30)
        if r["ok"]:
            for line in r["stdout"].split("\n"):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    try:
                        entries.append((int(parts[0]), parts[1]))
                    except ValueError:
                        pass

    # 也检查 / 下各顶层目录
    r2 = run_cmd(["du", "-sm", "--exclude=proc", "--exclude=sys",
                   "--max-depth=1", "/"], timeout=30)
    if r2["ok"]:
        for line in r2["stdout"].split("\n"):
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[1] != "/":
                try:
                    mb = int(parts[0])
                    entries.append((mb, parts[1]))
                except ValueError:
                    pass

    entries.sort(reverse=True)
    seen = set()
    unique = []
    for mb, path in entries:
        if path not in seen:
            seen.add(path)
            unique.append((mb, path))

    return unique[:top_n]


def analyze_large_safe_targets() -> list:
    """
    查找可安全清理的大文件/目录
    返回 [(size_mb, path, category, description), ...]
    """
    results = []

    # 1. 旧 rotated 日志 (.gz, .1, .2 等)
    for log_dir in ["/var/log"]:
        if not os.path.exists(log_dir):
            continue
        r = run_cmd(["find", log_dir, "-type", "f", "(", "-name", "*.gz",
                      "-o", "-name", "*.old", "-o", "-name", "*.1",
                      "-o", "-name", "*.2", "-o", "-name", "*.3",
                      "-o", "-name", "*.4", "-o", "-name", "*.5",
                      "-o", "-name", "*.6", "-o", "-name", "*.7",
                      "-o", "-name", "*.8", "-o", "-name", "*.9",
                      ")", "-exec", "du", "-sm", "{}", "+"], timeout=30)
        if r["ok"] and r["stdout"]:
            total = 0
            for line in r["stdout"].split("\n"):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    try:
                        total += int(parts[0])
                    except ValueError:
                        pass
            if total > 0:
                results.append((total, log_dir, "rotated_logs", "已轮转的旧日志文件 (.gz/.1/.2…)"))

    # 2. pip 缓存
    pip_cache_paths = [
        os.path.expanduser("~/.cache/pip"),
        "/root/.cache/pip",
    ]
    for p in pip_cache_paths:
        if os.path.exists(p):
            r = run_cmd(["du", "-sm", p], timeout=10)
            if r["ok"] and r["stdout"]:
                try:
                    mb = int(r["stdout"].split()[0])
                    if mb > 0:
                        results.append((mb, p, "pip_cache", "pip 下载缓存"))
                except ValueError:
                    pass

    # 3. snap 旧版本
    snap_dir = "/var/lib/snapd/cache"
    if os.path.exists(snap_dir):
        r = run_cmd(["du", "-sm", snap_dir], timeout=10)
        if r["ok"] and r["stdout"]:
            try:
                mb = int(r["stdout"].split()[0])
                if mb > 0:
                    results.append((mb, snap_dir, "snap_cache", "Snapd 缓存"))
            except ValueError:
                pass

    # 4. trash
    trash_paths = [
        os.path.expanduser("~/.local/share/Trash"),
        "/root/.local/share/Trash",
    ]
    for p in trash_paths:
        if os.path.exists(p):
            r = run_cmd(["du", "-sm", p], timeout=10)
            if r["ok"] and r["stdout"]:
                try:
                    mb = int(r["stdout"].split()[0])
                    if mb > 0:
                        results.append((mb, p, "trash", "回收站"))
                except ValueError:
                    pass

    # 5. 旧内核（保留最近 2 个）
    if os.path.exists("/boot"):
        r = run_cmd(["dpkg", "--list", "linux-image-*"], timeout=15)
        kernels = []
        if r["ok"]:
            for line in r["stdout"].split("\n"):
                if line.startswith("ii") and "linux-image-" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        kernels.append(parts[2])
        if len(kernels) > 2:
            keep = set(kernels[:2])  # dpkg 列表按版本排序，前 2 个最新
            old_kernels = [k for k in kernels if k not in keep]
            # 粗略估算大小（每个内核 ~200MB）
            results.append((len(old_kernels) * 200, "; ".join(old_kernels),
                           "old_kernels", f"旧内核 ({len(old_kernels)} 个, 估算)"))

    results.sort(reverse=True)
    return results


def clean_system_clutter(dry_run: bool = False) -> dict:
    """
    清理可安全删除的系统残留（旧日志、缓存、回收站等）
    自动识别并清理，无需用户确认
    """
    report = {"name": "系统大文件清理", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    targets = analyze_large_safe_targets()

    for mb, path, category, desc in targets:
        if dry_run:
            details.append(f"{desc} [{path}]: ~{mb}MB")
            freed += mb
            continue

        if category == "rotated_logs":
            r = run_cmd(["sudo", "-n", "find", "/var/log", "-type", "f", "(",
                          "-name", "*.gz", "-o", "-name", "*.old",
                          "-o", "-name", "*.1", "-o", "-name", "*.2",
                          "-o", "-name", "*.3", "-o", "-name", "*.4",
                          "-o", "-name", "*.5", "-o", "-name", "*.6",
                          "-o", "-name", "*.7", "-o", "-name", "*.8",
                          "-o", "-name", "*.9",
                          ")", "-delete"], timeout=60)
            if r["ok"]:
                details.append(f"旋转日志: +{mb}MB")
                freed += mb
            else:
                details.append(f"旋转日志: 清理失败 ({r['stderr'][:40]})")

        elif category == "pip_cache":
            r = run_cmd(["sudo", "-n", "rm", "-rf", path], timeout=30)
            if r["ok"] or (not r["ok"] and "No such" in r["stderr"]):
                details.append(f"pip 缓存: +{mb}MB")
                freed += mb
            else:
                details.append(f"pip 缓存: 清理失败 ({r['stderr'][:40]})")

        elif category == "snap_cache":
            r = run_cmd(["sudo", "-n", "rm", "-rf", f"{path}/*"], timeout=30)
            if r["ok"]:
                details.append(f"Snap 缓存: +{mb}MB")
                freed += mb
            else:
                details.append(f"Snap 缓存: 清理失败 ({r['stderr'][:40]})")

        elif category == "trash":
            r = run_cmd(["sudo", "-n", "rm", "-rf", path], timeout=30)
            if r["ok"] or (not r["ok"] and "No such" in r["stderr"]):
                details.append(f"回收站: +{mb}MB")
                freed += mb
            else:
                details.append(f"回收站: 清理失败 ({r['stderr'][:40]})")

        elif category == "old_kernels":
            # 只移除列表中的旧内核
            kernel_names = path.split("; ")
            for kn in kernel_names:
                kn = kn.strip()
                if kn:
                    r = run_cmd(["sudo", "-n", "apt", "remove", "-y", kn], timeout=60)
                    if r["ok"]:
                        details.append(f"旧内核: {kn}")
                        freed += mb // max(len(kernel_names), 1)
                    else:
                        details.append(f"旧内核 {kn}: 移除失败 ({r['stderr'][:40]})")
            # 再 autoremove 清理残留
            run_cmd(["sudo", "-n", "apt", "autoremove", "-y"], timeout=60)

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无可清理系统大文件"
    return report


def run_cleanup(dry_run: bool = False, aggressive: bool = False) -> dict:
    """
    执行所有清理步骤，返回汇总报告

    Returns:
        {
            "ok": bool,
            "total_freed_mb": int,
            "free_before_gb": float,
            "free_after_gb": float,
            "steps": [report_dict, ...],
            "large_files": [ ... ]  # 磁盘大文件分析
        }
    """
    free_before = get_disk_free()
    steps = []

    steps.append(clean_apt(dry_run))
    steps.append(clean_journal(dry_run))
    steps.append(clean_docker(dry_run))
    steps.append(clean_temp(dry_run, aggressive))
    steps.append(clean_system_clutter(dry_run))
    steps.append(clean_project_temp(dry_run))

    free_after = get_disk_free()
    total_freed = sum(s.get("freed_mb", 0) for s in steps)
    all_ok = all(s.get("ok", True) for s in steps)

    # 大文件分析（始终执行，用于报告）
    large_files = analyze_large_safe_targets()

    return {
        "ok": all_ok,
        "total_freed_mb": total_freed,
        "free_before_gb": round(free_before, 1),
        "free_after_gb": round(free_after, 1),
        "steps": steps,
        "large_files": large_files,
        "dry_run": dry_run,
    }


def format_report(result: dict) -> str:
    """格式化清理报告为可推送的文本"""
    lines = [
        f"🧹 **系统清理{'预览' if result['dry_run'] else '完成'}**",
        f"",
    ]

    for s in result["steps"]:
        icon = "✅" if s["ok"] else "⚠️"
        detail = s["detail"][:120] if s["detail"] else "无需处理"
        freed = s.get("freed_mb", 0)
        freed_str = f" (+{freed}MB)" if freed > 0 else ""
        lines.append(f"{icon} {s['name']}{freed_str}")
        lines.append(f"   └─ {detail}")

    # 大文件预警
    large = result.get("large_files", [])
    risky = [(mb, path, desc) for mb, path, cat, desc in large
             if cat not in ("rotated_logs", "pip_cache", "snap_cache", "trash")]
    if risky:
        lines.append("")
        lines.append("⚠️ **以下大文件需人工检查：**")
        for mb, path, desc in risky[:5]:
            lines.append(f"   • {desc}: {path} (~{mb}MB)")

    lines.append("")
    lines.append(f"💾 释放空间: **{result['total_freed_mb']} MB**")
    lines.append(f"📊 磁盘: {result['free_before_gb']}GB → {result['free_after_gb']}GB 可用")

    return "\n".join(lines)


# ══════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="系统清理工具")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不执行")
    parser.add_argument("--aggressive", action="store_true", help="更强力清理（如 /tmp 删除更早文件）")
    parser.add_argument("--scan", action="store_true", help="仅扫描大文件分析，不执行清理")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    if args.scan:
        large = analyze_large_safe_targets()
        print(f"🔍 **系统大文件分析**\n")
        if not large:
            print("无可清理的大文件")
        else:
            for mb, path, cat, desc in large:
                print(f"  • {desc} [{path}]: ~{mb}MB")
        sys.exit(0)

    result = run_cleanup(dry_run=args.dry_run, aggressive=args.aggressive)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result))
        print(f"\n{'=' * 40}")
        print(f"运行: python3 {__file__}  # 实际执行清理")
        if not args.dry_run:
            print("已实际执行清理 ✅")