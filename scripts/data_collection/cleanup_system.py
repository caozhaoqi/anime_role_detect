#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统清理工具 —— 支持 macOS / Windows / Linux，释放磁盘空间

用法:
    python3 cleanup_system.py                 # 完整清理
    python3 cleanup_system.py --dry-run       # 仅预览，不执行
    python3 cleanup_system.py --aggressive    # 更强力清理
    python3 cleanup_system.py --scan          # 仅扫描大文件
    python3 cleanup_system.py --json          # 以 JSON 格式输出
"""

import os
import re
import sys
import json
import shutil
import subprocess
import platform
import tempfile
import time
from pathlib import Path

# ══════════════════════════════════════════════
#  平台检测
# ══════════════════════════════════════════════

PLATFORM = platform.system()  # "Windows" / "Darwin" / "Linux"
IS_WINDOWS = PLATFORM == "Windows"
IS_MAC = PLATFORM == "Darwin"
IS_LINUX = PLATFORM == "Linux"

PROJECT_ROOT = Path(__file__).parent
PACK_DIR = PROJECT_ROOT / "packs"
LOG_DIR = PROJECT_ROOT / "logs"


# ══════════════════════════════════════════════
#  通用工具
# ══════════════════════════════════════════════

def run_cmd(cmd: list, timeout: int = 120, shell: bool = False) -> dict:
    """运行命令，返回 {"ok", "stdout", "stderr"}"""
    result = {"ok": False, "stdout": "", "stderr": ""}
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, shell=shell,
            encoding="utf-8", errors="replace"
        )
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


def get_disk_free(path: str = None) -> float:
    """获取路径可用空间（GB）"""
    if path is None:
        path = "C:\\" if IS_WINDOWS else "/"
    try:
        _, _, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    except Exception:
        return 0


def get_dir_size_mb(path: str) -> int:
    """获取目录大小（MB），跨平台实现"""
    try:
        p = Path(path)
        if not p.exists():
            return 0
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return total // (1024 * 1024)
    except Exception:
        return 0


def safe_remove_tree(path: str) -> bool:
    """安全删除目录树"""
    try:
        shutil.rmtree(path, ignore_errors=True)
        return True
    except Exception:
        return False


def safe_remove_file(path: str) -> bool:
    """安全删除文件"""
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except Exception:
        return False


def requires_admin() -> bool:
    """检查是否具有管理员/root 权限"""
    if IS_WINDOWS:
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    else:
        return os.geteuid() == 0


def sudo_cmd(cmd: list) -> list:
    """非 root Linux/macOS 下自动加 sudo -n"""
    if IS_WINDOWS:
        return cmd
    if requires_admin():
        return cmd
    return ["sudo", "-n"] + cmd


# ══════════════════════════════════════════════
#  macOS 专属清理
# ══════════════════════════════════════════════

def clean_mac_caches(dry_run: bool = False, aggressive: bool = False) -> dict:
    """清理 macOS 用户缓存目录"""
    report = {"name": "macOS 用户缓存", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    user_cache = Path.home() / "Library" / "Caches"
    if not user_cache.exists():
        report["detail"] = "缓存目录不存在"
        return report

    size_before = get_dir_size_mb(str(user_cache))

    if dry_run:
        report["detail"] = f"用户缓存: ~{size_before}MB (将清理)"
        report["freed_mb"] = size_before
        return report

    # 保守清理：删除各子目录中的文件，保留目录结构
    cleaned = 0
    try:
        for sub in user_cache.iterdir():
            if sub.is_dir():
                sub_size = get_dir_size_mb(str(sub))
                if sub_size > 0:
                    safe_remove_tree(str(sub))
                    cleaned += sub_size
    except Exception as e:
        report["detail"] = f"部分清理失败: {e}"

    freed = cleaned
    details.append(f"~/Library/Caches: ~{freed}MB")

    # aggressive: 系统日志
    if aggressive:
        log_dirs = [
            Path.home() / "Library" / "Logs",
            Path("/Library/Logs"),
        ]
        for ld in log_dirs:
            if ld.exists():
                sz = get_dir_size_mb(str(ld))
                for f in ld.rglob("*.log"):
                    safe_remove_file(str(f))
                freed += sz
                details.append(f"{ld}: ~{sz}MB")

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无需清理"
    return report


def clean_mac_trash(dry_run: bool = False) -> dict:
    """清空 macOS 回收站"""
    report = {"name": "macOS 回收站", "freed_mb": 0, "ok": True, "detail": ""}
    trash = Path.home() / ".Trash"
    if not trash.exists():
        report["detail"] = "回收站为空"
        return report

    sz = get_dir_size_mb(str(trash))
    if dry_run:
        report["detail"] = f"回收站: ~{sz}MB (将清空)"
        report["freed_mb"] = sz
        return report

    for item in trash.iterdir():
        if item.is_dir():
            safe_remove_tree(str(item))
        else:
            safe_remove_file(str(item))

    report["freed_mb"] = sz
    report["detail"] = f"已清空回收站: ~{sz}MB"
    return report


def clean_mac_brew(dry_run: bool = False) -> dict:
    """清理 Homebrew 缓存"""
    report = {"name": "Homebrew 缓存", "freed_mb": 0, "ok": True, "detail": "未安装 Homebrew"}
    if not shutil.which("brew"):
        return report

    r = run_cmd(["brew", "--cache"], timeout=10)
    brew_cache = r["stdout"].strip() if r["ok"] else ""

    sz = get_dir_size_mb(brew_cache) if brew_cache else 0
    if dry_run:
        report["detail"] = f"Homebrew 缓存: ~{sz}MB (将清理)"
        report["freed_mb"] = sz
        return report

    r = run_cmd(["brew", "cleanup", "--prune=all", "-s"], timeout=120)
    if r["ok"]:
        # 解析输出中的释放量
        m = re.search(r'(\d+(?:\.\d+)?)\s*(MB|GB)', r["stdout"])
        if m:
            val = float(m.group(1))
            freed = int(val * 1024) if m.group(2) == "GB" else int(val)
        else:
            freed = sz
        report["freed_mb"] = freed
        report["detail"] = f"brew cleanup 完成: ~{freed}MB"
        report["ok"] = True
    else:
        report["ok"] = False
        report["detail"] = f"失败: {r['stderr'][:80]}"
    return report


def clean_mac_xcode(dry_run: bool = False) -> dict:
    """清理 Xcode 派生数据和归档（如存在）"""
    report = {"name": "Xcode 派生数据", "freed_mb": 0, "ok": True, "detail": ""}
    targets = [
        Path.home() / "Library" / "Developer" / "Xcode" / "DerivedData",
        Path.home() / "Library" / "Developer" / "Xcode" / "Archives",
        Path.home() / "Library" / "Developer" / "CoreSimulator" / "Caches",
    ]
    freed = 0
    details = []
    for t in targets:
        if t.exists():
            sz = get_dir_size_mb(str(t))
            if sz > 0:
                if dry_run:
                    details.append(f"{t.name}: ~{sz}MB")
                else:
                    safe_remove_tree(str(t))
                    details.append(f"{t.name}: +{sz}MB")
                freed += sz

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无 Xcode 缓存"
    return report


# ══════════════════════════════════════════════
#  Windows 专属清理
# ══════════════════════════════════════════════

def clean_windows_temp(dry_run: bool = False, aggressive: bool = False) -> dict:
    """清理 Windows 临时文件"""
    report = {"name": "Windows 临时文件", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    temp_dirs = [
        os.environ.get("TEMP", ""),
        os.environ.get("TMP", ""),
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Temp"),
    ]
    temp_dirs = list({d for d in temp_dirs if d and os.path.exists(d)})

    for td in temp_dirs:
        sz = get_dir_size_mb(td)
        if dry_run:
            if sz > 0:
                details.append(f"{td}: ~{sz}MB")
            freed += sz
            continue

        deleted = 0
        for item in Path(td).iterdir():
            try:
                item_sz = item.stat().st_size if item.is_file() else get_dir_size_mb(str(item)) * 1024 * 1024
                if item.is_file():
                    item.unlink(missing_ok=True)
                    deleted += item_sz
                elif item.is_dir():
                    safe_remove_tree(str(item))
                    deleted += item_sz
            except PermissionError:
                pass
            except Exception:
                pass
        freed_mb = deleted // (1024 * 1024)
        if freed_mb > 0:
            details.append(f"{td}: ~{freed_mb}MB")
        freed += freed_mb

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无需清理"
    return report


def clean_windows_recycle(dry_run: bool = False) -> dict:
    """清空 Windows 回收站"""
    report = {"name": "Windows 回收站", "freed_mb": 0, "ok": True, "detail": ""}

    if dry_run:
        report["detail"] = "回收站 (将清空)"
        return report

    try:
        import ctypes
        # SHEmptyRecycleBin flags: SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND
        ret = ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, 0x0007)
        if ret == 0 or ret == -2147418113:  # S_OK or already empty
            report["detail"] = "回收站已清空"
        else:
            report["detail"] = f"清空完成 (code={ret})"
        report["ok"] = True
    except Exception as e:
        report["ok"] = False
        report["detail"] = f"失败: {e}"
    return report


def clean_windows_prefetch(dry_run: bool = False) -> dict:
    """清理 Windows Prefetch 文件（需管理员）"""
    report = {"name": "Windows Prefetch", "freed_mb": 0, "ok": True, "detail": ""}
    prefetch = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "Prefetch"
    if not prefetch.exists():
        report["detail"] = "Prefetch 目录不存在"
        return report

    sz = get_dir_size_mb(str(prefetch))
    if dry_run:
        report["detail"] = f"Prefetch: ~{sz}MB (需管理员权限)"
        report["freed_mb"] = sz
        return report

    if not requires_admin():
        report["detail"] = "需要管理员权限，跳过"
        return report

    freed = 0
    for f in prefetch.glob("*.pf"):
        try:
            fz = f.stat().st_size
            f.unlink()
            freed += fz
        except Exception:
            pass
    report["freed_mb"] = freed // (1024 * 1024)
    report["detail"] = f"已清理 Prefetch: ~{report['freed_mb']}MB"
    return report


def clean_windows_browser_cache(dry_run: bool = False) -> dict:
    """清理常见浏览器缓存（Edge / Chrome / Firefox）"""
    report = {"name": "浏览器缓存", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    local_app = os.environ.get("LOCALAPPDATA", "")
    appdata = os.environ.get("APPDATA", "")

    cache_paths = {
        "Chrome": os.path.join(local_app, "Google", "Chrome", "User Data", "Default", "Cache"),
        "Edge":   os.path.join(local_app, "Microsoft", "Edge", "User Data", "Default", "Cache"),
        "Firefox": os.path.join(appdata, "Mozilla", "Firefox", "Profiles"),
    }

    for browser, path in cache_paths.items():
        if not os.path.exists(path):
            continue
        sz = get_dir_size_mb(path)
        if sz == 0:
            continue
        if dry_run:
            details.append(f"{browser}: ~{sz}MB")
        else:
            safe_remove_tree(path)
            details.append(f"{browser}: +{sz}MB")
        freed += sz

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无浏览器缓存"
    return report


def clean_windows_delivery_optimization(dry_run: bool = False) -> dict:
    """清理 Windows Update 传递优化缓存"""
    report = {"name": "Windows 传递优化缓存", "freed_mb": 0, "ok": True, "detail": ""}
    do_path = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "ServiceProfiles" \
              / "NetworkService" / "AppData" / "Local" / "Microsoft" / "Windows" \
              / "DeliveryOptimization"

    if not do_path.exists():
        report["detail"] = "不存在"
        return report

    sz = get_dir_size_mb(str(do_path))
    if dry_run:
        report["detail"] = f"传递优化缓存: ~{sz}MB"
        report["freed_mb"] = sz
        return report

    if requires_admin():
        safe_remove_tree(str(do_path))
        report["freed_mb"] = sz
        report["detail"] = f"已清理: ~{sz}MB"
    else:
        report["detail"] = "需要管理员权限，跳过"
    return report


# ══════════════════════════════════════════════
#  Linux 专属清理
# ══════════════════════════════════════════════

def clean_apt(dry_run: bool = False) -> dict:
    """清理 APT 缓存 + 自动移除无用依赖"""
    report = {"name": "APT 缓存 & 无用依赖", "freed_mb": 0, "ok": True, "detail": ""}
    if not shutil.which("apt"):
        report["detail"] = "APT 不可用"
        return report

    size_before = get_dir_size_mb("/var/cache/apt/archives")
    if dry_run:
        report["detail"] = f"apt 缓存: ~{size_before}MB (将清理)"
        report["freed_mb"] = size_before
        return report

    steps = []
    r = run_cmd(sudo_cmd(["apt", "clean", "-y"]), timeout=60)
    if r["ok"]:
        steps.append("apt clean")
    r2 = run_cmd(sudo_cmd(["apt", "autoremove", "-y"]), timeout=120)
    if r2["ok"]:
        steps.append("autoremove")

    size_after = get_dir_size_mb("/var/cache/apt/archives")
    report["freed_mb"] = max(size_before - size_after, 0)
    report["detail"] = "清理完成: " + ", ".join(steps) if steps else "清理失败"
    report["ok"] = bool(steps)
    return report


def clean_journal(dry_run: bool = False) -> dict:
    """清理 systemd journal 日志，保留 200MB"""
    report = {"name": "Systemd Journal", "freed_mb": 0, "ok": True, "detail": ""}
    if not shutil.which("journalctl"):
        report["detail"] = "journalctl 不可用"
        return report

    size_before = 0
    r = run_cmd(["journalctl", "--disk-usage"], timeout=10)
    if r["ok"]:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(M|G)', r["stdout"])
        if m:
            val = float(m.group(1))
            size_before = int(val * 1000) if m.group(2) == "G" else int(val)

    if dry_run:
        report["detail"] = f"journal: ~{size_before}MB (将压缩至 200MB)"
        report["freed_mb"] = max(size_before - 200, 0)
        return report

    r = run_cmd(sudo_cmd(["journalctl", "--vacuum-size=200M"]), timeout=60)
    if r["ok"]:
        size_after = 0
        r2 = run_cmd(["journalctl", "--disk-usage"], timeout=10)
        if r2["ok"]:
            m = re.search(r'(\d+(?:\.\d+)?)\s*(M|G)', r2["stdout"])
            if m:
                val = float(m.group(1))
                size_after = int(val * 1000) if m.group(2) == "G" else int(val)
        report["freed_mb"] = max(size_before - size_after, 0)
        report["detail"] = "已压缩至 200MB"
    else:
        report["ok"] = False
        report["detail"] = f"失败: {r['stderr'][:80]}"
    return report


def clean_linux_temp(dry_run: bool = False, aggressive: bool = False) -> dict:
    """清理 /tmp 和 /var/tmp"""
    report = {"name": "Linux 临时文件", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []
    days = "1" if aggressive else "7"

    for tmp_dir in ["/tmp", "/var/tmp"]:
        if not os.path.exists(tmp_dir):
            continue
        sz = get_dir_size_mb(tmp_dir)
        if dry_run:
            if sz > 0:
                details.append(f"{tmp_dir}: ~{sz}MB")
            freed += sz
            continue

        r = run_cmd(sudo_cmd(["find", tmp_dir, "-type", "f",
                               "-atime", f"+{days}", "-delete"]), timeout=60)
        if r["ok"]:
            sz_after = get_dir_size_mb(tmp_dir)
            freed_tmp = max(sz - sz_after, 0)
            if freed_tmp > 0:
                details.append(f"{tmp_dir}: +{freed_tmp}MB")
            freed += freed_tmp

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无需清理"
    return report


def clean_snap(dry_run: bool = False) -> dict:
    """清理 Snap 旧版本"""
    report = {"name": "Snap 旧版本", "freed_mb": 0, "ok": True, "detail": "未安装 Snap"}
    if not shutil.which("snap"):
        return report

    r = run_cmd(["snap", "list", "--all"], timeout=15)
    if not r["ok"]:
        report["detail"] = "snap list 失败"
        return report

    disabled = []
    for line in r["stdout"].split("\n")[1:]:
        parts = line.split()
        if len(parts) >= 6 and "disabled" in parts:
            disabled.append((parts[0], parts[2]))  # name, revision

    if not disabled:
        report["detail"] = "无旧版 Snap"
        return report

    freed = len(disabled) * 100  # 粗估每个 100MB
    if dry_run:
        report["detail"] = f"{len(disabled)} 个旧版本 (将移除, 约 {freed}MB)"
        report["freed_mb"] = freed
        return report

    removed = 0
    for name, rev in disabled:
        r = run_cmd(sudo_cmd(["snap", "remove", name, "--revision", rev]), timeout=60)
        if r["ok"]:
            removed += 1

    report["freed_mb"] = removed * 100
    report["detail"] = f"移除 {removed}/{len(disabled)} 个旧版本"
    return report


# ══════════════════════════════════════════════
#  跨平台通用清理
# ══════════════════════════════════════════════

def clean_docker(dry_run: bool = False) -> dict:
    """清理 Docker 残留（无 Docker 则跳过）"""
    report = {"name": "Docker 残留", "freed_mb": 0, "ok": True, "detail": "未安装 Docker"}
    if not shutil.which("docker"):
        return report

    r = run_cmd(["docker", "info"], timeout=10)
    if not r["ok"]:
        report["detail"] = "Docker 未运行"
        return report

    if dry_run:
        report["detail"] = "Docker 残留 (将 prune)"
        return report

    r = run_cmd(["docker", "system", "prune", "-af", "--volumes"], timeout=120)
    if r["ok"]:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(MB|GB)', r["stdout"])
        if m:
            val = float(m.group(1))
            report["freed_mb"] = int(val * 1024) if m.group(2) == "GB" else int(val)
        for line in r["stdout"].split("\n"):
            if "freed" in line.lower() or "Space" in line:
                report["detail"] = line.strip()
                break
        if not report["detail"]:
            report["detail"] = "已清理"
    else:
        report["detail"] = f"清理失败: {r['stderr'][:80]}"
    return report


def clean_pip_cache(dry_run: bool = False) -> dict:
    """清理 pip 下载缓存（跨平台）"""
    report = {"name": "pip 缓存", "freed_mb": 0, "ok": True, "detail": ""}
    if not shutil.which("pip") and not shutil.which("pip3"):
        report["detail"] = "pip 未安装"
        return report

    pip_exe = shutil.which("pip3") or shutil.which("pip")
    r = run_cmd([pip_exe, "cache", "dir"], timeout=10)
    cache_dir = r["stdout"].strip() if r["ok"] else ""

    if not cache_dir or not os.path.exists(cache_dir):
        # 回退默认路径
        if IS_WINDOWS:
            cache_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "pip", "Cache")
        elif IS_MAC:
            cache_dir = str(Path.home() / "Library" / "Caches" / "pip")
        else:
            cache_dir = str(Path.home() / ".cache" / "pip")

    if not os.path.exists(cache_dir):
        report["detail"] = "pip 缓存目录不存在"
        return report

    sz = get_dir_size_mb(cache_dir)
    if dry_run:
        report["detail"] = f"pip 缓存: ~{sz}MB"
        report["freed_mb"] = sz
        return report

    r = run_cmd([pip_exe, "cache", "purge"], timeout=30)
    if r["ok"]:
        report["freed_mb"] = sz
        report["detail"] = f"已清理 pip 缓存: ~{sz}MB"
    else:
        # 直接删除
        safe_remove_tree(cache_dir)
        report["freed_mb"] = sz
        report["detail"] = f"直接删除缓存目录: ~{sz}MB"
    return report


def clean_npm_cache(dry_run: bool = False) -> dict:
    """清理 npm 缓存（跨平台）"""
    report = {"name": "npm 缓存", "freed_mb": 0, "ok": True, "detail": "未安装 npm"}
    if not shutil.which("npm"):
        return report

    r = run_cmd(["npm", "config", "get", "cache"], timeout=10)
    cache_dir = r["stdout"].strip() if r["ok"] else ""

    if not cache_dir or not os.path.exists(cache_dir):
        report["detail"] = "npm 缓存目录不存在"
        return report

    sz = get_dir_size_mb(cache_dir)
    if dry_run:
        report["detail"] = f"npm 缓存: ~{sz}MB"
        report["freed_mb"] = sz
        return report

    r = run_cmd(["npm", "cache", "clean", "--force"], timeout=60)
    if r["ok"]:
        report["freed_mb"] = sz
        report["detail"] = f"已清理 npm 缓存: ~{sz}MB"
    else:
        report["ok"] = False
        report["detail"] = f"失败: {r['stderr'][:80]}"
    return report


def clean_project_temp(dry_run: bool = False) -> dict:
    """清理项目内的 __pycache__ / .pyc / 旧日志 / 旧包"""
    report = {"name": "项目缓存 & 旧日志", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    # __pycache__ 和 .pyc
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            sz = sum(f.stat().st_size for f in pycache.rglob("*") if f.is_file())
            if not dry_run:
                shutil.rmtree(pycache, ignore_errors=True)
            freed += sz

    for pyc in PROJECT_ROOT.rglob("*.pyc"):
        sz = pyc.stat().st_size
        if not dry_run:
            safe_remove_file(str(pyc))
        freed += sz

    # logs 目录保留最近 5 个
    if LOG_DIR.exists():
        log_files = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(log_files) > 5:
            for f in log_files[5:]:
                freed += f.stat().st_size
                if not dry_run:
                    safe_remove_file(str(f))
            details.append(f"日志: 清理 {len(log_files) - 5} 个旧文件")

    # 旧 packs（保留最近 2 个）
    if PACK_DIR.exists():
        packs = sorted(PACK_DIR.glob("dataset_pack_*.zip"), key=lambda p: p.stat().st_mtime)
        if len(packs) > 2:
            for p in packs[:-2]:
                freed += p.stat().st_size
                if not dry_run:
                    safe_remove_file(str(p))
            details.append(f"旧包: 清理 {len(packs) - 2} 个")

    report["freed_mb"] = freed // (1024 * 1024)
    report["detail"] = "; ".join(details) if details else "无需清理"
    return report


def clean_browser_cache_cross(dry_run: bool = False) -> dict:
    """清理浏览器缓存（macOS / Linux）"""
    report = {"name": "浏览器缓存", "freed_mb": 0, "ok": True, "detail": ""}
    freed = 0
    details = []

    home = Path.home()
    if IS_MAC:
        cache_paths = {
            "Chrome":  home / "Library" / "Caches" / "Google" / "Chrome" / "Default" / "Cache",
            "Firefox": home / "Library" / "Caches" / "Firefox",
            "Safari":  home / "Library" / "Caches" / "com.apple.Safari",
        }
    else:  # Linux
        cache_paths = {
            "Chrome":  home / ".cache" / "google-chrome" / "Default" / "Cache",
            "Chromium": home / ".cache" / "chromium" / "Default" / "Cache",
            "Firefox": home / ".cache" / "mozilla" / "firefox",
        }

    for browser, path in cache_paths.items():
        if not path.exists():
            continue
        sz = get_dir_size_mb(str(path))
        if sz == 0:
            continue
        if dry_run:
            details.append(f"{browser}: ~{sz}MB")
        else:
            safe_remove_tree(str(path))
            details.append(f"{browser}: +{sz}MB")
        freed += sz

    report["freed_mb"] = freed
    report["detail"] = "; ".join(details) if details else "无浏览器缓存"
    return report


# ══════════════════════════════════════════════
#  磁盘扫描分析
# ══════════════════════════════════════════════

def analyze_large_safe_targets() -> list:
    """
    查找可安全清理的大文件/目录
    返回 [(size_mb, path, category, description), ...]
    """
    results = []

    if IS_WINDOWS:
        local_app = os.environ.get("LOCALAPPDATA", "")
        appdata = os.environ.get("APPDATA", "")
        temp = os.environ.get("TEMP", "")

        candidates = [
            (os.path.join(local_app, "Google", "Chrome", "User Data", "Default", "Cache"),
             "chrome_cache", "Chrome 缓存"),
            (os.path.join(local_app, "Microsoft", "Edge", "User Data", "Default", "Cache"),
             "edge_cache", "Edge 缓存"),
            (temp, "temp", "Windows 临时文件"),
        ]
        for path, cat, desc in candidates:
            if os.path.exists(path):
                sz = get_dir_size_mb(path)
                if sz > 0:
                    results.append((sz, path, cat, desc))

    elif IS_MAC:
        home = Path.home()
        candidates = [
            (str(home / "Library" / "Caches"), "mac_caches", "macOS 用户缓存"),
            (str(home / ".Trash"), "trash", "回收站"),
            (str(home / "Library" / "Developer" / "Xcode" / "DerivedData"),
             "xcode", "Xcode 派生数据"),
            (str(home / ".cache" / "pip"), "pip_cache", "pip 缓存"),
        ]
        for path, cat, desc in candidates:
            if os.path.exists(path):
                sz = get_dir_size_mb(path)
                if sz > 0:
                    results.append((sz, path, cat, desc))

    else:  # Linux
        candidates = [
            ("/var/cache/apt/archives", "apt_cache", "APT 缓存"),
            (str(Path.home() / ".cache" / "pip"), "pip_cache", "pip 缓存"),
            (str(Path.home() / ".cache"), "user_cache", "用户缓存"),
            ("/tmp", "tmp", "临时文件"),
            ("/var/log", "logs", "系统日志"),
        ]
        for path, cat, desc in candidates:
            if os.path.exists(path):
                sz = get_dir_size_mb(path)
                if sz > 0:
                    results.append((sz, path, cat, desc))

    results.sort(reverse=True)
    return results


# ══════════════════════════════════════════════
#  主清理流程（按平台分发）
# ══════════════════════════════════════════════

def run_cleanup(dry_run: bool = False, aggressive: bool = False) -> dict:
    """
    执行所有清理步骤，返回汇总报告

    Returns:
        {
            "ok": bool,
            "platform": str,
            "total_freed_mb": int,
            "free_before_gb": float,
            "free_after_gb": float,
            "steps": [report_dict, ...],
            "large_files": [ ... ]
        }
    """
    free_before = get_disk_free()
    steps = []

    if IS_WINDOWS:
        steps.append(clean_windows_temp(dry_run, aggressive))
        steps.append(clean_windows_recycle(dry_run))
        steps.append(clean_windows_prefetch(dry_run))
        steps.append(clean_windows_browser_cache(dry_run))
        steps.append(clean_windows_delivery_optimization(dry_run))
        steps.append(clean_pip_cache(dry_run))
        steps.append(clean_npm_cache(dry_run))
        steps.append(clean_docker(dry_run))
        steps.append(clean_project_temp(dry_run))

    elif IS_MAC:
        steps.append(clean_mac_caches(dry_run, aggressive))
        steps.append(clean_mac_trash(dry_run))
        steps.append(clean_mac_brew(dry_run))
        steps.append(clean_mac_xcode(dry_run))
        steps.append(clean_browser_cache_cross(dry_run))
        steps.append(clean_pip_cache(dry_run))
        steps.append(clean_npm_cache(dry_run))
        steps.append(clean_docker(dry_run))
        steps.append(clean_project_temp(dry_run))

    else:  # Linux
        steps.append(clean_apt(dry_run))
        steps.append(clean_journal(dry_run))
        steps.append(clean_snap(dry_run))
        steps.append(clean_linux_temp(dry_run, aggressive))
        steps.append(clean_browser_cache_cross(dry_run))
        steps.append(clean_pip_cache(dry_run))
        steps.append(clean_npm_cache(dry_run))
        steps.append(clean_docker(dry_run))
        steps.append(clean_project_temp(dry_run))

    free_after = get_disk_free()
    total_freed = sum(s.get("freed_mb", 0) for s in steps)
    all_ok = all(s.get("ok", True) for s in steps)
    large_files = analyze_large_safe_targets()

    return {
        "ok": all_ok,
        "platform": PLATFORM,
        "total_freed_mb": total_freed,
        "free_before_gb": round(free_before, 1),
        "free_after_gb": round(free_after, 1),
        "steps": steps,
        "large_files": large_files,
        "dry_run": dry_run,
    }


def format_report(result: dict) -> str:
    """格式化清理报告"""
    platform_label = {"Windows": "🪟 Windows", "Darwin": "🍎 macOS", "Linux": "🐧 Linux"}.get(
        result["platform"], result["platform"]
    )
    mode_label = "预览" if result["dry_run"] else "完成"
    lines = [
        f"🧹 **系统清理{mode_label}** [{platform_label}]",
        "",
    ]

    for s in result["steps"]:
        icon = "✅" if s["ok"] else "⚠️"
        detail = (s["detail"] or "无需处理")[:120]
        freed = s.get("freed_mb", 0)
        freed_str = f" (+{freed}MB)" if freed > 0 else ""
        lines.append(f"{icon} {s['name']}{freed_str}")
        lines.append(f"   └─ {detail}")

    large = result.get("large_files", [])
    if large:
        lines.append("")
        lines.append("📂 **磁盘占用较大的目录：**")
        for mb, path, cat, desc in large[:5]:
            lines.append(f"   • {desc}: {path} (~{mb}MB)")

    lines.append("")
    lines.append(f"💾 预计释放: **{result['total_freed_mb']} MB**")
    lines.append(f"📊 磁盘: {result['free_before_gb']}GB → {result['free_after_gb']}GB 可用")

    return "\n".join(lines)


# ══════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"系统清理工具 (当前平台: {PLATFORM})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 cleanup_system.py --dry-run       # 预览将清理的内容
  python3 cleanup_system.py                 # 执行清理
  python3 cleanup_system.py --aggressive    # 更强力清理
  python3 cleanup_system.py --scan          # 仅扫描大文件
  python3 cleanup_system.py --json          # JSON 输出
        """
    )
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不执行任何操作")
    parser.add_argument("--aggressive", action="store_true", help="更强力清理（如临时文件删除范围更广）")
    parser.add_argument("--scan", action="store_true", help="仅扫描大文件，不执行清理")
    parser.add_argument("--json", dest="output_json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    print(f"🖥  平台: {PLATFORM} | 管理员: {'是' if requires_admin() else '否'}")
    print()

    if args.scan:
        large = analyze_large_safe_targets()
        print("🔍 **可清理大文件分析**\n")
        if not large:
            print("未发现可清理的大文件/目录")
        else:
            for mb, path, cat, desc in large:
                print(f"  • {desc}\n    路径: {path}\n    大小: ~{mb}MB\n")
        sys.exit(0)

    result = run_cleanup(dry_run=args.dry_run, aggressive=args.aggressive)

    if args.output_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result))
        print(f"\n{'=' * 50}")
        if args.dry_run:
            print(f"预览模式，未实际操作。执行清理请去掉 --dry-run 参数。")
        else:
            print("✅ 已实际执行清理")