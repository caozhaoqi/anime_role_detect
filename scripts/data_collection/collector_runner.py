#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集任务启动器 —— 一键启动 + 飞书推送进度 + 资源监控告警 + 自动保护停机

用法:
    python3 scripts/data_collection/collector_runner.py
    python3 scripts/data_collection/collector_runner.py --feishu-config scripts/notification_config.json
    python3 scripts/data_collection/collector_runner.py --max-count 50 --skip-existing

功能:
    - 启动 collect_from_keywords.py 作为子进程
    - 定期检查内存 / 磁盘 / CPU 占用
    - 定时推送采集进度到飞书
    - 每新增 100 张图片自动打包成 zip（保持角色目录结构）
    - 自动推送打包通知，提醒用户下载
    - 资源告急时自动停止采集并推送告警
"""

import os
import re
import sys
import json
import time
import signal
import subprocess
import argparse
import threading
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# 同级模块导入
from cleanup_system import run_cleanup, format_report

# ── 路径 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
COLLECTOR_SCRIPT = PROJECT_ROOT / "scripts" / "data_collection" / "collect_from_keywords.py"
DEFAULT_FEISHU_CONFIG = PROJECT_ROOT / "scripts" / "notification_config.json"
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
HASH_DB_PATH = PROJECT_ROOT / "data" / "image_hashes.db"
LOG_DIR = PROJECT_ROOT / "logs"

# ── 资源阈值 ──
MEMORY_WARN_PCT = 70         # 内存使用 ≥70% 时警告
MEMORY_CRITICAL_PCT = 88     # 内存使用 ≥88% 时自动停止采集
DISK_WARN_PCT = 80           # 磁盘使用 ≥80% 时警告
DISK_CRITICAL_PCT = 92       # 磁盘使用 ≥92% 时自动停止采集
PROGRESS_INTERVAL = 300      # 进度推送间隔（秒）
RESOURCE_CHECK_INTERVAL = 60  # 资源检查间隔（秒）

# ── 打包配置 ──
PACK_INTERVAL = 100          # 每新增 100 张打包一次
PACK_INTERVAL_CHECK_INTERVAL = 60  # 打包检查间隔（秒）
PACK_DIR = PROJECT_ROOT / "packs"
PACK_STATE_FILE = PROJECT_ROOT / "data" / ".pack_state.json"
MAX_LOCAL_PACKS = 2          # 本地保留最近 N 个包，多余自动删除

# ── 采集日志路径 ──
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
COLLECTOR_LOG = LOG_DIR / f"collector_{TIMESTAMP}.log"


# ══════════════════════════════════════════════
#  飞书通知
# ══════════════════════════════════════════════
class FeishuNotifier:
    """飞书消息推送，直接 API 调用，无需额外依赖"""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.app_id = cfg["feishu"]["app_id"]
        self.app_secret = cfg["feishu"]["app_secret"]
        self.receive_id = cfg["feishu"]["receive_id"]
        self.receive_id_type = cfg["feishu"].get("receive_id_type", "chat_id")
        self._token: Optional[str] = None
        self._token_expires: float = 0

    def _get_token(self) -> Optional[str]:
        if self._token and time.time() < self._token_expires:
            return self._token
        try:
            import requests
            resp = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=10,
            )
            data = resp.json()
            if data.get("code") == 0:
                self._token = data["tenant_access_token"]
                self._token_expires = time.time() + data.get("expire", 7200) - 120
                return self._token
        except Exception as e:
            print(f"  [Feishu] 获取 token 失败: {e}")
        return None

    def send(self, title: str, message: str) -> bool:
        token = self._get_token()
        if not token:
            return False
        try:
            import requests
            content = json.dumps({"text": f"**{title}**\n\n{message}"})
            resp = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                params={"receive_id_type": self.receive_id_type},
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"receive_id": self.receive_id, "msg_type": "text", "content": content},
                timeout=10,
            )
            return resp.json().get("code") == 0
        except Exception as e:
            print(f"  [Feishu] 发送消息失败: {e}")
            return False


# ══════════════════════════════════════════════
#  OSS 上传
# ══════════════════════════════════════════════
class OssUploader:
    """阿里云 OSS 上传"""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        oss_cfg = cfg.get("oss")
        if not oss_cfg:
            raise ValueError("配置文件中缺少 oss 配置")
        self.endpoint = oss_cfg["endpoint"]
        self.bucket_name = oss_cfg["bucket"]
        self.access_key_id = oss_cfg["access_key_id"]
        self.access_key_secret = oss_cfg["access_key_secret"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            import oss2
            auth = oss2.Auth(self.access_key_id, self.access_key_secret)
            self._client = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        return self._client

    def upload(self, local_path: str, object_name: str = None,
                log_func=None) -> str:
        """
        上传文件到 OSS，返回带签名的下载链接（7 天有效）

        Args:
            local_path: 本地文件路径
            object_name: OSS 对象名（默认使用文件名）
            log_func: 日志回调函数，如 self.log
        """
        if object_name is None:
            object_name = Path(local_path).name

        local_path = Path(local_path)
        total_size = local_path.stat().st_size
        start_time = time.time()

        if log_func:
            log_func(f"  ☁️ OSS 上传开始: {object_name} ({total_size / 1024 / 1024:.1f} MB)")

        # 使用 tqdm 在终端显示实时进度条
        from tqdm import tqdm
        pbar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  ☁️ 上传 {object_name}",
            ncols=80,
            leave=False,
        )
        last_pct = [0]

        def _progress(consumed_bytes, _total_bytes=None):
            pbar.update(consumed_bytes - pbar.n)
            pct = int(consumed_bytes / total_size * 100) if total_size else 0
            if pct % 10 == 0 and pct > last_pct[0]:
                last_pct[0] = pct
                elapsed = time.time() - start_time
                speed = consumed_bytes / 1024 / 1024 / elapsed if elapsed > 0 else 0
                remain = (total_size - consumed_bytes) / speed / 60 if speed > 0 else 0
                pbar.set_postfix_str(f"{pct}% {speed:.1f}MB/s ~{remain:.0f}min")
                if log_func:
                    log_func(f"  ☁️ OSS 上传: {pct}% ({consumed_bytes / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB, {speed:.1f} MB/s, 剩余 ~{remain:.0f}分钟)")

        bucket = self._get_client()
        try:
            result = bucket.put_object_from_file(object_name, str(local_path),
                                                  progress_callback=_progress)
        finally:
            pbar.close()

        # 检查上传结果
        if result is None or (hasattr(result, 'status') and result.status not in (200, 201, 204)):
            status_code = getattr(result, 'status', 'unknown')
            raise RuntimeError(f"OSS 返回异常状态码: {status_code}")

        elapsed = time.time() - start_time
        speed = total_size / 1024 / 1024 / elapsed if elapsed > 0 else 0
        if log_func:
            log_func(f"  ✅ OSS 上传完成: {total_size / 1024 / 1024:.1f} MB, 用时 {elapsed:.0f}s ({speed:.1f} MB/s)")

        # 生成签名下载链接（7 天有效）
        download_url = bucket.sign_url('GET', object_name, 7 * 86400)
        if log_func:
            log_func(f"  🔗 下载链接已生成 (7天有效)")
        return download_url


# ══════════════════════════════════════════════
#  系统资源监控
# ══════════════════════════════════════════════
def get_disk_usage(path: str) -> dict:
    """获取磁盘使用率"""
    usage = shutil.disk_usage(path)
    pct = usage.used / usage.total * 100
    return {
        "total_gb": usage.total / (1024 ** 3),
        "used_gb": usage.used / (1024 ** 3),
        "free_gb": usage.free / (1024 ** 3),
        "pct": pct,
    }


def get_memory_usage() -> dict:
    """获取内存使用率（优先 psutil，回退解析 /proc/meminfo）"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {"total_gb": mem.total / (1024 ** 3), "used_gb": mem.used / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3), "pct": mem.percent}
    except ImportError:
        pass

    # Linux fallback
    try:
        with open("/proc/meminfo") as f:
            data = {}
            for line in f:
                parts = line.split()
                if parts[0].rstrip(":") in ("MemTotal", "MemAvailable", "MemFree"):
                    data[parts[0].rstrip(":")] = int(parts[1]) * 1024
        total = data.get("MemTotal", 0)
        available = data.get("MemAvailable", data.get("MemFree", 0))
        used = total - available
        pct = used / total * 100 if total else 0
        return {"total_gb": total / (1024 ** 3), "used_gb": used / (1024 ** 3),
                "available_gb": available / (1024 ** 3), "pct": pct}
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "pct": 0}


def get_load_avg() -> str:
    """获取系统负载"""
    try:
        import psutil
        return f"{psutil.getloadavg()[0]:.1f}"
    except ImportError:
        try:
            with open("/proc/loadavg") as f:
                return f.read().split()[0]
        except Exception:
            return "N/A"


# ══════════════════════════════════════════════
#  采集进度解析
# ══════════════════════════════════════════════
def count_dataset_images() -> tuple:
    """统计 final_dataset 的图片总数和角色数"""
    total = 0
    roles = 0
    if not FINAL_DATASET_DIR.exists():
        return 0, 0
    for d in FINAL_DATASET_DIR.iterdir():
        if d.is_dir():
            roles += 1
            total += sum(1 for f in d.iterdir()
                         if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    return total, roles


# ══════════════════════════════════════════════
#  主逻辑
# ══════════════════════════════════════════════
class CollectorRunner:
    def __init__(self, args):
        self.args = args

        # 飞书通知
        self.notifier = FeishuNotifier(args.feishu_config) if args.feishu_config else None

        # OSS 上传
        self.oss_uploader: Optional[OssUploader] = None
        if args.oss and args.feishu_config:
            try:
                self.oss_uploader = OssUploader(args.feishu_config)
                self.log("  ✅ OSS 上传器已初始化")
            except Exception as e:
                self.log(f"  ⚠️ OSS 初始化失败（跳过上传）: {e}")
        elif args.oss and not args.feishu_config:
            self.log("  ⚠️ 未指定配置文件，OSS 上传已禁用")

        # 子进程
        self.process: Optional[subprocess.Popen] = None
        self.stop_event = threading.Event()
        self.last_progress_time = 0
        self.start_time = time.time()

        # 资源告警状态（避免重复推送）
        self._mem_warned = False
        self._disk_warned = False
        self._mem_critical = False
        self._disk_critical = False

        # 日志目录
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 日志 ──
    def log(self, msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(COLLECTOR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── 飞书 ──
    def send_feishu(self, title: str, message: str):
        if self.notifier:
            ok = self.notifier.send(title, message)
            if ok:
                self.log(f"  📨 飞书通知已发送: {title}")
            else:
                self.log(f"  ⚠️ 飞书通知发送失败: {title}")
        else:
            self.log(f"  (飞书未配置，跳过通知: {title})")

    # ── 资源监控线程 ──
    def _resource_monitor(self):
        """定时检查内存和磁盘，告警或自动停止"""
        while not self.stop_event.is_set():
            time.sleep(RESOURCE_CHECK_INTERVAL)

            disk = get_disk_usage(str(FINAL_DATASET_DIR.parent))
            mem = get_memory_usage()
            load = get_load_avg()

            # 磁盘告警
            if disk["pct"] >= DISK_CRITICAL_PCT and not self._disk_critical:
                self._disk_critical = True
                pct_before = disk["pct"]  # 记录清理前的值
                used_before = disk["used_gb"]
                # 先尝试自动清理（告急模式: aggressive=True）
                cleanup_result = self._auto_cleanup(aggressive=True)
                disk = get_disk_usage(str(FINAL_DATASET_DIR.parent))  # 重新获取磁盘状态
                msg = (f"🚨 **磁盘空间告急！**\n"
                       f"清理前: {used_before:.1f}GB / {disk['total_gb']:.1f}GB ({pct_before:.1f}%)\n"
                       f"清理后: {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB ({disk['pct']:.1f}%)\n")
                if cleanup_result and cleanup_result.get("total_freed_mb", 0) > 0:
                    msg += f"🧹 自动清理: +{cleanup_result['total_freed_mb']}MB\n"
                msg += f"⚠️ 已触发自动保护，停止采集..."
                self.send_feishu("🚨 磁盘空间告急", msg)
                self.log(f"  ⛔ 磁盘 {pct_before:.1f}% ≥ {DISK_CRITICAL_PCT}%，已清理 +{cleanup_result.get('total_freed_mb', 0)}MB，触发自动停止")
                self._stop_collector(reason="磁盘空间不足")
                return
            elif disk["pct"] >= DISK_WARN_PCT and not self._disk_warned:
                self._disk_warned = True
                # 自动清理
                cleanup_result = self._auto_cleanup()
                disk = get_disk_usage(str(FINAL_DATASET_DIR.parent))
                msg = (f"⚠️ **磁盘空间预警**\n"
                       f"已用: {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB ({disk['pct']:.1f}%)\n")
                if cleanup_result and cleanup_result.get("total_freed_mb", 0) > 0:
                    msg += f"🧹 自动清理: +{cleanup_result['total_freed_mb']}MB\n"
                msg += f"💾 当前可用: {disk['free_gb']:.1f}GB"
                self.send_feishu("⚠️ 磁盘空间预警", msg)

            # 内存告警
            if mem["pct"] >= MEMORY_CRITICAL_PCT and not self._mem_critical:
                self._mem_critical = True
                msg = (f"🚨 **内存告急！**\n"
                       f"已用: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['pct']:.1f}%)\n"
                       f"⚠️ 即将自动停止采集...")
                self.send_feishu("🚨 内存告急", msg)
                self.log(f"  ⛔ 内存 {mem['pct']:.1f}% ≥ {MEMORY_CRITICAL_PCT}%，触发自动停止")
                self._stop_collector(reason="内存不足")
                return
            elif mem["pct"] >= MEMORY_WARN_PCT and not self._mem_warned:
                self._mem_warned = True
                msg = (f"⚠️ **内存预警**\n"
                       f"已用: {mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB ({mem['pct']:.1f}%)")
                self.send_feishu("⚠️ 内存预警", msg)

            # 资源日志
            self.log(
                f"  📊 资源: 内存 {mem['pct']:.0f}% | 磁盘 {disk['pct']:.0f}% "
                f"({disk['free_gb']:.1f}GB 剩余) | 负载 {load}"
            )

    # ── 进度推送线程 ──
    def _progress_reporter(self):
        """定时推送采集进度到飞书"""
        while not self.stop_event.is_set():
            time.sleep(PROGRESS_INTERVAL)

            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed // 3600)}h{int((elapsed % 3600) // 60)}m"
            img_count, role_count = count_dataset_images()

            msg = (f"⏱️ 已运行: {elapsed_str}\n"
                   f"🖼️ 图片总数: {img_count}\n"
                   f"👤 角色目录: {role_count}\n")

            # 读取日志尾部获取最近状态
            if COLLECTOR_LOG.exists():
                try:
                    with open(COLLECTOR_LOG, "r") as f:
                        lines = f.readlines()
                    # 取最近 5 行包含关键字的日志
                    recent = [l.strip() for l in lines[-20:]
                              if any(kw in l for kw in ("✅", "❌", "跳过", "成功=", "final_dataset"))]
                    if recent:
                        msg += "\n最近进展:\n" + "\n".join(recent[-5:])
                except Exception:
                    pass

            self.send_feishu("📸 采集进展报告", msg)

    # ── 打包状态 ──
    def _load_pack_state(self) -> dict:
        """加载打包状态"""
        if PACK_STATE_FILE.exists():
            try:
                with open(PACK_STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"pack_number": 0, "packed_files": []}

    def _save_pack_state(self, state: dict):
        """保存打包状态"""
        PACK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PACK_STATE_FILE, "w") as f:
            json.dump(state, f)

    def _do_pack(self) -> bool:
        """打包 final_dataset 新增图片（增量打包），返回是否成功"""
        import zipfile

        img_count, role_count = count_dataset_images()
        state = self._load_pack_state()
        pack_no = state["pack_number"] + 1

        # 计算自上次打包后新增的图片
        packed_set = set(state.get("packed_files", []))
        new_files = []
        for d in FINAL_DATASET_DIR.iterdir():
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                        rel = str(f.relative_to(FINAL_DATASET_DIR))
                        if rel not in packed_set:
                            new_files.append((f, rel))

        if not new_files:
            self.log(f"  ℹ️ 无可打包的新增图片")
            return True

        # 输出 zip
        PACK_DIR.mkdir(parents=True, exist_ok=True)
        zip_name = f"dataset_pack_{pack_no:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = PACK_DIR / zip_name

        self.log(f"  📦 开始增量打包 #{pack_no} ({len(new_files)} 张新增, 累计 {img_count} 张 → {zip_name})")

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_path, arcname in new_files:
                    zf.write(img_path, arcname)
        except Exception as e:
            self.log(f"  ❌ 打包失败: {e}")
            return False

        size_mb = zip_path.stat().st_size / (1024 * 1024)

        # 更新状态：记录本次已打包的文件
        new_rel_paths = [rel for _, rel in new_files]
        packed_files = state.get("packed_files", [])
        packed_files.extend(new_rel_paths)
        state["pack_number"] = pack_no
        state["packed_files"] = packed_files
        state[f"pack_{pack_no}"] = {
            "zip": zip_name,
            "size_mb": round(size_mb, 1),
            "new_image_count": len(new_files),
            "total_image_count": img_count,
            "pack_files": new_rel_paths,  # 记录本次打包的文件列表，供后续清理使用
            "timestamp": datetime.now().isoformat(),
        }
        self._save_pack_state(state)

        self.log(f"  ✅ 打包完成: {zip_name} ({size_mb:.1f} MB, {img_count} 张)")
        return True, zip_path, pack_no

    # ── OSS 上传 + 本地清理 ──
    def _upload_and_cleanup(self, zip_path: Path, pack_no: int) -> Tuple[Optional[str], bool]:
        """
        上传到 OSS → 返回 (oss_url, upload_success)
        清理旧本地包 → 清理对应状态元数据
        """
        oss_url = None
        upload_success = False

        # 上传到 OSS
        if self.oss_uploader:
            self.log(f"  ☁️ 正在上传到 OSS...")
            try:
                object_name = f"dataset_pack_{pack_no:03d}.zip"
                oss_url = self.oss_uploader.upload(str(zip_path), object_name, log_func=self.log)
                upload_success = True
                self.log(f"  ✅ OSS 上传成功")

                # 记录 OSS 信息到状态
                state = self._load_pack_state()
                key = f"pack_{pack_no}"
                if key in state:
                    state[key]["oss_url"] = oss_url
                    state[key]["oss_object"] = object_name
                    self._save_pack_state(state)
            except Exception as e:
                self.log(f"  ❌ OSS 上传失败: {e}")

        # 清理旧包 + 旧元数据：只保留最近 N 个
        all_zips = sorted(PACK_DIR.glob("dataset_pack_*.zip"), key=lambda p: p.stat().st_mtime)
        deleted_packs = set()
        while len(all_zips) > MAX_LOCAL_PACKS:
            oldest = all_zips.pop(0)
            self.log(f"  🗑️ 清理旧包: {oldest.name} ({oldest.stat().st_size / 1024 / 1024:.1f} MB)")
            # 记录被删的包号用于清理元数据
            m = re.search(r'pack_(\d+)', oldest.name)
            if m:
                deleted_packs.add(f"pack_{int(m.group(1))}")
            oldest.unlink(missing_ok=True)

        if not deleted_packs:
            self.log(f"  ℹ️ 旧包清理: 本地包数 ({len(all_zips)}) 未超过限制 ({MAX_LOCAL_PACKS})，无需清理")

        # 清理状态文件中对应的旧元数据
        if deleted_packs:
            state = self._load_pack_state()
            changed = False
            for k in list(state.keys()):
                if k in deleted_packs:
                    del state[k]
                    changed = True
                # 也清理 oss_object / oss_url 单独字段（如果后来改成顶层字段）
                if k == f"oss_url_{pack_no}" or k == f"oss_object_{pack_no}":
                    del state[k]
                    changed = True
            if changed:
                self._save_pack_state(state)
                self.log(f"  🧹 已清理状态元数据: {', '.join(sorted(deleted_packs))}")

        # 上传成功后，清理 final_dataset 中已打包的原图
        if upload_success:
            self._clean_packed_data(pack_no)

        return oss_url, upload_success

    # ── 清理已打包的原图 ──
    def _clean_packed_data(self, pack_no: int) -> None:
        """
        打包+上传成功后，删除 final_dataset 中已打包的原始图片以释放磁盘空间。
        只删除本次打包的文件（pack_files），不影响其他角色和历史数据。
        """
        state = self._load_pack_state()
        key = f"pack_{pack_no}"
        if key not in state:
            self.log(f"  ⚠️ 状态中找不到 {key}，跳过原图清理")
            return

        pack_files = state[key].get("pack_files", [])
        if not pack_files:
            self.log(f"  ℹ️ 本次打包无新增文件，跳过原图清理")
            return

        deleted = 0
        freed = 0
        for rel_path in pack_files:
            fp = FINAL_DATASET_DIR / rel_path
            if fp.exists():
                try:
                    size = fp.stat().st_size
                    fp.unlink()
                    deleted += 1
                    freed += size
                except Exception as e:
                    self.log(f"    ⚠️ 删除失败 {rel_path}: {e}")

        freed_mb = freed / (1024 * 1024)
        self.log(f"  🧹 原图清理: 已删除 {deleted}/{len(pack_files)} 张 ({freed_mb:.1f} MB 释放) "
                 f"— 图片已打包到包 #{pack_no}，原始文件已安全删除")
        if deleted > 0:
            # 更新状态：记录已清理的数量
            state[key]["cleaned_count"] = deleted
            state[key]["cleaned_mb"] = round(freed_mb, 1)
            self._save_pack_state(state)

    # ── 打包监控线程 ──
    def _pack_monitor(self):
        """检查图片增量，达到 PACK_INTERVAL 时自动打包并上传 OSS"""
        while not self.stop_event.is_set():
            time.sleep(PACK_INTERVAL_CHECK_INTERVAL)

            img_count, _ = count_dataset_images()
            state = self._load_pack_state()
            last_packed = len(state.get("packed_files", []))

            if img_count - last_packed >= PACK_INTERVAL:
                pack_ok, zip_path, pack_no = self._do_pack()
                if pack_ok:
                    # 上传 OSS + 清理旧包 + 清理元数据
                    oss_url, upload_success = self._upload_and_cleanup(zip_path, pack_no)

                    # 构造通知
                    new_state = self._load_pack_state()
                    pack_info = new_state.get(f"pack_{new_state['pack_number']}", {})
                    all_zips = list(sorted(PACK_DIR.glob("dataset_pack_*.zip")))

                    msg_parts = [
                        f"📦 **新数据集包已生成**",
                        f"包号: #{new_state['pack_number']}",
                        f"文件名: {pack_info.get('zip', '')}",
                        f"大小: {pack_info.get('size_mb', 0)} MB",
                        f"🖼️ 图片总数: {pack_info.get('image_count', 0)}",
                    ]

                    # 上传结果
                    if self.oss_uploader:
                        if upload_success:
                            msg_parts.append(f"☁️ **OSS 上传: ✅ 成功**")
                            msg_parts.append(f"🔗 下载链接 (7天有效): {oss_url}")
                        else:
                            msg_parts.append(f"☁️ **OSS 上传: ❌ 失败** — 包仅本地可用")
                            msg_parts.append(f"📂 本地路径: {PACK_DIR / pack_info.get('zip', '')}")
                    else:
                        msg_parts.append(f"📂 本地路径: {PACK_DIR / pack_info.get('zip', '')}")

                    # 本地保留情况
                    kept_count = len(all_zips)
                    if kept_count > 0:
                        msg_parts.append(f"💾 本地保留: {kept_count} 个最新包 (packs/)")
                    if kept_count < new_state['pack_number']:
                        msg_parts.append(f"🗑️ 旧包文件及 .pack_state.json 历史元数据已自动清理")

                    self.send_feishu("📦 数据集可下载", "\n".join(msg_parts))

    # ── 自动清理 ──
    def _auto_cleanup(self, aggressive: bool = False) -> Optional[dict]:
        """磁盘预警时自动执行系统清理，返回清理报告"""
        level = "🧹 磁盘预警，自动执行系统清理" if not aggressive else "🚨 磁盘告急，执行强力清理..."
        self.log(f"  {level}")
        try:
            result = run_cleanup(dry_run=False, aggressive=aggressive)

            # 输出每个清理步骤的详细日志
            for s in result.get("steps", []):
                icon = "✅" if s["ok"] else "⚠️"
                freed = s.get("freed_mb", 0)
                detail = s.get("detail", "").strip()
                if freed > 0:
                    self.log(f"    {icon} {s['name']}: +{freed}MB — {detail[:100]}")
                elif detail and detail != "无需处理":
                    self.log(f"    {icon} {s['name']}: {detail[:100]}")
                else:
                    self.log(f"    {icon} {s['name']}: 无需处理")

            # 大文件处理结果
            large_files = result.get("large_files", [])
            if large_files:
                total_large_mb = sum(item[0] for item in large_files)
                self.log(f"    📊 识别可清理大文件: {len(large_files)} 项, 共约 {total_large_mb}MB")
                for mb, path, cat, desc in large_files[:5]:
                    self.log(f"      • {desc}: {path} (~{mb}MB)")

            if result["total_freed_mb"] > 0:
                self.log(f"  ✅ 自动清理完成: +{result['total_freed_mb']}MB 释放 (磁盘: {result['free_before_gb']}GB → {result['free_after_gb']}GB)")
            else:
                self.log(f"  ℹ️ 自动清理: 无可释放空间 (当前磁盘: {result['free_after_gb']}GB 可用)")

            # 推送清理报告到飞书（仅释放 > 0 时）
            if result["total_freed_mb"] > 0 and self.notifier:
                report_text = format_report(result)
                title = "🚨 系统强力清理" if aggressive else "🧹 已自动清理系统"
                self.send_feishu(title, report_text)

            return result
        except Exception as e:
            self.log(f"  ⚠️ 自动清理异常: {e}")
            return None

    # ── 停止采集 ──
    def _stop_collector(self, reason: str = ""):
        """停止采集子进程"""
        if self.process and self.process.poll() is None:
            self.log(f"  ⛔ 正在停止采集进程 (PID={self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed // 3600)}h{int((elapsed % 3600) // 60)}m"
            img_count, role_count = count_dataset_images()

            alert_msg = (f"🛑 **采集已停止**\n"
                         f"原因: {reason or '手动停止'}\n"
                         f"⏱️ 运行时长: {elapsed_str}\n"
                         f"🖼️ 最终图片: {img_count}\n"
                         f"👤 角色目录: {role_count}")
            self.send_feishu("🛑 采集已停止", alert_msg)

        self.stop_event.set()

    # ── 启动采集 ──
    def run(self):
        # 1. 启动通知
        img_count, role_count = count_dataset_images()
        start_msg = (f"🔄 **采集任务启动**\n"
                     f"时间: {TIMESTAMP}\n"
                     f"当前数据: {img_count} 张 / {role_count} 角色\n"
                     f"日志: {COLLECTOR_LOG.name}\n"
                     f"资源阈值: 内存>{MEMORY_CRITICAL_PCT}% / 磁盘>{DISK_CRITICAL_PCT}% 自动停机")
        self.send_feishu("🔄 采集任务启动", start_msg)

        # 2. 构建采集命令
        cmd = [sys.executable, str(COLLECTOR_SCRIPT)]
        if self.args.max_count:
            cmd.extend(["--max-count", str(self.args.max_count)])
        if self.args.skip_existing:
            cmd.append("--skip-existing")
        if self.args.site:
            cmd.extend(["--site", self.args.site])
        if self.args.workers:
            cmd.extend(["--workers", str(self.args.workers)])

        self.log(f"🚀 启动采集: {' '.join(cmd)}")

        # 3. 启动子进程
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
            )
        except Exception as e:
            self.log(f"❌ 启动采集失败: {e}")
            self.send_feishu("❌ 采集启动失败", str(e))
            sys.exit(1)

        # 4. 启动监控线程
        monitor_thread = threading.Thread(target=self._resource_monitor, daemon=True)
        monitor_thread.start()

        progress_thread = threading.Thread(target=self._progress_reporter, daemon=True)
        progress_thread.start()

        pack_thread = threading.Thread(target=self._pack_monitor, daemon=True)
        pack_thread.start()

        # 5. 实时输出子进程日志 + 同时写入文件
        with open(COLLECTOR_LOG, "w", encoding="utf-8") as log_fp:
            log_fp.write(f"# 采集日志 {TIMESTAMP}\n# {' '.join(cmd)}\n\n")
            log_fp.flush()

            for line in iter(self.process.stdout.readline, ""):
                line = line.rstrip("\n")
                if line:
                    print(f"  {line}")
                    log_fp.write(line + "\n")
                    log_fp.flush()

        # 6. 等待子进程结束
        returncode = self.process.wait()
        self.stop_event.set()

        # 7. 完成通知
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 3600)}h{int((elapsed % 3600) // 60)}m"
        img_count, role_count = count_dataset_images()

        if returncode == 0:
            summary = (f"✅ **采集任务完成**\n"
                       f"⏱️ 运行时长: {elapsed_str}\n"
                       f"🖼️ 图片总数: {img_count}\n"
                       f"👤 角色目录: {role_count}\n"
                       f"📄 日志: {COLLECTOR_LOG.name}")
            self.send_feishu("✅ 采集任务完成", summary)
            self.log(f"✅ 采集完成! 退出码={returncode}")
        else:
            if not self._disk_critical and not self._mem_critical:
                summary = (f"⚠️ **采集异常退出**\n"
                           f"退出码: {returncode}\n"
                           f"⏱️ 运行时长: {elapsed_str}\n"
                           f"🖼️ 当前图片: {img_count}\n"
                           f"📄 日志: {COLLECTOR_LOG.name}")
                self.send_feishu("⚠️ 采集异常退出", summary)
                self.log(f"⚠️ 采集异常退出，退出码={returncode}")


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="一键采集启动器（含飞书推送 + 资源监控）")
    parser.add_argument("--feishu-config", type=str, default=str(DEFAULT_FEISHU_CONFIG),
                        help=f"飞书配置文件路径 (默认: {DEFAULT_FEISHU_CONFIG})")
    parser.add_argument("--max-count", type=int, default=None,
                        help="每个角色目标张数 (默认: 采集脚本默认值)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有≥max_count张的角色")
    parser.add_argument("--site", type=str, default=None,
                        help="首选站点 (默认: 采集脚本默认值)")
    parser.add_argument("--workers", type=int, default=None,
                        help="并发下载线程数 (默认: 采集脚本默认值)")
    parser.add_argument("--no-feishu", action="store_true",
                        help="禁用飞书推送")
    parser.add_argument("--no-oss", action="store_true",
                        help="禁用 OSS 上传 (默认启用，需配置文件含 oss 配置)")
    parser.add_argument("--memory-critical", type=float, default=MEMORY_CRITICAL_PCT,
                        help=f"内存告急阈值%% (默认: {MEMORY_CRITICAL_PCT})")
    parser.add_argument("--disk-critical", type=float, default=DISK_CRITICAL_PCT,
                        help=f"磁盘告急阈值%% (默认: {DISK_CRITICAL_PCT})")
    args = parser.parse_args()

    # 禁用飞书
    if args.no_feishu:
        args.feishu_config = None

    # 禁用 OSS
    args.oss = not args.no_oss

    # 检查配置文件
    if args.feishu_config and not Path(args.feishu_config).exists():
        print(f"⚠️ 飞书配置文件不存在: {args.feishu_config}，将禁用消息推送")
        args.feishu_config = None

    # 检查采集脚本
    if not COLLECTOR_SCRIPT.exists():
        print(f"❌ 采集脚本不存在: {COLLECTOR_SCRIPT}")
        sys.exit(1)

    # 启动
    runner = CollectorRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
        runner._stop_collector(reason="用户中断")
    except Exception as e:
        print(f"\n❌ 运行时异常: {e}")
        runner.send_feishu("❌ 采集运行异常", str(e))


if __name__ == "__main__":
    main()