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
    - 资源告急时自动停止采集并推送告警
"""

import os
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
from typing import Optional

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
                msg = (f"🚨 **磁盘空间告急！**\n"
                       f"已用: {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB ({disk['pct']:.1f}%)\n"
                       f"⚠️ 即将自动停止采集...")
                self.send_feishu("🚨 磁盘空间告急", msg)
                self.log(f"  ⛔ 磁盘 {disk['pct']:.1f}% ≥ {DISK_CRITICAL_PCT}%，触发自动停止")
                self._stop_collector(reason="磁盘空间不足")
                return
            elif disk["pct"] >= DISK_WARN_PCT and not self._disk_warned:
                self._disk_warned = True
                msg = (f"⚠️ **磁盘空间预警**\n"
                       f"已用: {disk['used_gb']:.1f}GB / {disk['total_gb']:.1f}GB ({disk['pct']:.1f}%)\n"
                       f"请及时清理空间")
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
    parser.add_argument("--memory-critical", type=float, default=MEMORY_CRITICAL_PCT,
                        help=f"内存告急阈值%% (默认: {MEMORY_CRITICAL_PCT})")
    parser.add_argument("--disk-critical", type=float, default=DISK_CRITICAL_PCT,
                        help=f"磁盘告急阈值%% (默认: {DISK_CRITICAL_PCT})")
    args = parser.parse_args()

    # 禁用飞书
    if args.no_feishu:
        args.feishu_config = None

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