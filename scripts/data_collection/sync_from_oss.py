"""
从阿里云 OSS 同步数据集包到本地

功能:
    - 列出 OSS Bucket 中所有 dataset_pack_*.zip
    - 增量下载（跳过已下载的包）
    - 自动解压到 data/final_dataset（保持 角色目录/图片 结构）
    - 可选飞书推送

用法:
    .venv/bin/python3 scripts/data_collection/sync_from_oss.py \\
        --feishu-config scripts/notification_config.json \\
        --dry-run
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from loguru import logger

import oss2

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "scripts" / "notification_config.json"
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
PACK_DIR = PROJECT_ROOT / "packs"
SYNC_STATE_FILE = PROJECT_ROOT / "data" / ".oss_sync_state.json"


def get_bucket(config_path: str):
    """获取 OSS Bucket 对象"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    oss_cfg = cfg.get("oss")
    if not oss_cfg:
        raise ValueError("配置文件中缺少 oss 配置")

    auth = oss2.Auth(oss_cfg["access_key_id"], oss_cfg["access_key_secret"])
    return oss2.Bucket(auth, oss_cfg["endpoint"], oss_cfg["bucket"])


def list_packs_on_oss(bucket) -> list:
    """列出 OSS 上所有 dataset_pack_*.zip，返回 [(pack_no, object_name, size, last_modified), ...]"""
    packs = []
    # 加上 delimiter，一次最多 1000 个
    for obj in oss2.ObjectIteratorV2(bucket):
        name = obj.key
        m = re.search(r"dataset_pack_(\d+)", name)
        if m and name.endswith(".zip"):
            packs.append((
                int(m.group(1)),
                name,
                obj.size,
                obj.last_modified,
            ))
    packs.sort(key=lambda x: x[0])  # 按包号升序
    return packs


def load_sync_state() -> dict:
    """加载同步状态"""
    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"downloaded_packs": [], "downloaded_files": []}


def save_sync_state(state: dict):
    """保存同步状态"""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def download_pack(bucket, object_name: str, dest_dir: Path, log_func=print) -> Path:
    """下载 OSS 包到本地，返回本地路径"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / object_name

    # 流式下载 + 进度
    total_size = bucket.get_object_meta(object_name).content_length
    log_func(f"  ⬇️ 下载 {object_name} ({total_size / 1024 / 1024:.1f} MB)")

    start = time.time()
    last_pct = [0]

    def _progress(consumed, total):
        if total > 0:
            pct = int(consumed / total * 100)
            if pct % 10 == 0 and pct > last_pct[0]:
                last_pct[0] = pct
                elapsed = time.time() - start
                speed = consumed / 1024 / 1024 / elapsed if elapsed > 0 else 0
                remain = (total - consumed) / speed / 60 if speed > 0 else 0
                log_func(f"     {pct}% ({consumed/1024/1024:.0f}/{total/1024/1024:.0f} MB, {speed:.1f} MB/s, ~{remain:.0f}min)")

    bucket.get_object_to_file(object_name, str(local_path), progress_callback=_progress)
    elapsed = time.time() - start
    speed = total_size / 1024 / 1024 / elapsed if elapsed > 0 else 0
    log_func(f"  ✅ 下载完成: {local_path.name} ({elapsed:.0f}s, {speed:.1f} MB/s)")
    return local_path


def unzip_pack(zip_path: Path, extract_to: Path, log_func=print) -> list:
    """解压包到目标目录，返回解压的文件列表（相对路径）"""
    import zipfile
    extract_to.mkdir(parents=True, exist_ok=True)

    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        log_func(f"  📦 解压 {len(names)} 个文件到 {extract_to}")
        for name in names:
            # 安全解压（防止路径穿越）
            target = extract_to / name
            target.parent.mkdir(parents=True, exist_ok=True)
            zf.extract(name, extract_to)
            extracted.append(name)
    log_func(f"  ✅ 解压完成: {len(extracted)} 个文件")
    return extracted


def send_feishu_notification(config_path: str, title: str, content: str):
    """发送飞书通知"""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from common.notification_utils import FeishuNotifier
        notifier = FeishuNotifier(config_path)
        notifier.send(title, content)
    except Exception as e:
        logger.error(f"  ⚠️ 飞书通知失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="从 OSS 同步数据集到本地")
    parser.add_argument("--feishu-config", default=str(DEFAULT_CONFIG),
                        help="飞书/OSS 配置文件路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅列出 OSS 上的包，不下载")
    args = parser.parse_args()

    config_path = args.feishu_config
    dry_run = args.dry_run

    log = logger.info

    log("=" * 60)
    log("OSS → 本地 数据集同步")
    log("=" * 60)

    # 连接 OSS
    log("\n🔌 连接 OSS...")
    try:
        bucket = get_bucket(config_path)
        log(f"  ✅ 已连接: {bucket.bucket_name}")
    except Exception as e:
        log(f"  ❌ 连接失败: {e}")
        sys.exit(1)

    # 列出 OSS 上的包
    log("\n📋 列出 OSS 上的数据包...")
    packs = list_packs_on_oss(bucket)
    if not packs:
        log("  ℹ️ OSS 上没有 dataset_pack_*.zip")
        return

    for no, name, size, mtime in packs:
        log(f"  #{no:3d}  {name}  ({size/1024/1024:.1f} MB)")

    # 检查本地同步状态
    state = load_sync_state()
    downloaded = set(state.get("downloaded_packs", []))
    pending = [(no, name, size, mtime) for no, name, size, mtime in packs
               if no not in downloaded]

    if dry_run:
        log(f"\n🔍 Dry-run 模式，共 {len(pending)} 个待同步包:")
        for no, name, size, mtime in pending:
            log(f"  #{no:3d}  {name}  ({size/1024/1024:.1f} MB)  → 待下载")
        if not pending:
            log("  全部已同步")
        return

    if not pending:
        log(f"\n✅ 全部已同步（共 {len(downloaded)} 个包）")
        return

    log(f"\n⬇️ 待同步: {len(pending)} 个包（{sum(s for _,_,s,_ in pending)/1024/1024:.1f} MB）")

    # 逐个下载 + 解压
    total_downloaded = 0
    total_extracted = 0
    for no, name, size, mtime in pending:
        log(f"\n--- #{no}: {name} ---")

        # 下载
        zip_path = download_pack(bucket, name, PACK_DIR, log)

        # 解压
        extracted = unzip_pack(zip_path, FINAL_DATASET_DIR, log)

        # 更新状态
        downloaded.add(no)
        state["downloaded_packs"] = sorted(downloaded)
        state["downloaded_files"].extend(extracted)
        save_sync_state(state)

        total_downloaded += 1
        total_extracted += len(extracted)

        log(f"  ✅ 包 #{no} 同步完成 ({len(extracted)} 张图片)")

    # 统计
    img_count = len(state.get("downloaded_files", []))
    log(f"\n{'=' * 60}")
    log(f"🎉 同步完成!")
    log(f"  下载包数: {total_downloaded}")
    log(f"  图片总数: {img_count}")
    log(f"  存放位置: {FINAL_DATASET_DIR}")

    # 飞书通知
    try:
        send_feishu_notification(
            config_path,
            "📥 OSS 数据同步完成",
            f"下载: {total_downloaded} 个包\n"
            f"图片: {img_count} 张\n"
            f"路径: {FINAL_DATASET_DIR}"
        )
        log("  📨 飞书通知已发送")
    except Exception as e:
        log(f"  ⚠️ 飞书通知失败: {e}")


if __name__ == "__main__":
    main()