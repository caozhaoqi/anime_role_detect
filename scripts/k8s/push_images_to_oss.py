#!/usr/bin/env python3
"""
将本地构建的 Docker 镜像导出为 tar.gz 并上传到阿里云 OSS

用法:
    python push_images_to_oss.py                    # 使用当前 git commit 作为 TAG
    python push_images_to_oss.py --tag c8cd26b      # 指定 TAG
    python push_images_to_oss.py --registry ardc    # 指定镜像前缀
    python push_images_to_oss.py --prefix docker-images/2026-07-02/  # 指定 OSS 路径前缀
    python push_images_to_oss.py --skip-save        # 跳过 docker save（已有 tar.gz 时直接上传）
    python push_images_to_oss.py --dry-run          # 只列出不执行
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "scripts" / "notification_config.json"
DEFAULT_EXPORT_DIR = "/tmp/docker-images-export"

SERVICES = [
    "base",
    "ml-base",
    "api-service",
    "model-service",
    "frontend",
    "api-gateway",
    "multimedia-service",
    "search-service",
    "search-worker",
    "inference-worker",
    "monitoring",
]


def log(msg: str):
    print(msg, flush=True)


def get_git_tag() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return time.strftime("%Y%m%d%H%M%S")


def load_oss_config(config_path: Path) -> dict:
    if not config_path.exists():
        log(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    oss_cfg = cfg.get("oss")
    if not oss_cfg:
        log("❌ 配置文件中缺少 oss 配置")
        sys.exit(1)
    return oss_cfg


def find_existing_images(registry: str, tag: str) -> list[str]:
    """检查哪些镜像已经存在于本地"""
    found = []
    for svc in SERVICES:
        image_name = f"{registry}/{svc}:{tag}"
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True
        )
        if result.returncode == 0:
            found.append(image_name)
        else:
            log(f"  ⚠️  镜像不存在，跳过: {image_name}")
    return found


def save_images(images: list[str], tag: str, export_dir: str) -> list[tuple[str, str]]:
    """docker save 导出镜像为 tar.gz，返回 [(image_name, tar_path), ...]"""
    os.makedirs(export_dir, exist_ok=True)
    saved = []

    for image_name in images:
        svc_name = image_name.split("/")[1].split(":")[0]
        tar_name = f"{svc_name}-{tag}.tar.gz"
        tar_path = os.path.join(export_dir, tar_name)

        log(f"  📦 导出 {image_name} → {tar_path}")
        start = time.time()

        # docker save | gzip > tar.gz
        docker_proc = subprocess.Popen(
            ["docker", "save", image_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        gzip_proc = subprocess.Popen(
            ["gzip", "-1"],
            stdin=docker_proc.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        docker_proc.stdout.close()

        with open(tar_path, "wb") as f:
            while True:
                chunk = gzip_proc.stdout.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        docker_proc.wait()
        gzip_proc.wait()

        size_mb = os.path.getsize(tar_path) / 1024 / 1024
        elapsed = time.time() - start
        log(f"  ✅ 导出完成: {size_mb:.1f} MB, 用时 {elapsed:.0f}s")
        saved.append((image_name, tar_path))

    return saved


def upload_to_oss(oss_cfg: dict, tar_files: list[tuple[str, str]],
                  prefix: str, dry_run: bool = False):
    """上传 tar.gz 文件到 OSS"""
    import oss2

    auth = oss2.Auth(oss_cfg["access_key_id"], oss_cfg["access_key_secret"])
    bucket = oss2.Bucket(auth, oss_cfg["endpoint"], oss_cfg["bucket"])

    for image_name, tar_path in tar_files:
        filename = os.path.basename(tar_path)
        object_key = f"{prefix}{filename}"
        size_mb = os.path.getsize(tar_path) / 1024 / 1024

        if dry_run:
            log(f"  [DRY RUN] 将上传: {object_key} ({size_mb:.1f} MB)")
            continue

        log(f"  ☁️  上传 {filename} ({size_mb:.1f} MB) → oss://{oss_cfg['bucket']}/{object_key}")
        start = time.time()
        last_pct = [0]

        def progress(consumed, _total=None):
            pct = int(consumed / os.path.getsize(tar_path) * 100)
            if pct % 20 == 0 and pct > last_pct[0]:
                last_pct[0] = pct
                elapsed = time.time() - start
                speed = consumed / 1024 / 1024 / elapsed if elapsed > 0 else 0
                log(f"    ☁️  {pct}% ({consumed / 1024 / 1024:.1f} MB, {speed:.1f} MB/s)")

        try:
            bucket.put_object_from_file(object_key, tar_path, progress_callback=progress)
            elapsed = time.time() - start
            speed = size_mb / elapsed if elapsed > 0 else 0
            log(f"  ✅ 上传完成: {elapsed:.0f}s ({speed:.1f} MB/s)")

            url = bucket.sign_url("GET", object_key, 7 * 86400)
            log(f"  🔗 下载链接 (7天有效): {url}")
        except Exception as e:
            log(f"  ❌ 上传失败: {e}")


def cleanup(tar_files: list[tuple[str, str]]):
    """清理本地 tar.gz 文件"""
    for _, tar_path in tar_files:
        if os.path.exists(tar_path):
            os.remove(tar_path)
            log(f"  🗑️  已清理: {tar_path}")


def main():
    parser = argparse.ArgumentParser(description="导出 Docker 镜像并上传到阿里云 OSS")
    parser.add_argument("--tag", default=None, help="镜像 TAG (默认: git short hash)")
    parser.add_argument("--registry", default="ardc", help="镜像前缀 (默认: ardc)")
    parser.add_argument("--prefix", default=None,
                        help="OSS 路径前缀 (默认: docker-images/<tag>/)")
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR, help="本地导出目录")
    parser.add_argument("--skip-save", action="store_true", help="跳过 docker save，直接使用已有 tar.gz")
    parser.add_argument("--no-cleanup", action="store_true", help="上传后不删除本地 tar.gz")
    parser.add_argument("--dry-run", action="store_true", help="只列出不执行")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="配置文件路径")
    args = parser.parse_args()

    tag = args.tag or get_git_tag()
    prefix = args.prefix or f"docker-images/{tag}/"
    if not prefix.endswith("/"):
        prefix += "/"

    log("=" * 50)
    log("  Docker 镜像 → 阿里云 OSS 推送工具")
    log("=" * 50)
    log(f"  镜像 TAG:    {tag}")
    log(f"  镜像前缀:    {args.registry}")
    log(f"  OSS 路径:    {prefix}")
    log(f"  导出目录:    {args.export_dir}")
    log(f"  配置文件:    {args.config}")
    log(f"  Dry Run:     {args.dry_run}")
    log("=" * 50)

    # 加载 OSS 配置
    config_path = Path(args.config)
    oss_cfg = load_oss_config(config_path)
    log(f"  OSS Endpoint: {oss_cfg['endpoint']}")
    log(f"  OSS Bucket:   {oss_cfg['bucket']}")
    log("")

    # 查找本地镜像或已有 tar.gz
    if args.skip_save:
        log("📋 Phase 1: 查找已有 tar.gz 文件...")
        tar_files = []
        for svc in SERVICES:
            tar_name = f"{svc}-{tag}.tar.gz"
            tar_path = os.path.join(args.export_dir, tar_name)
            if os.path.exists(tar_path):
                image_name = f"{args.registry}/{svc}:{tag}"
                tar_files.append((image_name, tar_path))
                size_mb = os.path.getsize(tar_path) / 1024 / 1024
                log(f"  ✅ {tar_name} ({size_mb:.1f} MB)")
            else:
                log(f"  ⚠️  文件不存在，跳过: {tar_path}")
        if not tar_files:
            log(f"❌ 在 {args.export_dir} 中没有找到任何 tar.gz 文件")
            sys.exit(1)
    else:
        log("📋 Phase 1: 检查本地 Docker 镜像...")
        images = find_existing_images(args.registry, tag)
        if not images:
            log("❌ 没有找到任何本地镜像，请先运行 build_k8s_images.sh")
            sys.exit(1)
        log(f"  找到 {len(images)} 个镜像\n")

        log(f"📦 Phase 2: 导出镜像 ({len(images)} 个)...")
        tar_files = save_images(images, tag, args.export_dir)

    total_mb = sum(os.path.getsize(p) for _, p in tar_files) / 1024 / 1024
    log(f"\n  总计 {len(tar_files)} 个文件, {total_mb:.1f} MB\n")

    # 上传到 OSS
    log("☁️  Phase 3: 上传到 OSS...")
    upload_to_oss(oss_cfg, tar_files, prefix, dry_run=args.dry_run)

    # 清理
    if not args.dry_run and not args.no_cleanup and not args.skip_save:
        log("\n🧹 Phase 4: 清理本地文件...")
        cleanup(tar_files)

    log("\n" + "=" * 50)
    log("✅ 全部完成!")
    log("=" * 50)


if __name__ == "__main__":
    main()
