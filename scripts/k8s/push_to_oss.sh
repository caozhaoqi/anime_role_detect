#!/bin/bash
# 导出 Docker 镜像并上传到阿里云 OSS
# 用法: bash push_to_oss.sh [TAG]

set -e

TAG=${1:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}
REGISTRY="ardc"
EXPORT_DIR="/tmp/docker-images-export"
OSS_PREFIX="docker-images/${TAG}"
CONFIG_FILE="$(cd "$(dirname "$0")/../.." && pwd)/scripts/notification_config.json"

echo "========================================"
echo "  Docker 镜像导出 & 上传 OSS"
echo "========================================"
echo "  TAG:      $TAG"
echo "  导出目录:  $EXPORT_DIR"
echo "  OSS 路径:  $OSS_PREFIX/"
echo "========================================"

mkdir -p "$EXPORT_DIR"

# 获取所有匹配的镜像
IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^${REGISTRY}/.*:${TAG}$" || true)
if [ -z "$IMAGES" ]; then
    echo "❌ 没有找到 ${REGISTRY}/*:${TAG} 镜像"
    exit 1
fi

echo ""
echo "📦 导出镜像..."
TAR_FILES=()
for img in $IMAGES; do
    name=$(echo "$img" | sed "s|${REGISTRY}/||;s|:|-|g")
    tar_file="${EXPORT_DIR}/${name}.tar.gz"
    size=$(docker image inspect "$img" --format='{{.Size}}' 2>/dev/null)
    size_mb=$(echo "scale=1; $size/1048576" | bc 2>/dev/null || echo "?")
    echo "  📦 ${img} (${size_mb} MB) → ${tar_file}"
    docker save "$img" | gzip -1 > "$tar_file"
    actual_size=$(du -sh "$tar_file" | awk '{print $1}')
    echo "  ✅ 完成 (${actual_size})"
    TAR_FILES+=("$tar_file")
done

echo ""
echo "☁️  上传到 OSS..."

python3 - "$CONFIG_FILE" "$OSS_PREFIX" "${TAR_FILES[@]}" <<'PYEOF'
import sys, json, os, time

config_file = sys.argv[1]
oss_prefix = sys.argv[2]
tar_files = sys.argv[3:]

with open(config_file) as f:
    cfg = json.load(f)["oss"]

import oss2
auth = oss2.Auth(cfg["access_key_id"], cfg["access_key_secret"])
bucket = oss2.Bucket(auth, cfg["endpoint"], cfg["bucket"])

for tar_path in tar_files:
    filename = os.path.basename(tar_path)
    key = f"{oss_prefix}/{filename}"
    size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f"  ☁️  上传 {filename} ({size_mb:.1f} MB) → oss://{cfg['bucket']}/{key}")
    start = time.time()
    last_pct = [0]
    def progress(consumed, _total=None):
        pct = int(consumed / os.path.getsize(tar_path) * 100)
        if pct % 25 == 0 and pct > last_pct[0]:
            last_pct[0] = pct
            elapsed = time.time() - start
            speed = consumed / 1024 / 1024 / elapsed if elapsed > 0 else 0
            print(f"    {pct}% ({speed:.1f} MB/s)")
    try:
        bucket.put_object_from_file(key, tar_path, progress_callback=progress)
        elapsed = time.time() - start
        speed = size_mb / elapsed if elapsed > 0 else 0
        url = bucket.sign_url("GET", key, 7 * 86400)
        print(f"  ✅ {elapsed:.0f}s ({speed:.1f} MB/s)")
        print(f"  🔗 {url}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
PYEOF

echo ""
echo "🧹 清理本地文件..."
rm -f "${EXPORT_DIR}"/*.tar.gz
echo "  ✅ 已清理"

echo ""
echo "========================================"
echo "✅ 全部完成!"
echo "========================================"
