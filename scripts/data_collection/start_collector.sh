#!/usr/bin/env bash
# ============================================================
# 一键启动采集脚本
#
# 用法:
#   bash scripts/data_collection/start_collector.sh                    # 默认参数启动
#   bash scripts/data_collection/start_collector.sh --skip-existing    # 跳过已有足够图片的角色
#   bash scripts/data_collection/start_collector.sh --no-feishu        # 不推送飞书消息
#   bash scripts/data_collection/start_collector.sh --max-count 50     # 每个角色采 50 张
#
# 前置条件:
#   - 已创建 Python 虚拟环境: .venv/
#   - 已构建哈希数据库: 运行 build_hash_db.py
# ============================================================

set -e

# ── 项目根目录 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════╗"
echo "║      📸 一键启动采集任务                  ║"
echo "╚══════════════════════════════════════════╝"

# ── 1. 检查 Python ──
PYTHON=""
if [ -f ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
    echo "  ✅ 虚拟环境: .venv"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    echo "  ✅ 系统 Python: $(python3 --version 2>&1)"
else
    echo "  ❌ 未找到 python3"
    exit 1
fi

# ── 2. 检查必要依赖 ──
echo -n "  🔍 检查依赖... "
DEP_MISSING=0
for mod in requests oss2; do
    if ! "$PYTHON" -c "import $mod" 2>/dev/null; then
        echo ""
        echo "  ⚠️ 缺少 $mod，正在安装..."
        "$PYTHON" -m pip install -q "$mod" || {
            echo "  ❌ 安装 $mod 失败"
            exit 1
        }
    fi
done
echo "ok"

# ── 3. 检查哈希数据库 ──
if [ ! -f "data/image_hashes.db" ]; then
    echo "  ⚠️ 哈希数据库不存在，正在构建..."
    "$PYTHON" scripts/data_collection/build_hash_db.py
fi

# ── 4. 检查飞书配置 ──
FEISHU_ARGS=""
if [ -f "scripts/notification_config.json" ]; then
    echo "  ✅ 飞书配置: scripts/notification_config.json"
else
    echo "  ⚠️ 飞书配置不存在，将禁用消息推送"
    FEISHU_ARGS="--no-feishu"
fi

# ── 5. 显示环境信息 ──
echo ""
echo "  📂 数据集目录: data/final_dataset"
echo "  📄 哈希数据库: data/image_hashes.db ($([ -f data/image_hashes.db ] && du -h data/image_hashes.db | cut -f1 || echo 'N/A'))"
echo "  📝 采集脚本: scripts/data_collection/collect_from_keywords.py"
echo ""

# ── 6. 启动采集 ──
echo "  🚀 启动采集..."
echo "─────────────────────────────────────────"
exec "$PYTHON" scripts/data_collection/collector_runner.py "$@" $FEISHU_ARGS