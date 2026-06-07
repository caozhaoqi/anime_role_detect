#!/bin/bash
# 多站点角色图片采集脚本 - 带去重功能
# 使用方法: ./scripts/spider/run_spider_with_dedup.sh [start_from]

# 配置
INPUT_FILE="/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/roles.json"
OUTPUT_DIR="/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_images"
MD5_INDEX="/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_images/.md5_index.json"
MAX_COUNT=30
WORKERS=8
DELAY=2.0
START_FROM=${1:-0}

# 站点列表（按优先级）
SITES="lolibooru yande.re konachan"

echo "=========================================="
echo "多站点角色图片采集器"
echo "=========================================="
echo "输入文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "MD5索引: $MD5_INDEX"
echo "每角色数量: $MAX_COUNT"
echo "并发线程: $WORKERS"
echo "请求延迟: ${DELAY}秒"
echo "起始位置: $START_FROM"
echo "站点列表: $SITES"
echo "=========================================="

# 检查MD5索引是否存在
if [ ! -f "$MD5_INDEX" ]; then
    echo "警告: MD5索引文件不存在，将创建新索引"
    echo "提示: 运行 python3 scripts/spider/md5_index_tool.py --image-dir $OUTPUT_DIR 生成索引"
fi

# 运行采集器
python3 scripts/spider/multi_site_spider.py \
    --site $SITES \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --max-count $MAX_COUNT \
    --workers $WORKERS \
    --delay $DELAY \
    --start-from $START_FROM \
    --md5-index "$MD5_INDEX"

echo "=========================================="
echo "采集完成！"
echo "=========================================="

# 更新MD5索引
echo "更新MD5索引..."
python3 scripts/spider/md5_index_tool.py --image-dir "$OUTPUT_DIR"

echo "完成！"
