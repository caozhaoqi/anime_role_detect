#!/bin/bash
# 图片同步定时任务脚本
# 用于定期将所有来源的图片同步到 merged_english_dataset

# 项目根目录
PROJECT_ROOT="/Users/caozhaoqi/PycharmProjects/anime_role_detect"

# 日志文件
LOG_FILE="$PROJECT_ROOT/logs/image_sync.log"

# 创建日志目录
mkdir -p "$PROJECT_ROOT/logs"

# 记录开始时间
echo "==========================================" >> "$LOG_FILE"
echo "图片同步任务开始: $(date)" >> "$LOG_FILE"

# 运行同步脚本
cd "$PROJECT_ROOT"
python3 scripts/analysis/sync_images_to_merged.py >> "$LOG_FILE" 2>&1

# 记录结束时间
echo "图片同步任务结束: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 输出最近的同步结果
echo "最近同步结果:"
tail -20 "$LOG_FILE"
