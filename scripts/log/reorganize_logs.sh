#!/bin/bash
# 日志系统重构脚本 - macOS兼容版本
# 将现有日志重新组织到新的目录结构中

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

echo "=========================================="
echo "日志系统重构工具"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${RED}错误: 找不到logs目录${NC}"
    exit 1
fi

echo -e "${YELLOW}当前日志目录: $LOG_DIR${NC}"
echo ""

# 询问是否继续
read -p "这将重新组织日志文件结构，是否继续? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}操作已取消${NC}"
    exit 0
fi

# 创建备份
BACKUP_DIR="$LOG_DIR/backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}创建备份到: $BACKUP_DIR${NC}"
mkdir -p "$BACKUP_DIR"
cp -r "$LOG_DIR"/* "$BACKUP_DIR/" 2>/dev/null || true
echo -e "${GREEN}✓ 备份完成${NC}"
echo ""

# 创建新的目录结构
echo -e "${YELLOW}创建新的目录结构...${NC}"

# 按服务分类的目录
mkdir -p "$LOG_DIR/services/api-service"
mkdir -p "$LOG_DIR/services/model-service"
mkdir -p "$LOG_DIR/services/api-gateway"
mkdir -p "$LOG_DIR/services/multimedia-service"
mkdir -p "$LOG_DIR/services/search-service"
mkdir -p "$LOG_DIR/services/inference-worker"
mkdir -p "$LOG_DIR/services/frontend"
mkdir -p "$LOG_DIR/services/monitoring"

# 功能分类目录
mkdir -p "$LOG_DIR/functional/health_check"
mkdir -p "$LOG_DIR/functional/inference"
mkdir -p "$LOG_DIR/functional/training"
mkdir -p "$LOG_DIR/functional/system"
mkdir -p "$LOG_DIR/functional/download"
mkdir -p "$LOG_DIR/functional/error"
mkdir -p "$LOG_DIR/archive/compressed"

echo -e "${GREEN}✓ 目录结构创建完成${NC}"
echo ""

# 移动服务日志
echo -e "${YELLOW}移动服务日志...${NC}"

move_log() {
    local log_file=$1
    local target_dir=$2
    if [ -f "$LOG_DIR/$log_file" ]; then
        mv "$LOG_DIR/$log_file" "$LOG_DIR/$target_dir/"
        echo -e "  ${GREEN}✓${NC} $log_file -> $target_dir/"
    fi
}

# API Service
move_log "api-service.log" "services/api-service"
move_log "api-service.err.log" "services/api-service"

# Model Service
move_log "model-service.log" "services/model-service"
move_log "model-service.err.log" "services/model-service"

# API Gateway
move_log "api-gateway.log" "services/api-gateway"
move_log "api-gateway.err.log" "services/api-gateway"

# Multimedia Service
move_log "multimedia-service.log" "services/multimedia-service"
move_log "multimedia-service.err.log" "services/multimedia-service"

# Search Service
move_log "search-service.log" "services/search-service"
move_log "search-service.err.log" "services/search-service"
move_log "search-worker.log" "services/search-service"
move_log "search-worker.err.log" "services/search-service"

# Inference Worker
move_log "inference-worker.log" "services/inference-worker"
move_log "inference-worker.err.log" "services/inference-worker"

# Frontend
move_log "frontend.log" "services/frontend"
move_log "frontend.err.log" "services/frontend"

# Monitoring
move_log "monitor-dashboard.log" "services/monitoring"
move_log "monitor-dashboard.err.log" "services/monitoring"
move_log "health-check.log" "services/monitoring"
move_log "health-check.err.log" "services/monitoring"
move_log "log-monitor.log" "services/monitoring"
move_log "log-monitor.err.log" "services/monitoring"
move_log "resource-monitor.log" "services/monitoring"
move_log "resource-monitor.err.log" "services/monitoring"

echo ""

# 移动功能分类日志
echo -e "${YELLOW}移动功能分类日志...${NC}"

# Health check reports
if [ -d "$LOG_DIR/health_check" ] && [ "$(ls -A $LOG_DIR/health_check 2>/dev/null)" ]; then
    mv "$LOG_DIR/health_check"/* "$LOG_DIR/functional/health_check/" 2>/dev/null || true
    rmdir "$LOG_DIR/health_check" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} health_check/ -> functional/health_check/"
fi

# Inference logs
if [ -d "$LOG_DIR/inference" ] && [ "$(ls -A $LOG_DIR/inference 2>/dev/null)" ]; then
    mv "$LOG_DIR/inference"/* "$LOG_DIR/functional/inference/" 2>/dev/null || true
    rmdir "$LOG_DIR/inference" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} inference/ -> functional/inference/"
fi

# Training logs
if [ -d "$LOG_DIR/training" ] && [ "$(ls -A $LOG_DIR/training 2>/dev/null)" ]; then
    mv "$LOG_DIR/training"/* "$LOG_DIR/functional/training/" 2>/dev/null || true
    rmdir "$LOG_DIR/training" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} training/ -> functional/training/"
fi

# System logs
if [ -d "$LOG_DIR/system" ] && [ "$(ls -A $LOG_DIR/system 2>/dev/null)" ]; then
    mv "$LOG_DIR/system"/* "$LOG_DIR/functional/system/" 2>/dev/null || true
    rmdir "$LOG_DIR/system" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} system/ -> functional/system/"
fi

# Download logs
if [ -d "$LOG_DIR/download" ] && [ "$(ls -A $LOG_DIR/download 2>/dev/null)" ]; then
    mv "$LOG_DIR/download"/* "$LOG_DIR/functional/download/" 2>/dev/null || true
    rmdir "$LOG_DIR/download" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} download/ -> functional/download/"
fi

# Error logs
if [ -d "$LOG_DIR/error" ] && [ "$(ls -A $LOG_DIR/error 2>/dev/null)" ]; then
    mv "$LOG_DIR/error"/* "$LOG_DIR/functional/error/" 2>/dev/null || true
    rmdir "$LOG_DIR/error" 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} error/ -> functional/error/"
fi

# 保留其他文件
echo ""
echo -e "${YELLOW}保留在根目录的文件:${NC}"
for file in supervisord.log unified.log redis.log github_action.log; do
    if [ -f "$LOG_DIR/$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    fi
done

echo ""
echo -e "${GREEN}=========================================="
echo "日志系统重构完成!"
echo "==========================================${NC}"
echo ""
echo -e "${YELLOW}新目录结构:${NC}"
echo "logs/"
echo "├── services/              # 按服务分类"
echo "│   ├── api-service/"
echo "│   ├── model-service/"
echo "│   ├── api-gateway/"
echo "│   ├── multimedia-service/"
echo "│   ├── search-service/"
echo "│   ├── inference-worker/"
echo "│   ├── frontend/"
echo "│   └── monitoring/"
echo "├── functional/            # 按功能分类"
echo "│   ├── health_check/"
echo "│   ├── inference/"
echo "│   ├── training/"
echo "│   ├── system/"
echo "│   ├── download/"
echo "│   └── error/"
echo "├── archive/               # 归档目录"
echo "│   └── compressed/"
echo "├── backup_*/              # 备份目录(可删除)"
echo "├── supervisord.log"
echo "├── unified.log"
echo "└── redis.log"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo "1. 重启supervisor使新配置生效:"
echo "   .venv/bin/supervisorctl -u admin -p admin123 -c supervisord.conf reload"
echo ""
echo "2. 运行日志归档脚本定期清理旧日志:"
echo "   python scripts/logs/archive_logs.py --help"
echo ""
echo "3. 如需删除备份,请确认新结构正常后执行:"
echo "   rm -rf $BACKUP_DIR"
echo ""
