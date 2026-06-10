#!/bin/bash
set -e

# 定义路径
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SUPERVISOR_CONF="$PROJECT_DIR/supervisord.conf"
LOG_DIR="$PROJECT_DIR/logs"
RUN_DIR="$PROJECT_DIR/run"
PID_FILE="$RUN_DIR/supervisord.pid"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建日志目录
create_log_dir() {
    info "创建日志目录..."
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
    fi
}

# 创建 PID 目录
create_run_dir() {
    info "创建运行目录..."
    if [ ! -d "$RUN_DIR" ]; then
        mkdir -p "$RUN_DIR"
    fi
}

# 释放占用端口（避免由于 set -e 导致无进程时脚本崩溃，加了 || true 保护）
release_ports() {
    # 9001: Supervisor控制台, 8000: 后端服务, 3000: 前端服务
    local ports=(9001 8000 3000)
    info "检查并释放占用端口 [${ports[*]}]..."
    
    for port in "${ports[@]}"; do
        local pids=""
        # 兼容 macOS 和 Linux 的端口检测，并采用 || true 保护防断
        if command -v lsof &> /dev/null; then
            pids=$(lsof -t -i :"$port" 2>/dev/null || true)
        elif command -v fuser &> /dev/null; then
            pids=$(fuser "$port"/tcp 2>/dev/null | awk '{print $NF}' || true)
        fi
        
        if [ -n "$pids" ]; then
            warn "端口 $port 被残留进程 (PIDs: $pids) 占用，正在强制终止释放..."
            for pid in $pids; do
                kill -9 "$pid" 2>/dev/null || true
            done
            sleep 0.5
        fi
    done
}

# 启动服务
start_services() {
    info "启动所有服务..."
    
    # 创建必要目录
    create_log_dir
    create_run_dir
    
    # 检查 supervisord 是否已安装
    if ! command -v supervisord &> /dev/null; then
        error "supervisord 未安装，请先安装: pip install supervisor"
        exit 1
    fi
    
    # 检查配置文件
    if [ ! -f "$SUPERVISOR_CONF" ]; then
        error "配置文件不存在: $SUPERVISOR_CONF"
        exit 1
    fi
    
    # 检查是否已有实例运行
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            warn "检测到 supervisord 已在运行 (PID: $pid)"
            info "重启服务..."
            supervisorctl -c "$SUPERVISOR_CONF" restart all
            return
        fi
    fi
    
    # 在全新拉起 supervisord 实例前，强制清空可能残留占用端口的残留进程
    release_ports
    
    # 启动 supervisord
    info "启动 supervisord..."
    cd "$PROJECT_DIR"
    supervisord -c "$SUPERVISOR_CONF"
    
    info "等待服务启动..."
    sleep 8
    
    # 检查服务状态
    supervisorctl -c "$SUPERVISOR_CONF" status
    
    info "服务启动完成！"
    info "Supervisor 管理界面: http://localhost:9001 (用户名: admin, 密码: admin)"
    info "前端: http://localhost:3000"
}

# 停止服务
stop_services() {
    info "停止所有服务..."
    
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            supervisorctl -c "$SUPERVISOR_CONF" stop all
            supervisorctl -c "$SUPERVISOR_CONF" shutdown
            sleep 2
        else
            warn "supervisord 进程不存在，进行孤儿进程清理..."
        fi
    else
        warn "PID 文件不存在，可能服务未运行"
    fi
    
    # 强力清理阶段：在停止后再次扫描并关闭依然存活的后台孤儿服务
    release_ports
    
    # 清理残留的 PID 缓存文件
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    
    info "服务已停止"
}

# 重启服务
restart_services() {
    info "重启所有服务..."
    stop_services
    sleep 2
    start_services
}

# 查看服务状态
status_services() {
    info "查看服务状态..."
    supervisorctl -c "$SUPERVISOR_CONF" status
}

# 查看日志
view_logs() {
    if [ -z "$1" ]; then
        info "查看所有服务日志..."
        tail -f "$LOG_DIR"/*.log
    else
        info "查看 $1 服务日志..."
        if [ -f "$LOG_DIR/$1.log" ]; then
            tail -f "$LOG_DIR/$1.log"
        elif [ -f "$LOG_DIR/$1.err.log" ]; then
            tail -f "$LOG_DIR/$1.err.log"
        else
            error "日志文件不存在: $1.log"
        fi
    fi
}

# 帮助信息
show_help() {
    echo "使用方法: $0 <command>"
    echo ""
    echo "命令列表:"
    echo "  start     - 启动所有服务（启动前自动清理残留端口）"
    echo "  stop      - 停止所有服务（停止后强制释放端口占用）"
    echo "  restart   - 重启所有服务"
    echo "  status    - 查看服务状态"
    echo "  logs [服务名] - 查看日志（可选服务名: model-service, api-service, frontend 等）"
    echo "  help      - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 logs model-service"
}

# 主函数
main() {
    case "$1" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            status_services
            ;;
        logs)
            view_logs "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"