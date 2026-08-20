#!/bin/bash
set -e

# 定义路径
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SUPERVISOR_CONF="$PROJECT_DIR/supervisord.conf"
LOG_DIR="$PROJECT_DIR/logs"
RUN_DIR="$PROJECT_DIR/run"
PID_FILE="$RUN_DIR/supervisord.pid"

# supervisor 鉴权凭据（与 supervisord.conf [inet_http_server] 一致；可通过环境变量覆盖：
#   SUP_USER=xxx SUP_PASS=xxx ./run_with_supervisor.sh start
# 修改 conf 密码后请用环境变量传入，勿硬编码明文。修复 2026-08-20）
SUP_USER="${SUP_USER:-CHANGE_ME_supervisor_admin}"
SUP_PASS="${SUP_PASS:-CHANGE_ME_supervisor_pwd}"

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

# 解析 supervisord / supervisorctl 二进制：优先项目 .venv，fallback 系统 PATH
resolve_supervisor_bin() {
    if [ -x "$PROJECT_DIR/.venv/bin/supervisord" ]; then
        echo "$PROJECT_DIR/.venv/bin/supervisord"
    elif command -v supervisord &>/dev/null; then
        command -v supervisord
    else
        return 1
    fi
}

SUP_BIN="$(resolve_supervisor_bin 2>/dev/null || true)"
SUPCTL_BIN="${SUP_BIN%/supervisord}/supervisorctl"

# supervisorctl 统一入口：自动带鉴权参数（修复 2026-08-20：原脚本无 -u/-p 会 401）
supctl() {
    if [ -x "$SUPCTL_BIN" ]; then
        "$SUPCTL_BIN" -c "$SUPERVISOR_CONF" -u "$SUP_USER" -p "$SUP_PASS" "$@"
    elif command -v supervisorctl &>/dev/null; then
        supervisorctl -c "$SUPERVISOR_CONF" -u "$SUP_USER" -p "$SUP_PASS" "$@"
    else
        error "supervisorctl 不可用"
        return 1
    fi
}

# 创建日志目录
create_log_dir() {
    info "创建日志目录..."
    # 从 supervisord.conf 自动推导所有 service 日志目录，避免新增 program 时遗漏建目录
    local conf="$SUPERVISOR_CONF"
    if [ -f "$conf" ]; then
        grep -oE 'logs/services/[^/]+' "$conf" | sort -u | while IFS= read -r rel; do
            [ -n "$rel" ] && mkdir -p "$PROJECT_DIR/$rel"
        done
    fi
    # 兜底：硬编码列表（conf 解析异常时仍保证核心目录存在）
    local services=(
        model-service api-service api-gateway multimedia-service
        search-service inference-worker monitoring frontend log-viewer
        t2i-service
    )
    for svc in "${services[@]}"; do
        mkdir -p "$LOG_DIR/services/$svc"
    done
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
    # 9001: Supervisor控制台, 8000: model-service, 8080: api-gateway,
    # 8100: t2i-service, 3000: 前端
    local ports=(9001 8000 8080 8100 3000)
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

    # 前置检查：supervisord 二进制 + 配置
    if [ -z "$SUP_BIN" ]; then
        error "supervisord 未安装（.venv 或系统 PATH 均无），请先安装: .venv/bin/pip install supervisor"
        exit 1
    fi

    # 创建必要目录
    create_log_dir
    create_run_dir

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
            supctl restart all
            return
        fi
    fi

    # 在全新拉起 supervisord 实例前，强制清空可能残留占用端口的残留进程
    release_ports

    # 启动 supervisord（后台）
    # 修复 2026-08-20：supervisord.conf 配置了 nodaemon=true（Docker 场景需要前台），
    # 直接调用会阻塞脚本；这里用 nohup 显式后台，脚本继续执行后续状态检查。
    info "启动 supervisord（后台）: $SUP_BIN"
    cd "$PROJECT_DIR"
    nohup "$SUP_BIN" -c "$SUPERVISOR_CONF" >"$LOG_DIR/supervisord-nohup.out" 2>&1 &

    # 等待 PID 文件出现（nodaemon=true 下 supervisord 仍会写 pidfile）
    for _i in $(seq 1 15); do
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    info "等待服务启动..."
    sleep 8

    # 检查服务状态
    supctl status || true

    info "服务启动完成！"
    info "Supervisor 管理界面: http://localhost:9001 (用户名: ${SUP_USER})"
    info "前端: http://localhost:3000"
}

# 停止服务
stop_services() {
    info "停止所有服务..."

    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            supctl stop all || true
            supctl shutdown || true
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
    supctl status || true
}

# 查看日志
view_logs() {
    if [ -z "$1" ]; then
        info "查看所有服务日志..."
        tail -f "$LOG_DIR"/services/*/*.log 2>/dev/null || {
            error "未找到任何服务日志（$LOG_DIR/services/ 下为空）"
            return 1
        }
    else
        info "查看 $1 服务日志..."
        local base="$LOG_DIR/services/$1"
        if [ -f "$base/$1.log" ]; then
            tail -f "$base/$1.log"
        elif [ -f "$base/$1.err.log" ]; then
            tail -f "$base/$1.err.log"
        else
            error "日志文件不存在: $base/ 下无 $1.log / $1.err.log"
            return 1
        fi
    fi
}

# 帮助信息
show_help() {
    echo "使用方法: $0 <command>"
    echo ""
    echo "命令列表:"
    echo "  start     - 启动所有服务（后台运行；启动前自动清理残留端口）"
    echo "  stop      - 停止所有服务（停止后强制释放端口占用）"
    echo "  restart   - 重启所有服务"
    echo "  status    - 查看服务状态"
    echo "  logs [服务名] - 查看日志（可选服务名: model-service, api-service, t2i-service 等）"
    echo "  help      - 显示帮助信息"
    echo ""
    echo "环境变量:"
    echo "  SUP_USER / SUP_PASS - 覆盖 supervisor 鉴权凭据（修改 conf 密码后使用）"
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
