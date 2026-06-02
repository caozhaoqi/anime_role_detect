#!/bin/bash
# 使用 supervisord 管理所有服务的启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERVISORD_CONF="${SCRIPT_DIR}/supervisord.conf"

start_services() {
    echo "🚀 启动所有服务..."

    # 检查 supervisord 是否已安装
    if ! command -v supervisord &> /dev/null; then
        echo "❌ supervisord 未安装，请运行: pip install supervisord"
        exit 1
    fi

    # 启动 supervisord
    supervisord -c "$SUPERVISORD_CONF"
    echo "✅ supervisord 已启动"

    # 等待服务启动
    sleep 5

    # 显示服务状态
    supervisorctl -c "$SUPERVISORD_CONF" status
}

stop_services() {
    echo "🛑 停止所有服务..."
    supervisorctl -c "$SUPERVISORD_CONF" shutdown
    echo "✅ 所有服务已停止"
}

restart_services() {
    echo "🔄 重启所有服务..."
    supervisorctl -c "$SUPERVISORD_CONF" restart all
}

status_services() {
    echo "📊 服务状态:"
    supervisorctl -c "$SUPERVISORD_CONF" status
}

tail_logs() {
    echo "📝 查看服务日志 (Ctrl+C 退出):"
    tail -f /tmp/model-service.log
}

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
        tail_logs
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
