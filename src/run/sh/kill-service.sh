#!/bin/bash

# 定义需要释放的端口（9001: 监控, 8000: 模型服务, 3000: 前端）
PORTS=(9001 8000 3000)

# 定义子服务进程特征，用于清理可能残留的孤儿进程
PROCESS_PATTERNS=(
    "model_service/app.py"
    "api/app.py"
    "api_gateway/app.py"
    "multimedia_service_app.py"
    "app_queue.py"
    "monitor_dashboard.py"
    "health_check.py"
    "log_monitor.py"
    "resource_monitor.py"
)

echo "=== 开始关闭 Supervisord 服务及释放端口 ==="

# 1. 优先尝试使用 supervisorctl 优雅关闭所有服务并退出
if command -v supervisorctl &> /dev/null; then
    echo "[INFO] 正在尝试通过 supervisorctl 优雅关闭服务..."
    # 使用配置文件中的默认用户名 admin 和密码 admin123
    supervisorctl -s http://127.0.0.1:9001 -u admin -p admin123 shutdown 2>/dev/null
    sleep 3
fi

# 2. 检查并强制关闭 supervisord 主进程
SUPERVISOR_PIDS=$(pgrep -f "supervisord")
if [ -not -z "$SUPERVISOR_PIDS" ] || pgrep -x "supervisord" &>/dev/null; then
    echo "[INFO] 发现 supervisord 残留进程，正在强制终止..."
    # 尝试优雅终止
    pkill -15 -f "supervisord" 2>/dev/null
    sleep 1
    # 强制终止
    pkill -9 -f "supervisord" 2>/dev/null
fi

# 3. 通过端口号强杀残留进程，确保端口释放
echo "[INFO] 正在检查并释放目标端口..."
for PORT in "${PORTS[@]}"; do
    # 使用 lsof 查找占用端口的 PID
    PIDS=$(lsof -t -i :$PORT)
    if [ ! -z "$PIDS" ]; then
        echo "[WARN] 端口 $PORT 仍被 PID: $PIDS 占用，正在强制释放..."
        kill -9 $PIDS 2>/dev/null
    else
        echo "[INFO] 端口 $PORT 已成功释放。"
    fi
done

# 4. 根据命令特征清理可能脱离控制的 Python 孤儿进程
echo "[INFO] 正在清理残留的子服务孤儿进程..."
for PATTERN in "${PROCESS_PATTERNS[@]}"; do
    ORPHAN_PIDS=$(pgrep -f "$PATTERN")
    if [ ! -z "$ORPHAN_PIDS" ]; then
        echo "[WARN] 发现残留服务进程: $PATTERN (PID: $ORPHAN_PIDS)，正在清理..."
        kill -9 $ORPHAN_PIDS 2>/dev/null
    fi
done

# 5. 清理前端 npm / node 残留服务（如果 3000 端口未被释放）
NODE_PIDS=$(pgrep -f "next-server|node.*frontend")
if [ ! -z "$NODE_PIDS" ]; then
    echo "[WARN] 发现残留的前端 Node/NPM 进程，正在清理..."
    kill -9 $NODE_PIDS 2>/dev/null
fi

echo "=== 清理完成 ==="
