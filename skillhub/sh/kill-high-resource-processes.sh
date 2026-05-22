#!/bin/bash
# ======================================================================
# 安全自动清理高资源进程脚本 —— 阿里云 ECS 推荐版
# 功能：检测并（可选）终止 CPU >80% 或 RSS内存 >1GB 的异常进程（排除白名单）
# 作者：阿里云客户端运维助手 | 注意：默认 DRY-RUN！务必先测试！
# ======================================================================

# ================ 【配置区】请按需修改 ==================
CPU_THRESHOLD=80          # CPU 使用率阈值（%），设为 0 则禁用 CPU 检测
MEM_THRESHOLD_KB=1048576  # 内存 RSS 阈值（KB），即 1GB；设为 0 则禁用内存检测
WHITELIST="^(sshd|systemd|kthreadd|kswapd|khungtaskd|mysqld|postgres|redis-server|mongod|nginx|apache2|httpd|java|node|python.*uwsgi|dockerd|containerd|kubelet|AliYunDun|cloud-init)$"
LOG_FILE="/var/log/aliyun-kill-high-resource.log"
KILL_ENABLED=false        # ⚠️ 默认 false（仅记录不终止）；设 true 并加 --kill 才执行 kill
MAX_AGE_MINUTES=10        # 同一 PID 10 分钟内不重复处理（防循环）
# =========================================================

# 日志函数
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOG_FILE"; }

# 参数解析
DRY_RUN=true
while [[ $# -gt 0 ]]; do
  case $1 in
    --kill)
      KILL_ENABLED=true
      DRY_RUN=false
      shift
      ;;
    --help|-h)
      echo "用法: $0 [--kill] [--help]"
      echo "  --kill     : 启用真实终止（默认仅打印不操作）"
      echo "  --help/-h  : 显示此帮助"
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
done

# 检查权限
if [[ $EUID -ne 0 ]]; then
  log "ERROR" "请以 root 用户运行：sudo $0 $*"
  exit 1
fi

# 获取当前时间戳（用于去重）
NOW_SEC=$(date +%s)

# 临时文件存储待处理进程
TMP_PROC_FILE=$(mktemp)
trap 'rm -f "$TMP_PROC_FILE"' EXIT

# 构建 ps 查询（兼容 CentOS/Alibaba Cloud Linux/Ubuntu）
ps -eo pid,ppid,pcpu,rss,comm,args --no-headers 2>/dev/null | \
  awk -v cpu_th="$CPU_THRESHOLD" -v mem_th="$MEM_THRESHOLD_KB" -v whitelist="$WHITELIST" '
    BEGIN { OFS="\t" }
    {
      pid=$1; ppid=$2; cpu=$3; rss=$4; comm=$5; args=$6
      # 跳过白名单进程（精确匹配 comm 或 args 中含关键字）
      if (comm ~ whitelist || args ~ /(AliYunDun|cloud-init|\/usr\/lib\/exec\/polkit-agent-helper-1)/) next
      # 检查阈值
      if ((cpu_th > 0 && cpu > cpu_th) || (mem_th > 0 && rss > mem_th)) {
        print pid, ppid, cpu, rss, comm, substr(args,1,100)
      }
    }' | sort -n -k1,1 | uniq > "$TMP_PROC_FILE"

# 加载已处理 PID 缓存（避免短时间重复）
CACHE_FILE="/tmp/aliyun-kill-cache-$(date -d '10 minutes ago' +%s)"
if [[ -f "/tmp/aliyun-kill-cache-last" ]]; then
  CACHE_PID_LIST=$(cat "/tmp/aliyun-kill-cache-last" 2>/dev/null | awk -v now="$NOW_SEC" '$2 > now - '"$MAX_AGE_MINUTES"'*60 {print $1}' | sort -u)
else
  CACHE_PID_LIST=""
fi

# 主处理循环
count=0
while IFS=$'\t' read -r pid ppid cpu rss comm args; do
  [[ -z "$pid" ]] && continue
  # 跳过已处理的 PID（10分钟内）
  if echo "$CACHE_PID_LIST" | grep -qw "^$pid$"; then
    log "SKIP" "PID $pid ($comm) 已在 $MAX_AGE_MINUTES 分钟内处理过，跳过"
    continue
  fi

  # 获取用户 & 启动时间（增强可追溯性）
  user=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
  start_time=$(ps -o lstart= -p "$pid" 2>/dev/null | xargs)

  action="WOULD KILL"
  if [[ "$DRY_RUN" == "false" ]]; then
    action="KILLING"
    if kill -0 "$pid" 2>/dev/null; then
      kill -15 "$pid" 2>/dev/null
      sleep 0.5
      if kill -0 "$pid" 2>/dev/null; then
        log "WARN" "PID $pid still alive → sending SIGKILL"
        kill -9 "$pid" 2>/dev/null
      fi
    else
      log "WARN" "PID $pid no longer exists (race condition)"
      continue
    fi
  fi

  log "ALERT" "$action PID $pid | USER:$user | CPU:${cpu}% | MEM:${rss}KB | CMD:$comm | ARGS:$args | START:$start_time"
  count=$((count + 1))

  # 记录到缓存
  echo "$pid $NOW_SEC" >> "/tmp/aliyun-kill-cache-last"

done < "$TMP_PROC_FILE"

# 清理过期缓存
if [[ -f "/tmp/aliyun-kill-cache-last" ]]; then
  awk -v now="$NOW_SEC" -v limit="$MAX_AGE_MINUTES" '$2 > now - limit*60' "/tmp/aliyun-kill-cache-last" > "/tmp/aliyun-kill-cache-last.tmp" && mv "/tmp/aliyun-kill-cache-last.tmp" "/tmp/aliyun-kill-cache-last"
fi

if [[ $count -eq 0 ]]; then
  log "INFO" "未发现超阈值进程（CPU>$CPU_THRESHOLD% 或 RSS>$MEM_THRESHOLD_KB KB）"
else
  log "INFO" "共处理 $count 个异常进程"
fi


# sudo ./sh/kill-high-resource.sh --kill