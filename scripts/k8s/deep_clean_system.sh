#!/bin/bash
# =============================================================================
# Ubuntu + Docker 深度空间释放脚本 (强力版)
# 用法: sudo ./deep_clean_system.sh [--dry-run] [--skip-docker]
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

DRY_RUN=false
SKIP_DOCKER=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-docker) SKIP_DOCKER=true ;;
        *) echo -e "${RED}[ERROR]${NC} 未知参数: $arg"; exit 1 ;;
    esac
done

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 安全删除: dry-run 模式下只打印不执行
safe_rm() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] rm -rf $*"
    else
        rm -rf "$@" 2>/dev/null || true
    fi
}

safe_run() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $*"
    else
        "$@" 2>/dev/null || true
    fi
}

# 计算空间变化
get_disk_usage() {
    df -BM / 2>/dev/null | awk 'NR==2 {print $3}' | tr -d 'M'
}

# 确保以 root/sudo 权限运行
if [ "$EUID" -ne 0 ]; then
    error "请使用 sudo 权限运行此脚本，例如: sudo $0"
    exit 1
fi

# 自动检测项目根目录
if [ -n "${SUDO_USER:-}" ]; then
    REAL_HOME=$(eval echo "~$SUDO_USER")
else
    REAL_HOME="$HOME"
fi
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)" || PROJECT_ROOT=""
PROJECT_FRONTEND="${PROJECT_ROOT}/src/frontend"

if [ "$DRY_RUN" = true ]; then
    echo "========================================"
    echo "  Ubuntu + Docker 深度清理 (DRY-RUN 模式)"
    echo "========================================"
else
    echo "========================================"
    echo "  Ubuntu + Docker 物理磁盘深度清理"
    echo "========================================"
fi
echo "  项目根目录: $PROJECT_ROOT"
echo "  用户目录:   $REAL_HOME"
echo ""

BEFORE=$(get_disk_usage)
info "清理前磁盘使用: ${BEFORE}M"
echo ""

# ======================== 1. 清理 Python pip 缓存 ========================
echo ""
info "--- 1. 清理 Python Pip 依赖缓存 ---"
if command -v pip3 &>/dev/null; then
    safe_run pip3 cache purge
fi
safe_rm "$REAL_HOME/.cache/pip"
safe_rm /root/.cache/pip
safe_rm "$REAL_HOME/.cache/pip-tools"
ok "Pip 缓存清理完成"

# ======================== 2. 清理 Systemd 系统日志 ========================
echo ""
info "--- 2. 清理 Systemd 历史日志 (保留近3天, 最大100M) ---"
if command -v journalctl &>/dev/null; then
    safe_run journalctl --vacuum-time=3d
    safe_run journalctl --vacuum-size=100M
fi
ok "系统日志清理完成"

# ======================== 3. 清理 /var/log 旧日志文件 ========================
echo ""
info "--- 3. 清理 /var/log 旧日志 ---"
if [ "$DRY_RUN" = false ]; then
    find /var/log -type f -name "*.log.*" -mtime +7 -delete 2>/dev/null || true
    find /var/log -type f -name "*.gz" -mtime +7 -delete 2>/dev/null || true
    # 清空大日志文件（保留文件句柄）
    find /var/log -type f -name "*.log" -size +100M -exec truncate -s 0 {} \; 2>/dev/null || true
fi
ok "旧日志清理完成"

# ======================== 4. 清理 APT 包管理器垃圾 ========================
echo ""
info "--- 4. 清理 APT 缓存及残留依赖包 ---"
if [ "$DRY_RUN" = false ]; then
    apt-get clean -y
    apt-get autoclean -y
    apt-get autoremove --purge -y
fi
ok "APT 垃圾清理完成"

# ======================== 5. 清理旧内核 (保留当前运行的) ========================
echo ""
info "--- 5. 清理旧内核 (保留当前版本) ---"
if [ "$DRY_RUN" = false ]; then
    if command -v dpkg &>/dev/null; then
        CURRENT_KERNEL=$(uname -r)
        dpkg -l 'linux-*' 2>/dev/null | awk '/^ii/ {print $2}' | \
            grep -v "$CURRENT_KERNEL" | \
            grep -E 'linux-(image|headers|modules)-[0-9]' | \
            xargs -r apt-get purge -y 2>/dev/null || true
        update-grub 2>/dev/null || true
    fi
fi
ok "旧内核清理完成"

# ======================== 6. 清理 Snap 缓存 ========================
echo ""
info "--- 6. 清理 Snap 缓存 ---"
if command -v snap &>/dev/null; then
    if [ "$DRY_RUN" = false ]; then
        # 清理旧版本 snap 包
        set +e
        snap list --all 2>/dev/null | awk '/disabled/ {print $1, $3}' | \
            while read -r snapname revision; do
                snap remove "$snapname" --revision="$revision" 2>/dev/null || true
            done
        set -e
    fi
    ok "Snap 缓存清理完成"
else
    warn "未安装 Snap，跳过"
fi

# ======================== 7. 彻底清理 Docker & Buildx 缓存 ========================
echo ""
if [ "$SKIP_DOCKER" = true ]; then
    info "--- 7. 跳过 Docker 清理 (--skip-docker) ---"
else
    info "--- 7. 彻底清理 Docker 深度垃圾 (含 Buildx 历史缓存) ---"
    if [ "$DRY_RUN" = false ]; then
        docker system prune -a --volumes -f || true
        docker buildx prune -a -f || true
        docker builder prune -a -f 2>/dev/null || true
    fi
    ok "Docker 深度清理完成"
fi

# ======================== 8. 清理项目开发缓存 ========================
echo ""
info "--- 8. 清理项目开发缓存 ---"
if [ -d "$PROJECT_FRONTEND" ]; then
    safe_rm "$PROJECT_FRONTEND/.next"
    safe_rm "$PROJECT_FRONTEND/node_modules"
    safe_rm "$PROJECT_FRONTEND/.turbo"
    ok "前端缓存清理完成"
else
    warn "未找到前端目录 ($PROJECT_FRONTEND)，跳过"
fi

# 清理项目中的 Python 缓存
if [ -n "$PROJECT_ROOT" ] && [ -d "$PROJECT_ROOT" ]; then
    if [ "$DRY_RUN" = false ]; then
        find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
        find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    fi
    ok "Python 缓存清理完成"
fi

# ======================== 9. 清理用户级缓存 ========================
echo ""
info "--- 9. 清理用户级缓存 ---"
safe_rm "$REAL_HOME/.cache/npm"
safe_rm "$REAL_HOME/.cache/yarn"
safe_rm "$REAL_HOME/.cache/pip"
safe_rm "$REAL_HOME/.npm/_cacache"
safe_rm "$REAL_HOME/.cache/ms-playwright"  # Playwright 浏览器缓存 (~500MB+)
ok "用户缓存清理完成"

# ======================== 10. 清理系统临时目录 ========================
echo ""
info "--- 10. 清理系统临时目录 ---"
safe_rm /tmp/build_*.log
safe_rm /tmp/tmp.*
if [ "$DRY_RUN" = false ]; then
    find /tmp -type f -atime +3 -delete 2>/dev/null || true
    find /tmp -type d -empty -delete 2>/dev/null || true
fi
ok "临时目录清理完成"

# ======================== 11. 清理用户回收站 ========================
echo ""
info "--- 11. 清理用户回收站 ---"
safe_rm "$REAL_HOME/.local/share/Trash"
safe_rm /root/.local/share/Trash
ok "回收站清理完成"

# ======================== 汇总 ========================
echo ""
echo "========================================"
AFTER=$(get_disk_usage)
FREED=$((BEFORE - AFTER))
info "清理前磁盘使用: ${BEFORE}M"
info "清理后磁盘使用: ${AFTER}M"
if [ "$FREED" -gt 0 ]; then
    ok "释放空间: ${FREED}M ($(echo "scale=1; $FREED/1024" | bc 2>/dev/null || echo "?")G)"
elif [ "$FREED" -lt 0 ]; then
    warn "磁盘使用增加: $((-FREED))M (可能有其他进程在写入)"
else
    info "磁盘使用无变化"
fi
echo "========================================"