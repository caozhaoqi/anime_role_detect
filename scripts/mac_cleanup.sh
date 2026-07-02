#!/bin/bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
size()  { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

show_disk() {
    df -h / | awk 'NR==2 {printf "磁盘: 总计 %s, 已用 %s, 可用 %s (%s)\n", $2, $3, $4, $5}'
}

confirm() {
    echo -en "${YELLOW}是否清理? [y/N] ${NC}"
    read -r ans
    [[ "$ans" =~ ^[Yy] ]]
}

clean_if_exists() {
    local path="$1"
    local desc="$2"
    if [[ -d "$path" ]]; then
        local s
        s=$(size "$path")
        echo -e "  ${desc}: ${RED}${s}${NC}"
        if confirm; then
            if sudo find "$path" -mindepth 1 -delete 2>/dev/null; then
                ok "已清理 ${desc} (${s})"
            else
                warn "清理 ${desc} 失败"
            fi
        else
            info "跳过 ${desc}"
        fi
    fi
}

echo ""
echo "========================================="
echo "       macOS 磁盘空间清理工具"
echo "========================================="
echo ""
show_disk
echo ""

# ---- 1. ToDesk 崩溃转储 ----
TODESK_DUMPS="/Library/Application Support/Todesk/dumps"
if [[ -d "$TODESK_DUMPS" ]]; then
    count=$(find "$TODESK_DUMPS" -type f 2>/dev/null | wc -l | tr -d ' ')
    s=$(size "$TODESK_DUMPS")
    echo -e "${RED}[1] ToDesk 崩溃转储${NC} — ${s} (${count} 个文件)"
    echo "    路径: $TODESK_DUMPS"
    if confirm; then
        sudo find "$TODESK_DUMPS" -type f -delete 2>/dev/null
        ok "已清理 ToDesk 崩溃转储 (${s})"
    else
        info "跳过 ToDesk 崩溃转储"
    fi
    echo ""
fi

# ---- 2. iOS / watchOS 模拟器 ----
CORE_SIM="/Library/Developer/CoreSimulator"
if [[ -d "$CORE_SIM" ]]; then
    s=$(size "$CORE_SIM")
    echo -e "${RED}[2] iOS/watchOS 模拟器${NC} — ${s}"
    echo "    路径: $CORE_SIM"
    if confirm; then
        sudo rm -rf "${CORE_SIM}/Volumes/"* "${CORE_SIM}/Caches/"* 2>/dev/null
        ok "已清理模拟器数据 (${s})"
    else
        info "跳过模拟器数据"
    fi
    echo ""
fi

# ---- 3. 系统诊断日志 ----
echo -e "${RED}[3] 系统诊断日志${NC}"
for d in /private/var/db/powerlog /private/var/db/diagnostics; do
    clean_if_exists "$d" "$(basename "$d")"
done
echo ""

# ---- 4. 系统日志文件 ----
SYS_LOGS="/private/var/log"
if [[ -d "$SYS_LOGS" ]]; then
    s=$(size "$SYS_LOGS")
    echo -e "${RED}[4] 系统日志${NC} — ${s}"
    echo "    路径: $SYS_LOGS"
    if confirm; then
        sudo find "$SYS_LOGS" -type f -name "*.gz" -delete 2>/dev/null
        sudo find "$SYS_LOGS" -type f -name "*.old" -delete 2>/dev/null
        sudo find "$SYS_LOGS" -type f -name "*.bz2" -delete 2>/dev/null
        ok "已清理压缩系统日志"
    else
        info "跳过系统日志"
    fi
    echo ""
fi

# ---- 5. Homebrew 缓存 ----
if command -v brew &>/dev/null; then
    BREW_CACHE="$(brew --cache 2>/dev/null || true)"
    if [[ -n "$BREW_CACHE" && -d "$BREW_CACHE" ]]; then
        s=$(size "$BREW_CACHE")
        echo -e "${RED}[5] Homebrew 缓存${NC} — ${s}"
        echo "    路径: $BREW_CACHE"
        if confirm; then
            brew cleanup --prune=all -s 2>/dev/null
            ok "已清理 Homebrew 缓存 (${s})"
        else
            info "跳过 Homebrew 缓存"
        fi
        echo ""
    fi
fi

# ---- 6. pip 缓存 ----
if command -v pip &>/dev/null; then
    PIP_CACHE=$(pip cache dir 2>/dev/null || true)
    if [[ -n "$PIP_CACHE" && -d "$PIP_CACHE" ]]; then
        s=$(size "$PIP_CACHE")
        echo -e "${RED}[6] pip 缓存${NC} — ${s}"
        echo "    路径: $PIP_CACHE"
        if confirm; then
            pip cache purge 2>/dev/null
            ok "已清理 pip 缓存 (${s})"
        else
            info "跳过 pip 缓存"
        fi
        echo ""
    fi
fi

# ---- 7. npm 缓存 ----
NPM_CACHE="$HOME/.npm"
if [[ -d "$NPM_CACHE/_cacache" ]]; then
    s=$(size "$NPM_CACHE")
    echo -e "${RED}[7] npm 缓存${NC} — ${s}"
    echo "    路径: $NPM_CACHE"
    if confirm; then
        npm cache clean --force 2>/dev/null
        ok "已清理 npm 缓存 (${s})"
    else
        info "跳过 npm 缓存"
    fi
    echo ""
fi

# ---- 8. Docker 未使用资源 ----
if command -v docker &>/dev/null && docker info &>/dev/null; then
    echo -e "${RED}[8] Docker 未使用资源${NC}"
    echo "    (悬空镜像、停止的容器、未使用的网络)"
    if confirm; then
        docker system prune -f 2>/dev/null
        ok "已清理 Docker 资源"
    else
        info "跳过 Docker"
    fi
    echo ""
fi

# ---- 9. 用户缓存 ----
USER_CACHE="$HOME/Library/Caches"
if [[ -d "$USER_CACHE" ]]; then
    s=$(size "$USER_CACHE")
    echo -e "${RED}[9] 用户缓存${NC} — ${s}"
    echo "    路径: $USER_CACHE"
    echo -e "  ${YELLOW}注意: 清理后部分应用可能需要重新加载数据${NC}"
    if confirm; then
        find "$USER_CACHE" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        ok "已清理用户缓存 (${s})"
    else
        info "跳过用户缓存"
    fi
    echo ""
fi

echo "========================================="
echo "  清理完成!"
echo ""
show_disk
echo "========================================="
