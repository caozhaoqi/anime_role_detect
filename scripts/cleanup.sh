#!/bin/bash
# Linux 一键清理脚本 — 安全清除无用缓存和文件
#
# 用法:
#   bash cleanup.sh          # 交互模式，逐项确认
#   bash cleanup.sh --all    # 一键全清（跳过确认）
#   bash cleanup.sh --dry    # 只扫描不删除，显示可回收空间

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODE="interactive"
for arg in "$@"; do
    case $arg in
        --all)  MODE="auto" ;;
        --dry)  MODE="dry" ;;
        -h|--help)
            echo "用法: bash $0 [--all|--dry]"
            echo "  --all   一键全清（跳过确认）"
            echo "  --dry   只扫描不删除"
            exit 0
            ;;
    esac
done

TOTAL_SAVED=0

# 获取目录大小 (KB)
dir_size() {
    if [ -d "$1" ] || [ -f "$1" ]; then
        du -sk "$1" 2>/dev/null | awk '{print $1}'
    else
        echo 0
    fi
}

# 格式化大小
fmt_size() {
    local kb=$1
    if [ "$kb" -ge 1048576 ]; then
        echo "$(echo "scale=1; $kb/1048576" | bc) GB"
    elif [ "$kb" -ge 1024 ]; then
        echo "$(echo "scale=1; $kb/1024" | bc) MB"
    else
        echo "${kb} KB"
    fi
}

# 确认是否执行
confirm() {
    if [ "$MODE" = "auto" ]; then
        return 0
    elif [ "$MODE" = "dry" ]; then
        return 1
    fi
    echo -n -e "${YELLOW}  执行清理? [y/N] ${NC}"
    read -r ans
    [[ "$ans" =~ ^[yY]$ ]]
}

# 执行清理并统计
do_clean() {
    local name="$1"
    local cmd="$2"
    local size_kb="$3"

    if [ "$size_kb" -eq 0 ] 2>/dev/null; then
        echo -e "  ${GREEN}⏭  跳过${NC} (已经是 0)"
        return
    fi

    echo -e "  ${CYAN}📦 可回收: $(fmt_size $size_kb)${NC}"

    if [ "$MODE" = "dry" ]; then
        echo -e "  ${YELLOW}[DRY RUN] 将执行: ${cmd}${NC}"
        TOTAL_SAVED=$((TOTAL_SAVED + size_kb))
        return
    fi

    if confirm; then
        eval "$cmd" 2>/dev/null || true
        echo -e "  ${GREEN}✅ 已清理 $(fmt_size $size_kb)${NC}"
        TOTAL_SAVED=$((TOTAL_SAVED + size_kb))
    else
        echo -e "  ${YELLOW}⏭  已跳过${NC}"
    fi
}

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Linux 系统清理脚本 v1.0              ║${NC}"
echo -e "${CYAN}║     模式: $(printf '%-28s' "$MODE")        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 磁盘概览 ──
echo -e "${CYAN}📊 磁盘使用概览:${NC}"
df -h / | awk 'NR==2{printf "  总计: %s  已用: %s (%s)  可用: %s\n", $2, $3, $5, $4}'
echo ""

# ── 1. APT 缓存 ──
if command -v apt-get &>/dev/null; then
    size=$(dir_size /var/cache/apt)
    echo -e "${YELLOW}[1/10] APT 包缓存${NC} (/var/cache/apt)"
    do_clean "apt" "sudo apt-get clean && sudo apt-get autoclean -y" "$size"
    echo ""
fi

# ── 2. Pip 缓存 ──
PIP_CACHE=""
if command -v pip &>/dev/null; then
    PIP_CACHE=$(pip cache dir 2>/dev/null || echo "$HOME/.cache/pip")
elif command -v pip3 &>/dev/null; then
    PIP_CACHE=$(pip3 cache dir 2>/dev/null || echo "$HOME/.cache/pip")
fi
if [ -n "$PIP_CACHE" ] && [ -d "$PIP_CACHE" ]; then
    size=$(dir_size "$PIP_CACHE")
    echo -e "${YELLOW}[2/10] Pip 缓存${NC} ($PIP_CACHE)"
    do_clean "pip" "pip cache purge 2>/dev/null; rm -rf ${PIP_CACHE}/*" "$size"
    echo ""
else
    echo -e "${YELLOW}[2/10] Pip 缓存${NC} — 未找到"
    echo ""
fi

# ── 3. npm 缓存 ──
if command -v npm &>/dev/null; then
    NPM_CACHE=$(npm config get cache 2>/dev/null || echo "$HOME/.npm")
    size=$(dir_size "$NPM_CACHE")
    echo -e "${YELLOW}[3/10] npm 缓存${NC} ($NPM_CACHE)"
    do_clean "npm" "npm cache clean --force" "$size"
    echo ""
else
    echo -e "${YELLOW}[3/10] npm 缓存${NC} — 未安装"
    echo ""
fi

# ── 4. Docker 无用资源 ──
if command -v docker &>/dev/null; then
    echo -e "${YELLOW}[4/10] Docker 无用资源${NC}"
    # 统计
    dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
    stopped_containers=$(docker ps -a -f "status=exited" -q 2>/dev/null | wc -l)
    unused_volumes=$(docker volume ls -f "dangling=true" -q 2>/dev/null | wc -l)
    unused_networks=$(docker network ls -f "type=custom" -q 2>/dev/null | wc -l)
    docker_size=$(docker system df 2>/dev/null | tail -1 | awk '{print $NF}' || echo "未知")

    echo -e "  悬空镜像: ${dangling_images}  停止容器: ${stopped_containers}  孤立卷: ${unused_volumes}"
    echo -e "  可回收空间: ${CYAN}${docker_size}${NC}"

    if [ "$dangling_images" -gt 0 ] || [ "$stopped_containers" -gt 0 ] || [ "$unused_volumes" -gt 0 ]; then
        if [ "$MODE" = "dry" ]; then
            echo -e "  ${YELLOW}[DRY RUN] docker system prune -f --volumes${NC}"
        elif confirm; then
            docker system prune -f --volumes 2>/dev/null || true
            echo -e "  ${GREEN}✅ Docker 清理完成${NC}"
        else
            echo -e "  ${YELLOW}⏭  已跳过${NC}"
        fi
    else
        echo -e "  ${GREEN}⏭  无需清理${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}[4/10] Docker${NC} — 未安装"
    echo ""
fi

# ── 5. Journal 日志 ──
if command -v journalctl &>/dev/null; then
    journal_size=$(sudo du -sm /var/log/journal 2>/dev/null | awk '{print $1}' || echo 0)
    journal_size_kb=$((journal_size * 1024))
    echo -e "${YELLOW}[5/10] Systemd Journal 日志${NC} (/var/log/journal)"
    do_clean "journal" "sudo journalctl --vacuum-time=3d --vacuum-size=100M" "$journal_size_kb"
    echo ""
else
    echo -e "${YELLOW}[5/10] Journal 日志${NC} — 不适用"
    echo ""
fi

# ── 6. 旧日志文件 ──
echo -e "${YELLOW}[6/10] 旧日志文件${NC} (/var/log 中的 .gz .old .1)"
old_logs_size=0
while IFS= read -r f; do
    s=$(dir_size "$f")
    old_logs_size=$((old_logs_size + s))
done < <(find /var/log -type f \( -name "*.gz" -o -name "*.old" -o -name "*.1" \) 2>/dev/null)
do_clean "old-logs" "sudo find /var/log -type f \( -name '*.gz' -o -name '*.old' -o -name '*.1' \) -delete" "$old_logs_size"
echo ""

# ── 7. 用户缓存 ──
USER_CACHE="$HOME/.cache"
if [ -d "$USER_CACHE" ]; then
    size=$(dir_size "$USER_CACHE")
    echo -e "${YELLOW}[7/10] 用户缓存${NC} ($USER_CACHE)"
    echo -e "  包含: 缩略图、字体缓存、应用缓存等"
    do_clean "user-cache" "rm -rf ${USER_CACHE}/*" "$size"
    echo ""
else
    echo -e "${YELLOW}[7/10] 用户缓存${NC} — 不存在"
    echo ""
fi

# ── 8. /tmp 旧文件 ──
tmp_size=$(sudo du -sk /tmp 2>/dev/null | awk '{print $1}' || echo 0)
echo -e "${YELLOW}[8/10] /tmp 临时文件${NC}"
do_clean "tmp" "sudo find /tmp -type f -atime +7 -delete 2>/dev/null; sudo find /tmp -type d -empty -delete 2>/dev/null" "$tmp_size"
echo ""

# ── 9. Trash 回收站 ──
TRASH="$HOME/.local/share/Trash"
if [ -d "$TRASH" ]; then
    size=$(dir_size "$TRASH")
    echo -e "${YELLOW}[9/10] 回收站${NC} ($TRASH)"
    do_clean "trash" "rm -rf ${TRASH}/files/* ${TRASH}/info/*" "$size"
    echo ""
else
    echo -e "${YELLOW}[9/10] 回收站${NC} — 为空"
    echo ""
fi

# ── 10. conda / uv 缓存 ──
echo -e "${YELLOW}[10/10] 其他包管理器缓存${NC}"
extra_cleaned=false

# conda
if command -v conda &>/dev/null; then
    conda_cache=$(conda info 2>/dev/null | grep "package cache" | awk -F: '{print $2}' | xargs || echo "$HOME/.conda/pkgs")
    for d in $conda_cache; do
        if [ -d "$d" ]; then
            size=$(dir_size "$d")
            echo -e "  conda 缓存 ($d):"
            do_clean "conda" "conda clean --all -y" "$size"
            extra_cleaned=true
        fi
    done
fi

# uv
if [ -d "$HOME/.cache/uv" ]; then
    size=$(dir_size "$HOME/.cache/uv")
    echo -e "  uv 缓存:"
    do_clean "uv" "rm -rf $HOME/.cache/uv/*" "$size"
    extra_cleaned=true
fi

# huggingface hub cache
if [ -d "$HOME/.cache/huggingface" ]; then
    size=$(dir_size "$HOME/.cache/huggingface")
    echo -e "  HuggingFace 缓存:"
    do_clean "hf" "rm -rf $HOME/.cache/huggingface/*" "$size"
    extra_cleaned=true
fi

if [ "$extra_cleaned" = false ]; then
    echo -e "  ${GREEN}⏭  无额外缓存${NC}"
fi
echo ""

# ── 汇总 ──
echo -e "${CYAN}════════════════════════════════════════════${NC}"
if [ "$MODE" = "dry" ]; then
    echo -e "  📊 扫描完成，预计可回收: ${GREEN}$(fmt_size $TOTAL_SAVED)${NC}"
    echo -e "  ${YELLOW}运行 bash $0 --all 执行清理${NC}"
else
    echo -e "  📊 本次共清理: ${GREEN}$(fmt_size $TOTAL_SAVED)${NC}"
fi
echo -e "${CYAN}════════════════════════════════════════════${NC}"

# 清理后磁盘概览
echo ""
echo -e "${CYAN}📊 清理后磁盘状态:${NC}"
df -h / | awk 'NR==2{printf "  总计: %s  已用: %s (%s)  可用: %s\n", $2, $3, $5, $4}'
echo ""
