#!/bin/bash

set -e  # 遇错退出

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查是否为 root
if [[ $EUID -ne 0 ]]; then
  error "此脚本必须以 root 权限运行（使用 sudo 或切换为 root）"
  exit 1
fi

# 检测发行版
detect_os() {
  if command -v lsb_release >/dev/null 2>&1; then
    DISTRO=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
    VERSION=$(lsb_release -rs | cut -d. -f1)
  elif [ -f /etc/os-release ]; then
    DISTRO=$(grep -i "^id=" /etc/os-release | cut -d= -f2 | tr -d '"')
    VERSION=$(grep -i "^version_id=" /etc/os-release | cut -d= -f2 | tr -d '"')
  else
    error "无法识别操作系统类型"
    exit 1
  fi
  echo "$DISTRO $VERSION"
}

OS_INFO=$(detect_os)
DISTRO=$(echo "$OS_INFO" | awk '{print $1}')
VERSION=$(echo "$OS_INFO" | awk '{print $2}')

log "检测到系统：$DISTRO $VERSION"

# 确认执行
read -rp "⚠️  此操作将彻底卸载 Docker 并删除所有本地数据（容器/镜像/卷/网络）。确定继续？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[yY][eE]?[sS]?$ ]]; then
  log "已取消卸载。退出。"
  exit 0
fi

# === 步骤 1：停止并禁用服务 ===
log "正在停止并禁用 Docker 服务..."
if systemctl is-active --quiet docker; then
  systemctl stop docker
fi
if systemctl is-enabled --quiet docker; then
  systemctl disable docker
fi
if systemctl is-active --quiet containerd; then
  systemctl stop containerd
fi
if systemctl is-enabled --quiet containerd; then
  systemctl disable containerd
fi

# === 步骤 2：卸载 Docker 包 ===
log "正在卸载 Docker 相关软件包..."

case "$DISTRO" in
  "ubuntu"|"debian")
    apt-get update -qq > /dev/null 2>&1
    apt-get remove -y docker docker-engine docker.io containerd runc docker-ce docker-ce-cli docker-ce-rootless-extras
    apt-get autoremove -y --purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    ;;
  "centos"|"rhel"|"almalinux"|"rocky"|"ol")
    if [ "$VERSION" = "8" ] || [ "$VERSION" = "9" ]; then
      dnf remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
      dnf autoremove -y
      dnf clean all
    else
      yum remove -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
      yum autoremove -y
      yum clean all
    fi
    ;;
  *)
    error "不支持的操作系统：$DISTRO"
    exit 1
    ;;
esac

# === 步骤 3：清理残留文件和目录 ===
log "正在清理 Docker 数据与配置..."
rm -rf \
  /var/lib/docker \
  /var/lib/containerd \
  /etc/docker \
  /etc/containerd \
  /usr/local/bin/docker* \
  /usr/local/lib/docker \
  ~/.docker \
  /usr/share/bash-completion/completions/docker* \
  /usr/share/zsh/site-functions/_docker

# === 步骤 4：清理用户组（可选，谨慎）===
read -rp "❓ 是否同时删除 'docker' 用户组？（仅当确认无其他服务依赖该组时建议启用）(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[yY][eE]?[sS]?$ ]]; then
  if getent group docker >/dev/null; then
    groupdel docker
    log "'docker' 用户组已删除"
  else
    warn "'docker' 组不存在，跳过"
  fi
else
  warn "跳过删除 'docker' 用户组"
fi

# === 步骤 5：清理 PATH 中可能残留的符号链接 ===
if [ -L "/usr/local/bin/docker" ]; then
  rm -f /usr/local/bin/docker
fi
if [ -L "/usr/local/bin/dockerd" ]; then
  rm -f /usr/local/bin/dockerd
fi

# === 最终验证 ===
log "✅ 卸载完成！正在验证残留..."
if command -v docker >/dev/null 2>&1 || command -v dockerd >/dev/null 2>&1; then
  warn "警告：仍检测到 docker 命令，请手动检查 PATH 或残留二进制文件"
else
  log "✔ docker 命令已不可用"
fi

if [ -d "/var/lib/docker" ]; then
  warn "/var/lib/docker 目录仍存在（已跳过自动删除，因含用户数据风险）"
  warn "如需彻底清理，请手动确认后执行：rm -rf /var/lib/docker"
else
  log "✔ /var/lib/docker 已被移除"
fi

log ""
log "🎉 Docker 已成功卸载。"
log "💡 如需重新安装，请参考官方文档："
log "   • Ubuntu/Debian: https://docs.docker.com/engine/install/ubuntu/"
log "   • CentOS/RHEL: https://docs.docker.com/engine/install/centos/"


# 示例：Linux/macOS 下获取 CPU >10% 或 内存 >500MB 的进程（按需可调阈值）
ps -eo pid,ppid,%cpu,%mem,vsz,rss,comm --sort=-%cpu | head -n 11 | sed '1s/^/PID   PPID  %CPU  %MEM   VSZ     RSS   CMD\n/'
ps -eo pid,ppid,%cpu,%mem,vsz,rss,comm --sort=-%mem | head -n 11 | sed '1s/^/PID   PPID  %CPU  %MEM   VSZ     RSS   CMD\n/'
