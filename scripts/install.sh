#!/bin/bash
# ============================================================
# ARDC SkillHub 一键安装脚本
# ============================================================
# 支持系统: macOS / Linux
# 依赖: Python 3.8+
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 版本信息
VERSION="1.0.0"

# 下载地址
SKILL_SYNC_URL="https://47.79.91.89:8888/api/install/ardc-skill-sync.py"
CONFIG_URL="https://47.79.91.89:8888/api/install/config.json"

# 安装目录
INSTALL_DIR="$HOME/.ardc"
SKILL_DIR="$INSTALL_DIR/skills"
BIN_DIR="$HOME/.local/bin"

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}       ARDC SkillHub 一键安装脚本 v${VERSION}${NC}"
echo -e "${BLUE}==============================================${NC}"
echo

# ============================================================
# 检查 Python 版本
# ============================================================
check_python() {
    echo -e "${YELLOW}检查 Python 环境...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 未安装${NC}"
        echo -e "${BLUE}请安装 Python 3.8+ 后重试${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        echo -e "${RED}❌ Python 版本过低: ${PYTHON_VERSION}${NC}"
        echo -e "${BLUE}需要 Python 3.8+${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"
}

# ============================================================
# 创建目录结构
# ============================================================
create_directories() {
    echo -e "${YELLOW}创建目录结构...${NC}"
    
    mkdir -p "$SKILL_DIR"
    mkdir -p "$BIN_DIR"
    
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# ============================================================
# 下载技能同步工具
# ============================================================
download_skill_sync() {
    echo -e "${YELLOW}下载技能同步工具...${NC}"
    
    if command -v curl &> /dev/null; then
        curl -fsSL -o "$BIN_DIR/ardc-skill-sync" "$SKILL_SYNC_URL"
    elif command -v wget &> /dev/null; then
        wget -q -O "$BIN_DIR/ardc-skill-sync" "$SKILL_SYNC_URL"
    else
        echo -e "${RED}❌ 未找到 curl 或 wget${NC}"
        exit 1
    fi
    
    chmod +x "$BIN_DIR/ardc-skill-sync"
    
    echo -e "${GREEN}✓ 技能同步工具下载完成${NC}"
}

# ============================================================
# 创建配置文件
# ============================================================
create_config() {
    echo -e "${YELLOW}创建配置文件...${NC}"
    
    # 如果配置文件不存在，创建默认配置
    if [ ! -f "$INSTALL_DIR/config.json" ]; then
        cat > "$INSTALL_DIR/config.json" <<EOF
{
  "skill_hub_url": "http://47.79.91.89:8888",
  "timeout": 30,
  "log_level": "INFO",
  "auto_update": true
}
EOF
    fi
    
    echo -e "${GREEN}✓ 配置文件创建完成${NC}"
}

# ============================================================
# 安装依赖
# ============================================================
install_dependencies() {
    echo -e "${YELLOW}安装依赖包...${NC}"
    
    python3 -m pip install requests --quiet
    
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
}

# ============================================================
# 设置环境变量
# ============================================================
setup_environment() {
    echo -e "${YELLOW}设置环境变量...${NC}"
    
    # 检查 .bashrc
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "export PATH.*\.local/bin" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        fi
    fi
    
    # 检查 .zshrc
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "export PATH.*\.local/bin" "$HOME/.zshrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
        fi
    fi
    
    # 检查 .profile
    if [ -f "$HOME/.profile" ]; then
        if ! grep -q "export PATH.*\.local/bin" "$HOME/.profile"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
        fi
    fi
    
    echo -e "${GREEN}✓ 环境变量设置完成${NC}"
}

# ============================================================
# 显示安装成功信息
# ============================================================
show_success() {
    echo
    echo -e "${GREEN}==============================================${NC}"
    echo -e "${GREEN}          安装成功！${NC}"
    echo -e "${GREEN}==============================================${NC}"
    echo
    echo -e "${BLUE}安装位置:${NC}"
    echo "  - 技能目录: $SKILL_DIR"
    echo "  - 工具脚本: $BIN_DIR/ardc-skill-sync"
    echo "  - 配置文件: $INSTALL_DIR/config.json"
    echo
    echo -e "${BLUE}使用方法:${NC}"
    echo "  # 登录认证"
    echo "  ardc-skill-sync login"
    echo
    echo "  # 查看技能列表"
    echo "  ardc-skill-sync list"
    echo
    echo "  # 安装技能"
    echo "  ardc-skill-sync install ardc-collector"
    echo
    echo "  # 检查更新"
    echo "  ardc-skill-sync check"
    echo
    echo -e "${YELLOW}注意: 需要重新打开终端或执行 source ~/.bashrc${NC}"
    echo -e "${YELLOW}以便环境变量生效${NC}"
    echo
}

# ============================================================
# 主流程
# ============================================================
main() {
    check_python
    create_directories
    download_skill_sync
    create_config
    install_dependencies
    setup_environment
    show_success
}

# 执行安装
main
