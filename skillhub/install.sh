#!/bin/bash
set -e

ARD_INSTALL_DIR="${HOME}/.ardc"
ARD_BIN_DIR="${ARD_INSTALL_DIR}/bin"
ARD_CLI_URL="http://47.79.91.89:8888/api/install/cli.py"

echo "🚀 正在安装 ARD Skill Hub CLI 工具..."

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "✅ 检测到 Python 版本: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ 错误: 需要 Python 3.8 或更高版本"
    echo "当前版本: $PYTHON_VERSION"
    exit 1
fi

# 创建安装目录
echo "📁 创建安装目录: $ARD_INSTALL_DIR"
mkdir -p "$ARD_INSTALL_DIR"
mkdir -p "$ARD_BIN_DIR"

# 下载 CLI 工具
echo "📥 下载 CLI 工具..."
if command -v curl &> /dev/null; then
    curl -fsSL "$ARD_CLI_URL" -o "$ARD_BIN_DIR/ardc"
elif command -v wget &> /dev/null; then
    wget -q "$ARD_CLI_URL" -O "$ARD_BIN_DIR/ardc"
else
    echo "❌ 错误: 需要 curl 或 wget 来下载 CLI 工具"
    exit 1
fi

# 设置执行权限
chmod +x "$ARD_BIN_DIR/ardc"

# 创建符号链接
echo "🔗 创建命令行链接..."
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.profile"
fi

# 添加 PATH
if ! grep -q "$ARD_BIN_DIR" "$SHELL_CONFIG" 2>/dev/null; then
    echo "" >> "$SHELL_CONFIG"
    echo "# ARD Skill Hub CLI" >> "$SHELL_CONFIG"
    echo "export PATH=\"\$PATH:$ARD_BIN_DIR\"" >> "$SHELL_CONFIG"
    echo "✅ 已添加 PATH 到 $SHELL_CONFIG"
else
    echo "✅ PATH 已配置"
fi

# 安装 Python 依赖
echo "📦 安装 Python 依赖..."
pip3 install --user requests pydantic fastapi uvicorn 2>/dev/null || {
    echo "⚠️  警告: 部分 Python 依赖安装失败"
    echo "请手动运行: pip3 install requests pydantic fastapi uvicorn"
}

# 创建配置文件
CONFIG_FILE="$ARD_INSTALL_DIR/config.json"
cat > "$CONFIG_FILE" << EOF
{
  "api_url": "http://47.79.91.89:8888/api",
  "install_dir": "$ARD_INSTALL_DIR",
  "version": "1.0.0",
  "installed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "✅ 配置文件已创建: $CONFIG_FILE"

# 显示安装信息
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║           ARD Skill Hub CLI 安装完成 ✅                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📍 安装位置: $ARD_BIN_DIR"
echo "📄 配置文件: $CONFIG_FILE"
echo ""
echo "📝 使用方法:"
echo "   1. 重新加载配置文件:"
echo "      source $SHELL_CONFIG"
echo ""
echo "   2. 验证安装:"
echo "      ardc --version"
echo ""
echo "   3. 查看帮助:"
echo "      ardc --help"
echo ""
echo "   4. 列出所有技能:"
echo "      ardc skill list"
echo ""
echo "   5. 搜索技能:"
echo "      ardc skill search <关键词>"
echo ""
echo "   6. 安装技能:"
echo "      ardc skill install <技能ID>"
echo ""
echo "🔗 更多信息请访问: http://47.79.91.89:8888"
echo ""