#!/bin/bash
# 修复损坏的 ARD Skill Hub 注册表

echo "=== 修复 ARD Skill Hub 注册表 ==="
echo

# 备份损坏的文件
if [ -f ~/.ardc/registry.json ]; then
    echo "📋 1. 备份损坏的注册表文件"
    cp ~/.ardc/registry.json ~/.ardc/registry.json.bak
    echo "   备份完成: ~/.ardc/registry.json.bak"
    echo

    # 删除损坏的文件
    echo "📋 2. 删除损坏的注册表文件"
    rm ~/.ardc/registry.json
    echo "   删除完成"
    echo
fi

# 清理已安装技能目录（可选）
echo "📋 3. 可选：清理已安装技能目录"
read -p "   是否清理已安装技能目录？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d ~/.ardc/skills ]; then
        rm -rf ~/.ardc/skills/*
        echo "   清理完成"
    fi
fi
echo

# 重新创建目录结构
echo "📋 4. 重新创建目录结构"
mkdir -p ~/.ardc/skills
echo "   创建完成: ~/.ardc/skills/"
echo

echo "✅ 修复完成！"
echo "   请重启服务: ./restart.sh"
