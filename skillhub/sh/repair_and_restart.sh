#!/bin/bash
# 一键修复 ARD Skill Hub 注册表并重启服务

echo "=== 一键修复 ARD Skill Hub ==="
echo

# 1. 备份并删除损坏的注册表
echo "📋 1. 修复注册表..."
if [ -f ~/.ardc/registry.json ]; then
    cp ~/.ardc/registry.json ~/.ardc/registry.json.bak 2>/dev/null || true
    rm ~/.ardc/registry.json 2>/dev/null || true
fi

# 2. 重新创建目录
mkdir -p ~/.ardc/skills

# 3. 停止旧服务
echo "📋 2. 停止旧服务..."
pkill -f "uvicorn ardc.api.main" 2>/dev/null || true
sleep 2

# 4. 启动新服务
echo "📋 3. 启动服务..."
cd ~/czq/anime_role_detect/skillhub
nohup python3 -m uvicorn ardc.api.main:app --host 127.0.0.1 --port 8000 --workers 1 > nohup.out 2>&1 &
sleep 3

# 5. 验证服务
echo "📋 4. 验证服务..."
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ 服务启动成功!"
    echo
    echo "📝 测试命令:"
    echo "   curl http://localhost:8000/api/health"
    echo "   curl -X POST http://localhost:8000/api/auth/login -H 'Content-Type: application/json' -d '{\"username\":\"testdev\",\"password\":\"dev1234\"}'"
else
    echo "❌ 服务启动失败，请检查日志"
fi
