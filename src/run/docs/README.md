# 动漫角色识别系统 - 服务管理

## 📋 目录结构

```
src/run/
├── services_config.py      # 服务配置文件（端口分配）
├── start_all.py           # 统一启动脚本
├── monitor_dashboard.py    # 监控仪表板
└── README.md             # 本文档
```

## 🚀 快速开始

### 1. 启动所有核心服务

```bash
cd /Users/caozhaoqi/PycharmProjects/anime_role_detect
python3 src/run/start_all.py -g core
```

### 2. 访问监控仪表板

打开浏览器访问: **http://localhost:9000**

监控仪表板会自动每5秒刷新一次，显示所有服务的实时状态。

## 📊 端口分配计划

| 服务名称 | 端口 | 状态 | 描述 |
|----------|------|------|------|
| 搜索服务 | 8002 | ✅ 运行中 | 以图搜图服务，支持图像特征提取和FAISS向量搜索 |
| 主API服务 | 8001 | ✅ 运行中 | 主API网关，提供角色识别和视频处理接口 |
| 模型服务 | 8000 | ⏳ 暂未启用 | AI模型推理服务 |
| 视频服务 | 8003 | ⏳ 暂未启用 | 视频实时抽帧和识别服务 |
| API网关 | 8080 | ⏳ 暂未启用 | 统一API网关 |
| 监控仪表板 | 9000 | ✅ 运行中 | 服务监控仪表板 |

## 🎯 使用方式

### 启动服务

```bash
# 列出所有服务配置
python3 src/run/start_all.py --list

# 启动核心服务（推荐）
python3 src/run/start_all.py -g core

# 启动所有服务（包括辅助服务）
python3 src/run/start_all.py -g all

# 启动特定分组
python3 src/run/start_all.py -g ai      # AI模型服务
python3 src/run/start_all.py -g video   # 视频服务
python3 src/run/start_all.py -g gateway # API网关
```

### 独立启动监控仪表板

```bash
cd /Users/caozhaoqi/PycharmProjects/anime_role_detect
PYTHONPATH=/Users/caozhaoqi/PycharmProjects/anime_role_detect \
python3 src/run/monitor_dashboard.py
```

## 📚 API文档

| 服务 | 基础URL | Swagger文档 |
|------|---------|-------------|
| 搜索服务 | http://localhost:8002 | http://localhost:8002/docs |
| 主API服务 | http://localhost:8001 | http://localhost:8001/docs |

## 🔍 监控仪表板功能

### 主要特性

1. **实时监控**: 自动每5秒刷新一次服务状态
2. **健康检查**: 实时检测所有服务的健康状态
3. **响应时间**: 显示每个服务的响应时间
4. **服务分类**: 区分核心服务和辅助服务
5. **快速访问**: 一键跳转到各服务的API文档

### 监控接口

```bash
# 获取所有服务状态（JSON格式）
curl http://localhost:9000/api/status

# 健康检查
curl http://localhost:9000/api/health
```

### 状态说明

| 状态 | 说明 |
|------|------|
| ✅ 运行正常 | 服务健康运行 |
| ❌ 无法连接 | 服务未启动或网络问题 |
| ⏰ 超时 | 服务响应超时 |
| ⏸️ 已禁用 | 服务在配置中被禁用 |
| ❓ 未知 | 无法确定服务状态 |

## 🏗️ 服务架构

### 为什么有两个API服务？

| 服务 | 职责 | 技术栈 |
|------|------|--------|
| **搜索服务 (8002)** | 专门处理图像搜索和视频识别 | FAISS向量搜索 + 传统图像特征 |
| **主API服务 (8001)** | 作为API网关，聚合所有后端服务 | FastAPI + HTTP客户端 |

**架构优势：**
- **解耦性**: 搜索服务独立部署，便于横向扩展
- **容错性**: 单个服务故障不影响其他服务
- **性能优化**: 搜索服务可以独立优化和扩展
- **技术隔离**: 搜索服务使用传统图像特征避免PyTorch依赖问题

## 🛠️ 配置说明

### 修改服务端口

编辑 `src/run/services_config.py` 文件：

```python
SERVICES = {
    "search_service": {
        "port": 8002,  # 修改端口
        # ...
    }
}
```

### 启用/禁用服务

```python
SERVICES = {
    "model_service": {
        "enabled": True,  # 设置为 True 启用服务
        # ...
    }
}
```

## 📝 停止服务

按 `Ctrl+C` 停止所有服务。

## 🐛 故障排除

### 端口被占用

```bash
# 查找占用端口的进程
lsof -ti:8001 | xargs -r kill -9
lsof -ti:8002 | xargs -r kill -9
lsof -ti:9000 | xargs -r kill -9
```

### 服务启动失败

1. 检查Python路径是否正确
2. 检查依赖是否安装完整
3. 查看服务日志输出

### 监控仪表板无法访问

1. 确认监控服务已启动
2. 检查端口9000是否被占用
3. 查看浏览器控制台错误信息

## 📞 支持

如有问题，请查看项目文档或提交Issue。