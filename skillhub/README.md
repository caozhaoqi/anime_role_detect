# ARD Skill Hub

Anime Role Detect 技能仓库系统 - 一个现代化的技能管理平台，支持技能注册、版本控制、搜索发现和工作流编排。

## 功能特性

- **技能仓库** - 一站式技能管理，支持注册、安装、卸载
- **版本控制** - 语义化版本管理，支持版本回滚
- **全文搜索** - 支持关键词搜索和分类筛选
- **Web 界面** - 直观的可视化界面，参考开源 SkillHub 设计
- **工作流引擎** - 支持技能编排和流程自动化
- **性能监控** - 集成 Prometheus 指标收集

## 技术栈

### 后端
- Python 3.8+
- FastAPI
- Pydantic
- Prometheus Client

### 前端
- Vue 3 + Vite
- Tailwind CSS 3
- Lucide Icons

### 部署
- Docker / Docker Compose
- Nginx (反向代理)
- Prometheus (监控)

## 快速开始

### 方法一：Docker Compose（推荐）

```bash
# 克隆项目
git clone https://github.com/your-username/anime_role_detect.git
cd anime_role_detect/skillhub

# 启动服务（后台运行）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart
```

启动后访问：
- Web 界面: `http://localhost`
- API 接口: `http://localhost/api/`
- Prometheus: `http://localhost:9090`

### 方法二：手动部署（生产环境）

#### 1. 前端打包构建

```bash
cd skillhub/web

# 安装依赖
npm install

# 生产构建（输出到 dist 目录）
npm run build

# 构建产物位置
ls -la dist/
```

构建完成后，`dist/` 目录包含所有静态资源，可直接部署到 Nginx 或 CDN。

#### 2. 后端服务（后台运行）

```bash
cd skillhub

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 方式一：使用 nohup 后台运行
nohup uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4 > ardc.log 2>&1 &

# 方式二：使用 systemd 服务（推荐）
# 创建服务文件
sudo tee /etc/systemd/system/ardc.service <<EOF
[Unit]
Description=ARD Skill Hub API Service
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/anime_role_detect/skillhub
ExecStart=/path/to/anime_role_detect/skillhub/venv/bin/uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start ardc
sudo systemctl enable ardc  # 设置开机自启

# 查看服务状态
sudo systemctl status ardc

# 查看日志
sudo journalctl -u ardc -f
```

#### 3. Nginx 反向代理配置

```nginx
server {
    listen 8888;
    server_name caozhaoqi.top;

    # 前端静态文件
    location / {
        root /czq/anime_role_detect/skillhub/web/dist;
        try_files $uri $uri/ /index.html;
    }

    # API 反向代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

## 项目结构

```
skillhub/
├── ardc/                    # 后端核心模块
│   ├── api/                 # RESTful API 服务
│   │   ├── __init__.py
│   │   └── main.py          # API 主入口
│   ├── cli/                 # 命令行工具
│   │   ├── __init__.py
│   │   └── cli.py           # CLI 主入口
│   ├── monitoring/          # Prometheus 监控
│   │   ├── __init__.py
│   │   └── metrics.py       # 指标收集器
│   ├── store/               # 技能存储模块
│   │   ├── __init__.py
│   │   ├── metadata.py      # 数据模型定义
│   │   ├── registry.py      # 技能注册中心
│   │   └── index.py         # 全文索引搜索
│   ├── version/             # 版本管理
│   │   ├── __init__.py
│   │   └── manager.py       # 版本控制与回滚
│   ├── workflow/            # 工作流引擎
│   │   ├── __init__.py
│   │   └── engine.py        # 工作流编排引擎
│   └── __init__.py
├── skills/                  # 技能包目录
│   ├── ardc-client/         # ARD 客户端技能
│   │   ├── scripts/         # 脚本文件
│   │   └── SKILL.md         # 技能描述
│   └── ardc-collector/      # 数据采集技能
│       ├── scripts/         # 脚本文件
│       └── SKILL.md         # 技能描述
├── web/                     # 前端 Web 界面
│   ├── src/
│   │   ├── api/
│   │   │   └── skillApi.js  # API 请求封装
│   │   ├── components/
│   │   │   ├── Header.vue           # 顶部导航
│   │   │   ├── CategoryFilter.vue   # 分类筛选
│   │   │   ├── SkillList.vue        # 技能列表
│   │   │   ├── SkillDetail.vue      # 技能详情
│   │   │   └── RegisterSkill.vue    # 注册技能
│   │   ├── App.vue           # 主应用组件
│   │   ├── main.js           # 入口文件
│   │   └── style.css         # 全局样式
│   ├── dist/                 # 生产构建产物（npm run build 生成）
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── Dockerfile.backend        # 后端 Docker 镜像
├── Dockerfile.frontend       # 前端 Docker 镜像
├── docker-compose.yml        # Docker Compose 配置
├── nginx.conf                # Nginx 配置
├── prometheus.yml            # Prometheus 配置
├── Makefile                  # 自动化脚本
├── requirements.txt          # Python 依赖
└── setup.py                  # 安装配置
```

## API 接口

### 基础路径
```
http://your-domain.com/api/
```

### 接口列表

| 方法 | 路径 | 描述 | 参数 |
|------|------|------|------|
| GET | `/skills` | 获取技能列表 | `category` (可选) |
| GET | `/skills/{skill_id}` | 获取技能详情 | `version` (可选) |
| POST | `/skills` | 注册新技能 | JSON 数据 |
| DELETE | `/skills/{skill_id}` | 删除技能 | - |
| GET | `/skills/{skill_id}/versions` | 获取版本列表 | - |
| POST | `/skills/{skill_id}/install` | 安装技能 | `version` (可选) |
| DELETE | `/skills/{skill_id}/uninstall` | 卸载技能 | - |
| GET | `/search` | 搜索技能 | `keyword`, `category`, `limit` |
| GET | `/tags` | 获取所有标签 | - |
| GET | `/categories` | 获取所有分类 | - |
| GET | `/stats` | 获取统计信息 | - |
| GET | `/health` | 健康检查 | - |

### 注册技能示例

```bash
curl -X POST http://localhost:8000/api/skills \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ardc-my-skill",
    "name": "我的技能",
    "version": "1.0.0",
    "description": "这是一个示例技能",
    "author": "your-name",
    "category": "utility",
    "entry_point": "scripts/main.py",
    "tags": ["示例", "工具"],
    "release_notes": "初始版本"
  }'
```

## CLI 命令

```bash
# 安装 CLI
pip install -e .

# 安装技能
ardc skill install ardc-collector

# 列出技能
ardc skill list

# 搜索技能
ardc skill search 采集

# 注册技能
ardc skill register \
  --id ardc-my-skill \
  --name "我的技能" \
  --version 1.0.0 \
  --author "your-name" \
  --category utility \
  --entry-point scripts/main.py

# 查看版本列表
ardc version list ardc-collector

# 版本回滚
ardc version rollback ardc-collector 1.0.0

# 查看统计信息
ardc stats
```

## 技能分类

| 分类 | 说明 |
|------|------|
| collector | 数据采集 |
| cleaner | 数据清洗 |
| classifier | 分类识别 |
| trainer | 模型训练 |
| search | 搜索检索 |
| analyzer | 数据分析 |
| utility | 工具辅助 |

## 技能状态

| 状态 | 说明 |
|------|------|
| stable | 稳定版 |
| testing | 测试中 |
| development | 开发中 |
| deprecated | 已弃用 |

## 配置说明

### 数据存储路径

默认数据存储在用户主目录下：
- 注册表: `~/.ardc/registry.json`
- 索引: `~/.ardc/skill_index.json`
- 技能文件: `~/.ardc/skills/`
- 版本信息: `~/.ardc/versions/`
- 工作流: `~/.ardc/workflows/`

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| ARDC_DATA_DIR | 数据存储目录 | `~/.ardc` |
| PORT | API 服务端口 | 8000 |

## 监控指标

访问 `http://localhost:8000/metrics` 查看 Prometheus 指标：

```
# HELP ardc_skill_executions_total Total skill executions
# TYPE ardc_skill_executions_total counter
ardc_skill_executions_total{skill_id="ardc-collector",status="success",version="1.0.0"} 5

# HELP ardc_requests_total Total API requests
# TYPE ardc_requests_total counter
ardc_requests_total{endpoint="/api/skills",method="GET",status_code="200"} 10

# HELP ardc_uptime_seconds Service uptime in seconds
# TYPE ardc_uptime_seconds gauge
ardc_uptime_seconds 3600
```

## 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      客户端浏览器                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Nginx                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 静态文件 (Vue 前端)         API 反向代理            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌───────────────────┐         ┌───────────────────┐
│   Backend API     │         │    Prometheus     │
│   (FastAPI)       │◄────────│   (监控采集)      │
└───────────────────┘         └───────────────────┘
          │
          ▼
┌───────────────────┐
│   数据存储         │
│ (~/.ardc/)        │
└───────────────────┘
```

## 运维指南

### 后台运行方式

| 方式 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| Docker Compose | 开发/测试/生产 | 一键部署、隔离性好 | 需要 Docker 环境 |
| systemd | 生产环境 | 系统级管理、开机自启 | 配置稍复杂 |
| nohup | 临时运行 | 简单快捷 | 无自动重启 |

### 日志管理

```bash
# Docker Compose 日志
docker-compose logs -f backend
docker-compose logs --tail=100 backend

# systemd 日志
sudo journalctl -u ardc -f
sudo journalctl -u ardc --since "1 hour ago"

# nohup 日志
tail -f ardc.log
```

### 备份与恢复

```bash
# 备份数据目录
tar -czvf ardc_backup_$(date +%Y%m%d).tar.gz ~/.ardc/

# 恢复数据
tar -xzvf ardc_backup_20240101.tar.gz -C ~/
```

### 性能优化建议

1. **增加工作进程数**: 根据 CPU 核心数调整 `--workers` 参数
2. **启用 Gunicorn**: 生产环境推荐使用 Gunicorn + Uvicorn 组合
3. **配置 Nginx 缓存**: 对静态文件设置适当的缓存策略
4. **定期清理日志**: 设置日志轮转防止磁盘空间溢出

## 开发指南

### 添加新技能

1. 使用 Web 界面或 CLI 注册技能
2. 编写技能实现代码
3. 上传技能包到仓库

### 扩展功能

1. 在 `ardc/` 目录下创建新模块
2. 添加对应的 API 端点
3. 更新前端组件

## 故障排查

### 服务无法启动

```bash
# 查看日志
docker-compose logs backend

# 检查端口占用
netstat -tlnp | grep 8000

# 检查 Docker 状态
docker ps -a
```

### API 无法访问

```bash
# 测试 API
curl http://localhost:8000/api/skills

# 检查防火墙
ufw status

# 检查 Nginx 配置
nginx -t
```

### 技能安装失败

检查技能 ID 是否正确，以及网络连接是否正常。

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或联系维护人员。