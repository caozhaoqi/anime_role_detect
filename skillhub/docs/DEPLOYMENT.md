# ARD Skill Hub 部署指南

## 概述

本文档详细描述如何在公网服务器上部署 ARD（Anime Role Detect）技能仓库系统。

## 环境要求

### 硬件要求
- CPU: 2核及以上
- 内存: 4GB及以上
- 存储: 10GB及以上可用空间

### 软件要求
- Python 3.8+
- Node.js 16+
- Docker 20.10+（可选）
- Docker Compose 2.0+（可选）
- Nginx（可选，用于反向代理）

## 部署方式

### 方式一：Docker Compose（推荐）

这是最简单的部署方式，适合生产环境。

```bash
# 克隆项目
git clone https://github.com/your-username/anime_role_detect.git
cd anime_role_detect/skillhub

# 启动服务（后台模式）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

**服务访问：**
- Web 界面: `http://localhost`
- API 接口: `http://localhost/api/`
- Prometheus 监控: `http://localhost:9090`

**停止服务：**
```bash
docker-compose down
```

### 方式二：手动部署

#### 1. 安装依赖

```bash
cd skillhub

# 后端依赖
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 前端依赖
cd web
npm install
npm run build
cd ..
```

#### 2. 启动后端服务

```bash
source venv/bin/activate

# 开发模式
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 3. 部署前端

```bash
cd web
npm run build

# 使用 serve 部署
npm install -g serve
serve -s dist -l 5173

# 或使用 nginx 部署
# 将 dist 目录内容复制到 nginx html 目录
```

### 方式三：使用 Makefile

```bash
cd skillhub

# 安装依赖
make install

# 开发模式
make run-dev

# 生产模式
make run-prod

# 完整部署
make deploy
```

### 方式四：系统服务（systemd）

创建服务文件 `/etc/systemd/system/ardc.service`：

```ini
[Unit]
Description=ARD Skill Repository API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/skillhub
Environment="PATH=/path/to/skillhub/venv/bin"
ExecStart=/path/to/skillhub/venv/bin/uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable ardc
sudo systemctl start ardc

# 查看状态
sudo systemctl status ardc

# 查看日志
sudo journalctl -u ardc -f
```

## Nginx 反向代理配置

创建配置文件 `/etc/nginx/sites-available/ardc`：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /path/to/skillhub/web/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 监控指标
    location /metrics {
        proxy_pass http://127.0.0.1:8000/metrics;
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/ardc /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## HTTPS 配置（推荐）

使用 Let's Encrypt 获取免费 SSL 证书：

```bash
# 安装 Certbot
sudo apt update && sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

配置完成后，Nginx 会自动更新配置文件。

## 安全配置

### 防火墙设置

```bash
# 允许 SSH
sudo ufw allow ssh

# 允许 HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 如果直接访问 API，允许 8000 端口
sudo ufw allow 8000/tcp

# 启用防火墙
sudo ufw enable
```

### 访问控制（可选）

可以在 API 层面添加认证机制：

```python
# ardc/api/main.py
from fastapi import Header, HTTPException

async def get_api_key(api_key: str = Header(None)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
```

## 数据备份与恢复

### 备份

```bash
# 备份整个数据目录
tar -czvf ardc_backup_$(date +%Y%m%d).tar.gz ~/.ardc/

# 备份数据库文件
cp ~/.ardc/registry.json ~/.ardc/registry.json.bak
cp ~/.ardc/skill_index.json ~/.ardc/skill_index.json.bak
```

### 恢复

```bash
# 恢复备份
tar -xzvf ardc_backup_20240101.tar.gz -C /

# 恢复单个文件
cp ~/.ardc/registry.json.bak ~/.ardc/registry.json
```

## 性能优化

### 后端优化

```bash
# 使用多个工作进程
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# 启用 gzip 压缩（通过 Nginx）
gzip on;
gzip_types text/plain text/css application/json application/javascript;
```

### 前端优化

```bash
# 生产构建
npm run build

# 启用缓存
# 在 Nginx 中添加缓存配置
location ~* \.(js|css|png|jpg|jpeg|gif|ico)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## 监控与日志

### Prometheus 监控

访问 `http://your-domain.com/metrics` 查看指标。

主要指标：
- `ardc_skill_executions_total` - 技能执行次数
- `ardc_requests_total` - API 请求次数
- `ardc_uptime_seconds` - 服务运行时间
- `ardc_skill_execution_latency_seconds` - 技能执行延迟

### 日志管理

```bash
# 查看 Nginx 日志
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# 查看应用日志
sudo journalctl -u ardc

# 查看 Docker 日志
docker-compose logs -f backend
```

## 故障排查

### 服务无法启动

```bash
# 检查端口占用
netstat -tlnp | grep 8000

# 检查 Python 环境
source venv/bin/activate
python -c "import ardc; print(ardc.__version__)"

# 检查依赖
pip list | grep -E "fastapi|uvicorn|pydantic"
```

### API 无法访问

```bash
# 测试本地 API
curl http://localhost:8000/api/skills

# 检查防火墙
ufw status

# 检查 Nginx 配置
nginx -t
```

### 前端无法访问

```bash
# 检查前端服务
curl http://localhost:5173

# 检查 Nginx 配置
cat /var/log/nginx/error.log
```

### 技能安装失败

```bash
# 检查技能 ID 是否正确
ardc skill search <skill_id>

# 检查数据目录权限
ls -la ~/.ardc/

# 检查网络连接
curl -v http://localhost:8000/api/skills/<skill_id>
```

## 更新维护

### 更新代码

```bash
cd skillhub

# 拉取最新代码
git pull

# 更新后端依赖
source venv/bin/activate
pip install -r requirements.txt

# 更新前端
cd web
npm install
npm run build

# 重启服务
sudo systemctl restart ardc
```

### 版本升级

```bash
# 备份数据
tar -czvf ardc_backup_$(date +%Y%m%d).tar.gz ~/.ardc/

# 升级版本
git checkout v2.0.0
pip install -r requirements.txt
npm install && npm run build

# 重启服务
sudo systemctl restart ardc
```

## 常见问题

**Q: 如何添加新技能？**

A: 使用 Web 界面或 CLI 命令注册技能：
```bash
ardc skill register --id ardc-my-skill --name "我的技能" --version 1.0.0 --author me --category utility --entry-point scripts/main.py
```

**Q: 如何发布新版本？**

A: 使用相同的技能 ID 和不同的版本号注册即可。

**Q: 如何卸载技能？**

A: 使用 CLI 命令或 Web 界面卸载：
```bash
ardc skill uninstall ardc-my-skill
```

**Q: 监控指标在哪里查看？**

A: 访问 `http://your-domain.com/metrics`。

**Q: 如何配置域名？**

A: 在 Nginx 配置文件中修改 `server_name` 指令，并确保 DNS 解析正确。

**Q: 如何设置 HTTPS？**

A: 使用 Let's Encrypt 获取 SSL 证书：
```bash
sudo certbot --nginx -d your-domain.com
```

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      客户端浏览器                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTPS
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Nginx                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  / → 前端静态文件 (dist)      /api/ → Backend API      │   │
│  │  /metrics → Prometheus指标                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │
│  │  store  │  │ version │  │workflow │  │   monitoring    │   │
│  │ (存储)  │  │ (版本)  │  │ (引擎)  │  │ (Prometheus)    │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ 文件系统
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据存储 (~/.ardc/)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ registry.json│ │skill_index.json│ │   skills/   │           │
│  │ (技能注册)   │ │  (搜索索引)   │ │ (已安装技能) │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│  ┌──────────────┐ ┌──────────────┐                             │
│  │  versions/  │ │ workflows/   │                             │
│  │ (版本历史)   │ │ (工作流定义) │                             │
│  └──────────────┘ └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## 联系方式

如有问题，请提交 Issue 或联系维护人员。
