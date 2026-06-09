# ARD Skill Hub 部署指南

## 概述

本文档详细描述如何在公网服务器上部署 ARD（Anime Role Detect）技能仓库系统。

---

## 环境要求

### 硬件要求
| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 2核 | 4核及以上 |
| 内存 | 4GB | 8GB及以上 |
| 存储 | 10GB可用空间 | 20GB及以上 |

### 软件要求
- **Python**: 3.8+
- **Node.js**: 16+
- **Docker**: 20.10+（可选）
- **Docker Compose**: 2.0+（可选）
- **Nginx**: 1.20+（推荐用于反向代理）

---

## 部署方式

### 方式一：Docker Compose（推荐）

这是最简单的部署方式，适合开发、测试和生产环境。

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

# 停止服务
docker-compose down

# 重启服务
docker-compose restart
```

**服务访问：**
- Web 界面: `http://localhost`
- API 接口: `http://localhost/api/`
- Prometheus 监控: `http://localhost:9090`

### 方式二：手动部署（生产环境）

#### 1. 安装依赖

```bash
cd skillhub

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装后端依赖
pip install -r requirements.txt
pip install -e .

# 安装前端依赖并构建
cd web
npm install
npm run build
cd ..
```

#### 2. 启动后端服务

**开发模式（带热重载）：**
```bash
source .venv/bin/activate
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**生产模式：**
```bash
source .venv/bin/activate
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 3. 使用 Makefile（简化操作）

```bash
cd skillhub

# 安装所有依赖
make install

# 开发模式
make run-dev

# 生产模式
make run-prod

# 完整部署（安装 + 前端构建）
make deploy
```

### 方式三：系统服务（systemd）

项目已提供预配置的 systemd 服务文件：

```bash
# 复制服务配置
cp conf/ardc-api.service /etc/systemd/system/

# 根据实际路径修改配置
sed -i 's|/root/czq/anime_role_detect|/your/actual/path/to/anime_role_detect|g' /etc/systemd/system/ardc-api.service

# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable ardc-api
sudo systemctl start ardc-api

# 查看状态
sudo systemctl status ardc-api

# 查看日志
sudo journalctl -u ardc-api -f
```

---

## Nginx 反向代理配置

### 基础配置

创建或修改 `/etc/nginx/sites-available/default`：

```nginx
server {
    listen 8888;
    server_name caozhaoqi.top;

    # 前端静态文件目录
    root /var/www/ardc-web;
    index index.html;

    # 静态资源缓存（重要：提升性能）
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header Access-Control-Allow-Origin *;
    }

    # API 反向代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 连接超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    # Prometheus 监控指标
    location /metrics {
        proxy_pass http://127.0.0.1:8000/metrics;
    }

    # SPA 路由回退（关键：使用 named location 避免循环）
    location / {
        try_files $uri $uri/ @fallback;
    }

    location @fallback {
        rewrite ^ /index.html break;
    }
}
```

### 启用配置

```bash
# 验证配置
sudo nginx -t

# 重启服务
sudo systemctl reload nginx
```

---

## HTTPS 配置（推荐）

使用 Let's Encrypt 获取免费 SSL 证书：

```bash
# 安装 Certbot
sudo apt update && sudo apt install certbot python3-certbot-nginx -y

# 获取证书（自动配置 Nginx）
sudo certbot --nginx -d caozhaoqi.top

# 测试自动续期
sudo certbot renew --dry-run
```

配置完成后，Nginx 会自动更新配置文件，包括重定向 HTTP 到 HTTPS。

---

## 安全配置

### 防火墙设置

```bash
# 允许 SSH（保持远程访问）
sudo ufw allow ssh

# 允许 HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 如果使用 8888 端口（开发/测试）
sudo ufw allow 8888/tcp

# 如果直接访问 API（不推荐，应通过 Nginx 代理）
# sudo ufw allow 8000/tcp

# 启用防火墙
sudo ufw enable

# 查看状态
sudo ufw status
```

### 目录权限

**重要：Nginx 以 `www-data` 用户运行，无法访问 `/root/` 目录！**

```bash
# 创建标准 Web 目录（推荐）
mkdir -p /var/www/ardc-web

# 复制前端构建文件
cp -r /path/to/skillhub/web/dist/* /var/www/ardc-web/

# 设置正确权限
chown -R www-data:www-data /var/www/ardc-web
chmod -R 755 /var/www/ardc-web
```

### 环境变量配置

创建 `.env` 文件或在启动时设置环境变量：

```bash
# 设置环境变量
export JWT_SECRET_KEY="your-secret-key-here-keep-it-safe"
export ALLOWED_ORIGINS="http://localhost:3000,https://your-domain.com"
export ARD_C_DATA_DIR="/var/lib/ardc"

# 启动服务
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 数据备份与恢复

### 备份

```bash
# 备份整个数据目录
tar -czvf ardc_backup_$(date +%Y%m%d).tar.gz ~/.ardc/

# 备份到远程服务器（可选）
scp ardc_backup_$(date +%Y%m%d).tar.gz user@backup-server:/backup/
```

### 恢复

```bash
# 恢复备份
tar -xzvf ardc_backup_20240101.tar.gz -C ~/

# 恢复后重启服务
sudo systemctl restart ardc-api
```

---

## 性能优化

### 后端优化

```bash
# 使用多个工作进程（建议值：CPU核心数 * 2 + 1）
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# 使用 Gunicorn（生产环境推荐）
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker ardc.api.main:app --bind 0.0.0.0:8000
```

### Nginx 优化

```nginx
# 在 http 块中添加
http {
    # 启用 gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    gzip_min_length 1024;
    
    # 连接池优化
    keepalive_timeout 65;
    keepalive_requests 100;
    
    # 缓冲区设置
    client_max_body_size 10M;
    client_body_buffer_size 128k;
}
```

---

## 监控与日志

### Prometheus 监控

访问 `http://your-domain.com/metrics` 查看指标：

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `ardc_skill_executions_total` | counter | 技能执行总次数 |
| `ardc_requests_total` | counter | API 请求总次数 |
| `ardc_uptime_seconds` | gauge | 服务运行时间（秒） |
| `ardc_skills_count` | gauge | 已注册技能数量 |

### 日志管理

```bash
# 查看 Nginx 访问日志
tail -f /var/log/nginx/access.log

# 查看 Nginx 错误日志
tail -f /var/log/nginx/error.log

# 查看应用日志（systemd）
sudo journalctl -u ardc-api -f

# 查看 Docker 日志
docker-compose logs -f backend
```

---

## 故障排查

### 服务无法启动

```bash
# 检查端口占用
netstat -tlnp | grep 8000
lsof -i:8000

# 检查 Python 环境
source .venv/bin/activate
python -c "import ardc; print('OK')"

# 检查依赖版本
pip list | grep -E "fastapi|uvicorn|pydantic"

# 直接运行查看错误
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000
```

### API 返回 502 Bad Gateway

**问题原因：** 后端服务未运行或端口未监听。

```bash
# 检查后端服务状态
netstat -tlnp | grep 8000

# 如果未运行，启动后端服务
cd /path/to/skillhub
source .venv/bin/activate
nohup uvicorn ardc.api.main:app --host 127.0.0.1 --port 8000 --workers 4 > /var/log/ardc.log 2>&1 &

# 验证服务启动
sleep 3
curl http://127.0.0.1:8000/api/health
```

### Nginx 重定向循环

**错误信息：**
```
rewrite or internal redirection cycle while internally redirecting to "/index.html"
```

**解决方案：** 使用 named location 避免循环（参考本文档 Nginx 配置部分）。

### 403 Forbidden 权限错误

**错误信息：**
```
stat() "/root/czq/anime_role_detect/skillhub/web/dist/" failed (13: Permission denied)
```

**解决方案：** 将前端文件移动到标准 Web 目录：

```bash
mkdir -p /var/www/ardc-web
cp -r /root/czq/anime_role_detect/skillhub/web/dist/* /var/www/ardc-web/
chown -R www-data:www-data /var/www/ardc-web
chmod -R 755 /var/www/ardc-web
```

### 前端页面显示空白

```bash
# 检查前端文件是否存在
ls -la /var/www/ardc-web/

# 检查 Nginx 配置
nginx -t

# 检查浏览器控制台错误
# 在浏览器中按 F12 查看 Console 标签
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

---

## 更新维护

### 更新代码

```bash
cd skillhub

# 拉取最新代码
git pull

# 更新后端依赖
source .venv/bin/activate
pip install -r requirements.txt

# 更新前端
cd web
npm install
npm run build

# 重启服务
sudo systemctl restart ardc-api
sudo systemctl reload nginx
```

### 版本升级

```bash
# 备份数据（重要！）
tar -czvf ardc_backup_$(date +%Y%m%d).tar.gz ~/.ardc/

# 切换到新版本
git checkout v2.0.0

# 更新依赖
pip install -r requirements.txt
npm install && npm run build

# 重启服务
sudo systemctl restart ardc-api
```

---

## 常见问题

**Q: 如何添加新技能？**

A: 使用 Web 界面或 CLI 命令注册技能：
```bash
ardc skill register \
  --id ardc-my-skill \
  --name "我的技能" \
  --version 1.0.0 \
  --author "your-name" \
  --category utility \
  --entry-point scripts/main.py
```

**Q: 如何发布新版本？**

A: 使用相同的技能 ID 和不同的版本号注册即可。系统会自动管理版本历史。

**Q: 如何卸载技能？**

A: 使用 CLI 命令或 Web 界面：
```bash
ardc skill uninstall ardc-my-skill
```

**Q: 监控指标在哪里查看？**

A: 访问 `http://your-domain.com/metrics`。

**Q: 如何配置域名？**

A: 在 Nginx 配置文件中修改 `server_name` 指令，并确保 DNS 解析正确指向服务器 IP。

**Q: 如何设置 HTTPS？**

A: 使用 Let's Encrypt 获取 SSL 证书：
```bash
sudo certbot --nginx -d your-domain.com
```

---

## 部署架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      客户端浏览器                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTPS
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Nginx (反向代理)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  / → 前端静态文件 (dist)      /api/ → Backend API      │   │
│  │  /metrics → Prometheus指标   /health → 健康检查        │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│   Backend API     │                     │    Prometheus     │
│   (FastAPI)       │◄────────────────────│   (监控采集)      │
└───────────────────┘                     └───────────────────┘
          │
          ▼ 文件系统
┌─────────────────────────────────────────────────────────────────┐
│                      数据存储 (~/.ardc/)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │ registry.json│ │skill_index.json│ │   skills/   │           │
│  │ (技能注册表)  │ │  (搜索索引)   │ │ (已安装技能) │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │  versions/  │ │ workflows/   │ │ favorites.json│           │
│  │ (版本历史)   │ │ (工作流定义) │ │  (收藏列表)   │           │
│  └──────────────┘ └──────────────┘ └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 服务管理命令速查

| 操作 | 命令 |
|------|------|
| 启动服务 | `sudo systemctl start ardc-api` |
| 停止服务 | `sudo systemctl stop ardc-api` |
| 重启服务 | `sudo systemctl restart ardc-api` |
| 查看状态 | `sudo systemctl status ardc-api` |
| 开机自启 | `sudo systemctl enable ardc-api` |
| 查看日志 | `sudo journalctl -u ardc-api -f` |

---

## 联系方式

如有问题，请提交 Issue 或联系维护人员。

**文档版本**: v1.0  
**最后更新**: 2026年5月