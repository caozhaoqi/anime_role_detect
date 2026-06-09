# ARD Skill Hub

技能仓库管理系统 - 为 ARD 角色分类系统提供可扩展的技能管理能力。

## ✨ 功能特性

- **技能注册**: 注册和管理自定义技能
- **版本管理**: 支持技能多版本发布
- **分类浏览**: 按类别浏览技能
- **一键安装**: 简单的技能安装流程
- **CLI 工具**: 命令行管理工具
- **RESTful API**: 完整的 API 接口

## 🚀 快速开始

### 安装 CLI

```bash
# 安装 ARD CLI
bash sh/install.sh

# 验证安装
ardc --help
```

### 启动服务

```bash
# 使用 systemd
sudo systemctl start ardc-api

# 或手动启动
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000
```

### API 访问

- **API 地址**: `http://localhost:8000/api/`
- **文档地址**: `http://localhost:8000/docs`

## 🌐 CLI 命令

```bash
# 技能管理
ardc skill list              # 列出所有技能
ardc skill install <id>      # 安装技能
ardc skill uninstall <id>    # 卸载技能
ardc skill register          # 注册新技能
ardc skill update <id>       # 更新技能

# 系统管理
ardc system status           # 检查系统状态
ardc system upgrade          # 升级系统
ardc system clean            # 清理缓存

# 搜索
ardc search <keyword>        # 搜索技能
```

## 📁 项目结构

```
skillhub/
├── ardc/                    # ARD CLI 核心代码
│   ├── api/                 # RESTful API
│   ├── cli/                 # 命令行接口
│   └── core/                # 核心模块
├── web/                     # 前端界面
├── conf/                    # 配置文件
├── sh/                      # Shell 脚本
└── docs/                    # 文档
```

## 📊 API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/skills` | GET | 获取技能列表 |
| `/api/skills/{id}` | GET | 获取技能详情 |
| `/api/skills` | POST | 注册技能 |
| `/api/skills/{id}/install` | POST | 安装技能 |
| `/api/skills/{id}/uninstall` | POST | 卸载技能 |
| `/api/health` | GET | 健康检查 |

## 🔧 配置

环境变量配置：

```bash
export JWT_SECRET_KEY="your-secret-key"
export ALLOWED_ORIGINS="http://localhost:3000"
export ARD_C_DATA_DIR="~/.ardc"
```

## 📚 文档

详细文档：
- `docs/DEPLOYMENT.md` - 部署指南
- `docs/API.md` - API 文档
- `docs/CLI.md` - CLI 使用手册

## 📄 许可证

MIT License

---

**版本**: v1.0 | **最后更新**: 2026年5月