# Contributing to Anime Role Detect

欢迎贡献代码！我们非常欢迎任何形式的贡献，包括但不限于：

- 🐛 Bug 报告
- ✨ 功能请求
- 📝 文档改进
- 💻 代码提交
- 🧪 测试用例
- 🎨 UI/UX 改进

## 贡献流程

### 1. Fork 仓库

点击页面右上角的 "Fork" 按钮创建你的分支仓库。

### 2. 克隆仓库

```bash
git clone https://github.com/caozhaoqi/anime_role_detect.git
cd anime_role_detect
```

### 3. 创建开发分支

```bash
git checkout -b feature/your-feature-name
```

分支命名规范：
- `feature/xxx` - 新功能
- `fix/xxx` - Bug 修复
- `docs/xxx` - 文档更新
- `refactor/xxx` - 代码重构

### 4. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装基础依赖
pip install -r requirements-base.txt

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装 ML 依赖（可选，用于模型训练）
pip install -r requirements-ml.txt
```

### 5. 配置 pre-commit

```bash
pre-commit install
```

这将在每次提交前自动运行代码检查。

### 6. 编写代码

请遵循以下规范：

- **代码风格**: 使用 `black` 和 `isort` 进行格式化
- **类型提示**: 为所有函数和方法添加类型注解
- **测试覆盖**: 为新功能编写单元测试
- **文档**: 为公共 API 添加 docstring

### 7. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_api.py -v

# 运行测试并生成覆盖率报告
pytest tests/ -v --cov=src --cov=skillhub
```

### 8. 提交代码

```bash
git add .
git commit -m "feat: add new feature"
```

提交信息规范（使用 Conventional Commits）：

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 代码重构 |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖更新 |

### 9. 推送分支

```bash
git push origin feature/your-feature-name
```

### 10. 创建 Pull Request

在 GitHub 上创建 Pull Request，描述你的更改内容。

## 代码审查

所有 Pull Request 都需要至少一位维护者审查通过才能合并。

## 行为准则

请遵守以下行为准则：

- 尊重他人，友好沟通
- 使用中文或英文进行交流
- 保持代码仓库整洁
- 不提交无意义的更改

## 问题反馈

如果你遇到问题或有建议，请创建 Issue：

1. 检查是否已有类似 Issue
2. 提供清晰的问题描述
3. 包含重现步骤
4. 附上相关日志或截图

## 开发环境

### 推荐工具

- **编辑器**: VS Code / PyCharm
- **Python 版本**: 3.9+
- **包管理**: pip / poetry

### 环境变量

复制 `.env.example` 为 `.env` 并根据需要修改配置。

## 许可证

所有贡献将遵循 MIT 许可证。

---

感谢你的贡献！🎉
