# 数据库切换至腾讯云 TencentDB — 变更说明

## 背景
项目 `anime_role_detect` 原 MySQL 主库指向阿里云 RDS（`.env` 中 `MYSQL_HOST=czq.rwlb.rds.aliyuncs.com`），当前 `DATABASE_MODE=dual`，因 MySQL 未运行走 SQLite 降级。用户希望将主库切到其独立的腾讯云数据库（CDB）。

## 改动文件
- **`.env`**
  - 删除原阿里云 RDS 配置段
  - 删除用户误追加的重复腾讯云段（导致 `MYSQL_*` 重复、`DATABASE_MODE` 仍为 `dual`）
  - 写入腾讯云连接：`MYSQL_HOST=bj-cdb-3om1u3c8.sql.tencentcdb.com`、`MYSQL_PORT=29394`、`MYSQL_USER=root`、`MYSQL_DB=anime_role_detect`
  - `DATABASE_MODE` 由 `dual` 改为 `remote`
  - 保留 `ecs` 配置段（`111.231.75.125`，与数据库无关，供其他脚本使用）
- **`.env.rds`**（阿里云 + `RDS_*` 错误变量名，代码从不读取）→ 改名 `.env.rds.bak`
- **`src/core/config/database.py`**：**无需改动**。代码原生支持远程 MySQL，由 `MYSQL_*`/`MYSQL_URL` 环境变量驱动；`init_remote_database()` 会 `SELECT 1` 探测，失败自动降级回 SQLite
- 校验：`py_compile database.py` → `COMPILE_OK`

## 用户侧前置条件（需确认）
- 腾讯云控制台已创建库 `anime_role_detect`（`utf8mb4`）—— 项目只建表不建库（代码无 `CREATE DATABASE`）
- `root` 账号 `host = %`（允许外网）
- 安全组放通本机公网 IP 到端口 `29394`

## 验证方式（用户本机）
- 启动项目后观察日志 `远程MySQL数据库初始化`
- 或：`.venv/bin/python3 -c "from src.core.config import database; database.init_remote_database(); print(database.is_remote_connected())"` → 期望 `True`
- 连不上时自动降级回 SQLite，不报错但数据不进云

## 风险提醒
- 数据库 `root` 密码与阿里云 ECS 的 SSH 密码**相同**，且已在本对话明文出现 → 建议更换为独立密码
- `.env` 含明文密码，确认不提交 git
