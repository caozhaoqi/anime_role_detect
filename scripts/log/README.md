# 日志管理工具集

本目录包含用于管理和分析系统日志的工具脚本。

## 工具列表

### 1. reorganize_logs.sh - 日志重构脚本

**功能：** 将现有日志文件重新组织到新的目录结构中

**用法：**
```bash
./scripts/logs/reorganize_logs.sh
```

**执行步骤：**
1. 自动备份当前所有日志
2. 创建新的目录结构（按服务分类）
3. 移动日志文件到对应位置
4. 显示新结构和使用说明

**注意：** 首次使用前请阅读 `docs/logging_system_design.md`

---

### 2. archive_logs.py - 日志归档脚本

**功能：** 定期将旧日志压缩归档，节省磁盘空间

**用法：**
```bash
# 归档7天前的日志
python scripts/logs/archive_logs.py --archive-days 7

# 仅预览（不实际执行）
python scripts/logs/archive_logs.py --dry-run --archive-days 7

# 自定义参数
python scripts/logs/archive_logs.py \
    --archive-days 7 \
    --retention-days 90 \
    --log-dir logs
```

**参数说明：**
- `--archive-days`: 归档多少天前的日志（默认：7）
- `--retention-days`: 归档保留天数（默认：180）
- `--log-dir`: 日志根目录（默认：logs）
- `--archive-dir`: 归档目录（默认：logs/archive）
- `--dry-run`: 仅显示将要归档的文件，不实际执行

**建议：** 设置每周日凌晨2点自动执行

---

### 3. analyze_logs.py - 日志分析工具

**功能：** 提供多种日志查询和分析功能

**用法：**

#### 显示统计信息
```bash
# 所有服务的统计
python scripts/logs/analyze_logs.py --stats

# 特定服务的统计
python scripts/logs/analyze_logs.py --stats --service api-service
```

#### 搜索日志
```bash
# 搜索关键词
python scripts/logs/analyze_logs.py --search "timeout"

# 在特定服务中搜索
python scripts/logs/analyze_logs.py --search "error" \
    --services api-service,model-service

# 搜索最近1小时的错误
python scripts/logs/analyze_logs.py --search "error" \
    --level ERROR --since 1h

# 限制显示条数
python scripts/logs/analyze_logs.py --search "timeout" --limit 100
```

**参数说明：**
- `--stats`: 显示统计信息模式
- `--search KEYWORD`: 搜索关键词模式
- `--service SERVICE`: 指定单个服务
- `--services SVC1,SVC2`: 指定多个服务（逗号分隔）
- `--level LEVEL`: 日志级别过滤（ERROR, WARNING, INFO等）
- `--since TIME`: 起始时间（如：1h, 30m, 2d）
- `--limit N`: 显示条数限制（默认：50）
- `--log-dir DIR`: 日志目录（默认：logs）

---

## 目录结构

使用这些工具后，日志目录将变为：

```
logs/
├── services/              # 按服务分类
│   ├── api-service/
│   ├── model-service/
│   ├── api-gateway/
│   ├── multimedia-service/
│   ├── search-service/
│   ├── inference-worker/
│   ├── frontend/
│   └── monitoring/
├── functional/            # 按功能分类
│   ├── health_check/
│   ├── inference/
│   ├── training/
│   ├── system/
│   ├── download/
│   └── error/
├── archive/               # 归档目录
│   └── compressed/
├── backup_*/              # 备份目录（临时）
├── supervisord.log
├── unified.log
└── redis.log
```

---

## 快速开始

### 首次使用

1. **阅读文档**
   ```bash
   cat docs/logging_system_design.md
   ```

2. **执行重构**
   ```bash
   ./scripts/logs/reorganize_logs.sh
   ```

3. **重启Supervisor**
   ```bash
   supervisorctl -c supervisord.conf reload
   ```

4. **验证新结构**
   ```bash
   ls -la logs/services/
   tail -f logs/services/api-service/api-service.log
   ```

### 日常使用

```bash
# 查看实时日志
tail -f logs/services/api-service/api-service.log

# 搜索最近的错误
python scripts/logs/analyze_logs.py --search "error" --level ERROR --since 1h

# 每周归档
python scripts/logs/archive_logs.py --archive-days 7
```

---

## 常见问题

### Q: 重构后找不到旧日志？
A: 所有日志都备份到了 `logs/backup_YYYYMMDD_HHMMSS/` 目录

### Q: 如何恢复旧结构？
A: 从备份目录复制回去即可：
```bash
cp -r logs/backup_*/* logs/
```

### Q: 归档会删除原文件吗？
A: 是的，归档后会删除原始日志文件，但保留在压缩归档中

### Q: 如何查看归档的日志？
A: 使用zcat或zgrep：
```bash
zcat logs/archive/compressed/.../*.log.gz | less
zgrep "timeout" logs/archive/compressed/.../*.log.gz
```

---

## 相关文档

- [日志系统设计文档](../../docs/logging_system_design.md)
- [Supervisor配置](../../supervisord.conf)

---

## 维护者

如有问题或建议，请联系项目维护者。
