# Scripts Directory

## 📁 目录结构

```
scripts/
├── utils/                          # 🧰 项目工具库
│   └── __init__.py
│
├── data_cleaning/                  # 🧹 数据清洗
│   ├── utils.py                    # 清洗工具库
│   ├── cleaner.py                  # 统一清洗工具
│   ├── merge_datasets.py           # 数据集合并
│   ├── analyze_final.py            # 最终数据集分析
│   ├── README.md                   # 文档
│   └── archived/                   # 📦 旧脚本归档
│
├── data_collection/                # 📥 数据收集
│   ├── utils/                      # 收集工具
│   ├── batch_collectors/           # 批量收集
│   ├── downloaders/                # 下载器
│   └── ...
│
├── analysis/                       # 📊 数据分析
├── classification/                 # 🎯 分类
├── detection/                      # 🔍 检测
├── model_evaluation/               # 📈 模型评估
├── model_training/                 # 🤖 模型训练
├── api/                            # 📡 API
├── services/                       # 🛠️ 服务
└── ...
```

## 🧰 工具库使用

### 项目通用工具

```python
from scripts.utils import (
    calculate_md5,
    scan_images,
    find_duplicate_files,
    delete_files,
    check_image_quality,
    save_json,
    load_json
)
```

### 数据清洗专用工具

```python
from scripts.data_cleaning.utils import (
    # 与上面相同，但包含数据清洗专用函数
)
```

## 📋 已整合的功能

### ✅ 去重功能统一

| 位置 | 旧脚本 | 新方案 |
|------|---------|---------|
| `data_cleaning/` | 16个去重脚本 | `cleaner.py` |
| `data_collection/utils/` | `deduplicate_images.py` | `utils.py` |
| `analysis/` | `merge_datasets.py` | `data_cleaning/merge_datasets.py` |

### ✅ 质量检查统一

| 旧脚本 | 新方案 |
|---------|---------|
| `check_data_quality*.py` | `utils.check_image_quality()` |
| `check_corrupted_images*.py` | `utils.batch_quality_check()` |
| `data_quality_evaluator.py` | `utils.batch_quality_check()` |

## 🚀 推荐工作流

### 1. 数据清洗

```bash
cd scripts/data_cleaning

# 试运行
python3 cleaner.py --data-dir ../../data/final_dataset

# 执行清理
python3 cleaner.py --data-dir ../../data/final_dataset --auto-delete --output-report report.json
```

### 2. 查看文档

```bash
cat scripts/data_cleaning/README.md
```

## 📦 归档的脚本

所有重复、临时、废弃的脚本已整理到：
- `scripts/data_cleaning/archived/` - 数据清洗旧脚本
- 其他目录中的旧脚本可考虑后续整合

## 📝 注意事项

1. 优先使用 `scripts/utils/` 中的工具
2. 数据清洗优先使用 `data_cleaning/cleaner.py`
3. 旧脚本仅用于特殊情况参考
4. 新增功能优先整合到工具库
