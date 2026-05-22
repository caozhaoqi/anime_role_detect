# 数据清洗工具

## 📁 目录结构

```
scripts/data_cleaning/
├── utils.py                    # 🧰 工具库 (核心)
├── cleaner.py                  # 🧹 统一清理工具
├── merge_datasets.py           # 🔗 数据集合并
├── analyze_final.py            # 📊 最终数据集分析
├── analyze_dataset.py          # 📊 通用数据集分析
├── archived/                   # 📦 已归档的旧脚本
│   ├── find_duplicates.py
│   ├── remove_duplicates.py
│   └── ...
└── cascades/                   # 💡 OpenCV级联分类器
```

## 🚀 快速开始

### 1. 统一清理（推荐）

```bash
# 试运行（不删除）
python3 cleaner.py --data-dir data/final_dataset

# 执行清理并自动删除
python3 cleaner.py --data-dir data/final_dataset --auto-delete --output-report report.json

# 只去重
python3 cleaner.py --data-dir data/final_dataset --skip-size --skip-quality
```

### 2. 合并数据集

```bash
python3 merge_datasets.py --src1 data/dataset1 --src2 data/dataset2 --dst data/final
```

### 3. 分析数据集

```bash
python3 analyze_final.py
```

## 🧰 工具库 (`utils.py`)

### 文件扫描

```python
import utils

# 扫描所有图片
all_images = utils.scan_images("data/final_dataset")

# 按角色扫描
role_images = utils.scan_role_images("data/final_dataset")
```

### MD5计算

```python
# 单个文件
file_hash = utils.calculate_md5("image.jpg")

# 批量计算
hash_map = utils.batch_calculate_md5(all_images)
```

### 去重

```python
# 查找重复
duplicates = utils.find_duplicate_files(all_images)

# 确定删除策略
to_delete = utils.get_deletion_candidates(duplicates)

# 执行删除
success, failed = utils.delete_files(to_delete)
```

### 质量检查

```python
# 单个图片
ok, reason = utils.check_image_quality("image.jpg")

# 批量检查
passed, failed = utils.batch_quality_check(all_images)
```

## 📊 整理说明

### 已归档的脚本

以下脚本已被整合到 `utils.py` 和 `cleaner.py` 中：

| 旧脚本 | 替代方案 |
|---------|---------|
| `find_duplicates.py` | `utils.find_duplicate_files()` |
| `remove_duplicates.py` | `cleaner.py --skip-size --skip-quality` |
| `clean_duplicates.py` | `cleaner.py` |
| `deduplicate_and_normalize.py` | `cleaner.py` |
| `quick_filter.py` | `cleaner.py` |
| `check_data_quality.py` | `utils.check_image_quality()` |
| `check_corrupted_images.py` | `utils.batch_quality_check()` |
| `data_cleaner.py` | `cleaner.py` |

## 🔧 配置

修改 `utils.py` 中的常量：

```python
MIN_FILE_SIZE_KB = 10      # 最小文件大小
MIN_IMAGE_WIDTH = 100      # 最小宽度
MIN_IMAGE_HEIGHT = 100     # 最小高度
```

## 📝 使用示例

### 完整清理流程

```python
import utils

# 1. 扫描
all_files = utils.scan_images("data/dataset")

# 2. 质量检查
passed, failed = utils.batch_quality_check(all_files)

# 3. 去重
duplicates = utils.find_duplicate_files(passed)
to_delete = utils.get_deletion_candidates(duplicates)

# 4. 删除
success, failed = utils.delete_files(to_delete)
```
