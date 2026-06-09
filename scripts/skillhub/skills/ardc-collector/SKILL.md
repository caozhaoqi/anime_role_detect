# ARDC Collector - 数据采集技能

## 概述

ARD 数据采集技能，用于从各种数据源采集动漫图片数据。

## 技能信息

| 属性 | 值 |
|------|------|
| ID | ardc-collector |
| 名称 | 数据采集器 |
| 版本 | 1.0.0 |
| 作者 | ARD Team |
| 分类 | collector |
| 状态 | stable |

## 功能特性

- 支持从 URL 批量下载图片
- 自动去重，避免重复下载
- 支持从 API 获取图片列表
- 提供下载统计信息

## 入口文件

```
scripts/collect_images.py
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output_dir | string | data/images | 输出目录 |
| delay | float | 0.5 | 下载间隔（秒） |

## 使用示例

```python
from ardc_collector import ImageCollector

collector = ImageCollector(output_dir="data/images")
collector.download_batch([
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
])
```

## 依赖

- requests >= 2.28.0

## 更新日志

### v1.0.0
- 初始版本
- 支持 URL 批量下载
- 支持 API 采集
- 自动去重功能
