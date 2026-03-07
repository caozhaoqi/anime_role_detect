# 数据采集脚本

本目录包含按数据源分类的数据采集脚本。

## 目录结构

```
collection/
├── safebooru/              # Safebooru数据采集器
│   └── collector.py
├── pixiv/                  # Pixiv数据采集器
│   └── collector.py
├── danbooru/               # Danbooru数据采集器
│   └── collector.py
├── bilibili/                # B站数据采集器
│   ├── collector.py         # 图片采集器
│   └── video_collector.py  # 视频采集器
│   └── download_anime_videos.py  # 动漫视频下载
├── wiki/                   # Wiki数据采集器
│   ├── collector.py         # 图片采集器
│   ├── characters_fetcher.py  # 角色列表获取
│   └── real_characters_fetcher.py  # 真实角色获取
├── bangdream/              # BangDream专用采集器
│   ├── crawl_bangdream_characters.py
│   └── crawl_bangdream_images.py
├── blue_archive/           # 蔚蓝档案专用采集器
│   ├── collector.py        # 标准采集器
│   ├── legacy_collector.py  # 旧版采集器
│   └── optimized_collector.py  # 优化版采集器
├── general/                # 通用采集器
│   ├── keyword_based_collector.py  # 基于关键词的采集
│   ├── series_based_collector.py   # 基于系列的采集
│   ├── data_collection.py          # 通用数据采集
│   ├── data_collection_fast.py     # 快速数据采集
│   └── character_name_collector.py # 角色名称采集
├── legacy/                 # 旧版采集器（已废弃）
│   ├── supplement_collector.py
│   └── text_description_collector.py
├── universal_collector.py   # 通用数据采集器（整合多个数据源）
└── README.md              # 本文档
```

## 使用方法

### 1. Safebooru采集器

```bash
python src/data/collection/safebooru/collector.py \
    --tag "arona_(blue_archive)" \
    --output_dir data/train/arona \
    --limit 20
```

### 2. Pixiv采集器

```bash
python src/data/collection/pixiv/collector.py \
    --tag "arona" \
    --output_dir data/train/arona \
    --limit 20 \
    --refresh_token YOUR_REFRESH_TOKEN
```

### 3. Danbooru采集器

```bash
python src/data/collection/danbooru/collector.py \
    --tag "arona_(blue_archive)" \
    --output_dir data/train/arona \
    --limit 20 \
    --api_key YOUR_API_KEY \
    --user YOUR_USERNAME
```

### 4. B站采集器

#### 图片采集
```bash
python src/data/collection/bilibili/collector.py \
    --keyword "蔚蓝档案 阿罗娜" \
    --output_dir data/train/arona \
    --limit 20
```

#### 视频采集
```bash
python src/data/collection/bilibili/video_collector.py \
    --url "https://www.bilibili.com/video/xxx" \
    --output_dir data/videos
```

### 5. Wiki采集器

#### 图片采集
```bash
python src/data/collection/wiki/collector.py \
    --wiki_url "https://bluearchive.fandom.com/wiki/Arona" \
    --output_dir data/train/arona \
    --limit 20
```

#### 角色列表获取
```bash
python src/data/collection/wiki/characters_fetcher.py \
    --series "genshin_impact"
```

### 6. 蔚蓝档案专用采集器

#### 标准采集器
```bash
python src/data/collection/blue_archive/collector.py \
    --output_dir data/train \
    --limit 20
```

#### 优化版采集器
```bash
python src/data/collection/blue_archive/optimized_collector.py \
    --output_dir data/train
```

### 7. BangDream专用采集器

```bash
python src/data/collection/bangdream/crawl_bangdream_images.py \
    --character "mygo" \
    --series "bangdream" \
    --output_dir data/train
```

### 8. 通用采集器（推荐）

```bash
python src/data/collection/universal_collector.py \
    --character "arona_(blue_archive)" \
    --output_dir data/train/arona \
    --limit 20 \
    --sources safebooru danbooru pixiv
```

### 9. 基于关键词的采集

```bash
python src/data/collection/general/keyword_based_collector.py \
    --keyword_file "auto_spider_img/blda_spider_img_keyword.txt" \
    --output_dir data/train
```

### 10. 基于系列的采集

```bash
python src/data/collection/general/series_based_collector.py \
    --mode priority
```

## 数据源说明

### Safebooru
- **优点**: 无需API密钥，内容安全，质量较高
- **缺点**: 图片数量相对较少
- **适用场景**: 快速采集少量高质量图片

### Pixiv
- **优点**: 图片数量多，质量高，更新快
- **缺点**: 需要refresh_token，有速率限制
- **适用场景**: 批量采集大量图片

### Danbooru
- **优点**: 标签系统完善，搜索精准
- **缺点**: 需要API密钥才能访问更多内容
- **适用场景**: 精准搜索特定标签

### B站
- **优点**: 视频封面图，内容丰富
- **缺点**: 图片尺寸较小，质量一般
- **适用场景**: 补充数据源

### Wiki
- **优点**: 官方图片，信息准确
- **缺点**: 图片数量有限
- **适用场景**: 获取官方立绘

### BangDream
- **优点**: 专门针对BangDream系列
- **缺点**: 仅适用于特定系列
- **适用场景**: BangDream角色数据采集

### 蔚蓝档案
- **优点**: 专门针对蔚蓝档案系列
- **缺点**: 仅适用于特定系列
- **适用场景**: 蔚蓝档案角色数据采集

## 最佳实践

1. **优先使用Safebooru**: 无需配置，快速开始
2. **配置Pixiv**: 获取refresh_token后，可批量采集更多数据
3. **多源采集**: 使用通用采集器，自动尝试多个数据源
4. **质量控制**: 采集后检查图片质量，过滤低质量图片
5. **数据增强**: 使用数据增强脚本扩充数据集
6. **专用采集器**: 对于特定系列（如蔚蓝档案、BangDream），使用专用采集器

## 注意事项

1. **遵守网站规则**: 不要过度请求，避免被封禁
2. **尊重版权**: 仅用于学习和研究目的
3. **内容过滤**: 避免采集不适宜内容
4. **存储空间**: 注意磁盘空间，及时清理临时文件
5. **API限制**: 注意各平台的API速率限制
6. **数据质量**: 采集后检查图片质量，删除重复或损坏的图片

## 旧版脚本

`legacy/` 目录包含旧版的采集脚本，这些脚本已被新的分类采集器替代，仅供参考。

## 扩展指南

如需添加新的数据源采集器：

1. 在相应目录下创建新的采集器
2. 遵循现有的接口规范
3. 更新本README文档
4. 在`universal_collector.py`中集成新数据源
