# ardc-collector - 数据采集技能

## 版本
1.0.0

## 功能说明

提供动漫/游戏角色图片的批量采集能力。

## 安装

```bash
ardc-skill-sync install ardc-collector
```

## 使用示例

### 批量采集图片

```bash
python ~/.ardc/skills/ardc-collector/scripts/collect_images.py \
    --role "Arona" \
    --count 100 \
    --output ./data/arona
```

### 爬取 URL

```bash
python ~/.ardc/skills/ardc-collector/scripts/crawl_urls.py \
    --keyword "blue archive arona" \
    --source "danbooru" \
    --output ./urls/arona_urls.txt
```

## 依赖技能

- ardc-client (必须)

## API 参考

| 脚本 | 功能 | 参数 |
|------|------|------|
| `collect_images.py` | 批量采集 | `--role`, `--count`, `--output` |
| `crawl_urls.py` | URL 爬取 | `--keyword`, `--source`, `--output` |
| `download_batch.py` | 批量下载 | `--url-file`, `--output`, `--concurrent` |
