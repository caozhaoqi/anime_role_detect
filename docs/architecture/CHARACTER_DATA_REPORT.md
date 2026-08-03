# 角色数据现状分析报告

> 日期：2026-08-02  
> 数据来源：项目文件系统 + SQLite 数据库 + JSON 统计报告

---

## 一、数据总览

| 指标 | 数值 |
|------|------|
| 角色列表文件记录数 | 251 条（含大量脏数据） |
| training_dataset 角色数 | 32（实际 31 个有图片，1 个空目录） |
| training_dataset 总图片数 | 2,401 |
| final_dataset 角色数 | 157（实际 108 个有图片，49 个空目录） |
| final_dataset 总图片数 | 3,049 |
| 数据库文件数 | 7 个 SQLite（共约 5.4MB） |
| 识别记录数 | 2 条（几乎无使用） |

---

## 二、数据集详情

### 2.1 training_dataset（训练集）

> 32 个角色目录，1 个空目录（marisa_kirisame），实有 31 个角色，共 2,401 张图片。

| 角色 | 图片数 | 状态 |
|------|--------|------|
| hatsune_miku | 168 | ⚠️ 超出 150 上限 |
| columbina | 137 | 🟡 接近上限 |
| belle | 110 | ✅ |
| rita | 106 | ✅ |
| qiqi | 101 | ✅ |
| komeiji_koishi | 99 | ✅ |
| vivian | 98 | ✅ |
| qingyi | 96 | ✅ |
| griseo | 96 | ✅ |
| march_7th | 94 | ✅ |
| guizhong | 88 | ✅ |
| chevreuse | 83 | ✅ |
| mualani | 81 | ✅ |
| acheron | 81 | ✅ |
| nefer | 76 | ✅ |
| sara | 74 | ✅ |
| xilonen | 73 | ✅ |
| mavuika | 73 | ✅ |
| Sparkle | 69 | ✅ |
| pela | 67 | ✅ |
| barbara | 59 | ✅ |
| Sigewinne | 50 | ✅ |
| lynette | 49 | ✅ |
| fu_xuan | 48 | ✅ |
| ruan_mei | 47 | ✅ |
| huohuo | 47 | ✅ |
| dehya | 47 | ✅ |
| cloud_retainer | 47 | ✅ |
| asta | 46 | ✅ |
| Arona | 46 | ✅ |
| jingliu | 45 | ✅ |
| marisa_kirisame | 0 | ❌ 空目录 |

**问题**：
- hatsune_miku（168 张）超出 150 张上限，需裁剪
- columbina（137 张）接近上限，需关注
- marisa_kirisame 目录为空，需采集或删除

---

### 2.2 final_dataset（最终数据集）

> 157 个角色目录，仅 108 个有图片，49 个为空目录，共 3,049 张图片。

**图片数量分布**：

| 区间 | 角色数 | 占比 |
|------|--------|------|
| 50-100 张 | 5 | 3.2% |
| 30-50 张 | 38 | 24.2% |
| 10-30 张 | 52 | 33.1% |
| 1-10 张 | 13 | 8.3% |
| 0 张 | 49 | 31.2% |

**Top 20 角色**：

| 角色 | 图片数 | 系列 |
|------|--------|------|
| asuna | 82 | Blue Archive |
| Aru | 73 | Blue Archive |
| aglaea | 66 | Blue Archive |
| Firefly | 62 | Honkai: Star Rail |
| Furina | 56 | Genshin Impact |
| sandrone | 52 | Genshin Impact |
| collei | 51 | Genshin Impact |
| ako | 50 | Blue Archive |
| yoimiya | 48 | Genshin Impact |
| dori | 48 | Genshin Impact |
| topaz | 47 | Honkai: Star Rail |
| kafka | 47 | Honkai: Star Rail |
| mona | 46 | Genshin Impact |
| xiangling | 45 | Genshin Impact |
| lucia | 45 | Punishing: Gray Raven |
| Kachina | 45 | Genshin Impact |
| charlotte | 45 | Genshin Impact |
| Kirara | 44 | Genshin Impact |
| clorinde | 44 | Genshin Impact |
| tingyun | 43 | Honkai: Star Rail |

**49 个空目录（无图片）**：

```
akari, alice, aloy, ayaka, beidou, bronya, candace, chinatsu, chiori,
constance, corin, dahli, dubra, eden, eula, frieren, ganyu, haruki,
himeko, icarus, iori, katrina, kazusa, koishi, kuki_shinobu, lyney,
marisa_kirisame, mikoto, navia, pardofelis, platelet, raiden_shogun,
raven, reimu, sayu, seele, shenhe, shigure_kira, sleeper, sucrose,
tama, umaru, urume, xianyun, xinyan, yae_sakura, yelan, yuki, yuko
```

**严重不足（<10 张）的角色**：

| 角色 | 图片数 |
|------|--------|
| amber | 4 |
| hu_tao | 4 |
| pelagia | 4 |
| yuuka | 3 |
| aris | 2 |
| ellen | 2 |
| fern | 2 |
| houshou_marine | 2 |
| jade | 2 |
| lynx | 2 |
| rem | 8 |
| lisa | 8 |
| theresa | 5 |
| ram | 6 |
| 其他 13 个角色 | 1 |

---

## 三、数据库文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `data/collection.db` | 4.5MB | 采集任务记录 |
| `data/recognition.db` | 108KB | 识别记录 |
| `data/image_hashes.db` | 632KB | 图片哈希去重 |
| `data/data_pipeline.db` | 72KB | 数据流水线（characters/samples/annotations） |
| `data/auth.db` | 16KB | 用户认证 |
| `data/packages.db` | 24KB | 数据包管理 |
| `recognition.db` | 64KB | 旧版识别记录 |

**数据库表结构**（`data_pipeline.db`）：

| 表名 | 说明 |
|------|------|
| `characters` | 角色信息（名称、系列、别名、搜索词） |
| `samples` | 样本图片（路径、质量评分、动漫/AI 判定、单人检测、属性标签） |
| `collection_tasks` | 采集任务（角色、搜索词、状态、采集数量） |
| `deduplication_records` | 去重记录（相似度、方法） |
| `annotations` | 标注记录（bbox、角色名、置信度） |

---

## 四、角色列表文件问题

### 4.1 `all_characters_complete.txt`

- 声称 251 个角色，但文件严重损坏
- 大量垃圾字符（`·`、`Λ`、`•`、`-`、数字 `3`、`TOP`）
- 混合中英文拼音（如 `Kōng Zhī Lǜzhě`、`Shǐyuán Zhī Lǜzhě`）
- 角色名重复出现（如 Qiqi 出现在多个区块）
- 区块标题混乱（`# (251)`、`# (76)`、`# TOP (29)` 等）

### 4.2 `all_characters_formatted.txt`

- 与 `all_characters_complete.txt` 内容相同，同样损坏
- 中文名居多，英文名不统一（PascalCase vs snake_case vs 拼音）

### 4.3 命名不一致

同一角色在项目中存在多种命名方式：

| 角色 | 变体 |
|------|------|
| 胡桃 | `hu_tao`, `Hu Tao`, `hu tao` |
| 刻晴 | `keqing`, `Keqing`, `ke qing` |
| 可莉 | `Klee`, `klee` |
| 七七 | `Qiqi`, `qiqi`, `Qi Qi` |
| 三月七 | `march_7th`, `March 7th` |
| 符玄 | `fu_xuan`, `Fu Xuan`, `Fuhua` |

---

## 五、数据质量问题

### 5.1 图片格式分布

| 格式 | final_dataset | training_dataset |
|------|:---:|:---:|
| JPG | 75.0% | 74.6% |
| PNG | 23.5% | 18.2% |
| JPEG | 1.3% | 7.2% |
| WebP | 0.07% | — |

### 5.2 图片大小分布

| 区间 | final_dataset | training_dataset |
|------|:---:|:---:|
| >1MB | 51.3% | 44.9% |
| 100-500KB | 24.6% | 32.8% |
| 500KB-1MB | 22.9% | 20.6% |
| 50-100KB | 0.9% | 1.2% |
| <50KB | 0.3% | 0.6% |

### 5.3 主要问题

1. **角色列表文件损坏** — 无法正常解析，需重建
2. **49 个空目录** — 占 final_dataset 的 31.2%，大量角色从未采集到数据
3. **命名不一致** — 同一角色有 snake_case / PascalCase / 中文拼音多种写法，影响去重和匹配
4. **训练集与最终集角色重叠** — 部分角色同时出现在两个数据集中（如 Sparkle、Sigewinne）
5. **数量上限违规** — hatsune_miku 训练集 168 张，超出 150 上限
6. **识别记录几乎为空** — 仅 2 条记录，系统使用率低或记录未持久化
7. **旧爬虫数据残留** — `archived/auto_spider_img/`、`spider_image_system/` 等目录仍有数据

---

## 六、建议修复方案

### 6.1 立即修复（P0）

| 任务 | 说明 |
|------|------|
| 裁剪 hatsune_miku | 从 168 张随机删减至 150 张 |
| 清理 49 个空目录 | 删除无图片的角色目录，或在角色列表中标记为"待采集" |
| 修复角色列表文件 | 清理 `all_characters_complete.txt` 中的垃圾字符，统一为英文名 |

### 6.2 短期优化（P1）

| 任务 | 说明 |
|------|------|
| 统一角色命名规范 | 全项目统一为 snake_case 英文名（如 `hu_tao`、`march_7th`） |
| 去重训练集与最终集 | 检查两个数据集的重叠角色，合并或去重 |
| 补充低数据角色 | 针对 <10 张图片的 13 个角色启动采集任务 |
| 数据库内容同步 | 将文件系统中的角色数据同步到 `data_pipeline.db` 的 `characters` 表 |

### 6.3 中期优化（P2）

| 任务 | 说明 |
|------|------|
| 角色别名系统 | 在 `characters` 表中建立 aliases 字段，支持多名称匹配 |
| 数据质量自动检查 | CI 中加入数据质量检查脚本（数量上限、空目录、格式检查） |
| 旧数据清理 | 清理 `archived/` 和 `spider_image_system/` 中的残留数据 |

---

## 七、数据统计汇总

```
项目角色数据总览
├── 角色列表文件: 251 条（损坏严重）
├── 训练集: 32 目录 / 31 有效 / 2,401 图片
│   ├── 正常范围(30-150): 30 个角色
│   ├── 超出上限: 1 个（hatsune_miku 168）
│   └── 空目录: 1 个（marisa_kirisame）
├── 最终集: 157 目录 / 108 有效 / 3,049 图片
│   ├── 充足(50+): 5 个角色
│   ├── 良好(30-50): 38 个角色
│   ├── 一般(10-30): 52 个角色
│   ├── 不足(<10): 13 个角色
│   └── 空目录: 49 个角色
├── 数据库: 7 个 SQLite 文件 / 5.4MB
└── 识别记录: 仅 2 条
```