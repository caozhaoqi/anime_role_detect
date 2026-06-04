# 数据流水线优化开发计划

**分支**: `feature/data-pipeline-optimization`
**创建时间**: 2026-06-04
**优先级**: 高

---

## 📋 项目概述

基于 `collect_todo.md` 的分析，本项目将实现完整的数据采集、清洗、标注流水线，重点优化数据质量，提升模型性能。

---

## 🎯 核心目标

| 目标 | 指标 | 优先级 |
|------|------|--------|
| CLIP自动去重 | 相似度>0.98自动过滤 | P0 |
| 自动预标注 | 标注效率提升5-10倍 | P0 |
| 主动学习 | 困难样本自动回流 | P0 |
| 数据质量过滤 | 噪声率<5% | P1 |
| 多维标签系统 | 支持10+属性标签 | P1 |

---

## 📅 开发阶段

### 阶段一：基础架构搭建（Week 1）

#### 1.1 项目结构设计
```
src/data_pipeline/
├── collector/          # 数据采集
│   ├── search_engine.py
│   ├── image_downloader.py
│   └── deduplication.py
├── cleaner/            # 数据清洗
│   ├── quality_filter.py
│   ├── anime_classifier.py
│   └── ai_detector.py
├── annotator/          # 自动标注
│   ├── yolo_detector.py
│   ├── clip_tagger.py
│   └── pre_annotation.py
├── active_learning/    # 主动学习
│   ├── confidence_filter.py
│   └── sample_review.py
└── database/           # 数据管理
    ├── character_db.py
    └── sample_db.py
```

#### 1.2 数据库设计
```sql
-- 角色表
CREATE TABLE characters (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    series VARCHAR(100),
    aliases JSON,
    created_at TIMESTAMP
);

-- 样本表
CREATE TABLE samples (
    id INTEGER PRIMARY KEY,
    image_path VARCHAR(500),
    character_id INTEGER,
    quality_score FLOAT,
    is_anime BOOLEAN,
    is_ai_generated BOOLEAN,
    person_count INTEGER,
    bbox_area_ratio FLOAT,
    attributes JSON,
    confidence FLOAT,
    is_difficult BOOLEAN,
    status VARCHAR(20),  -- pending, reviewed, rejected
    created_at TIMESTAMP
);

-- 采集任务表
CREATE TABLE collection_tasks (
    id INTEGER PRIMARY KEY,
    character_id INTEGER,
    search_terms JSON,
    max_samples INTEGER,
    status VARCHAR(20),
    created_at TIMESTAMP
);
```

#### 1.3 配置文件
```yaml
# config/data_pipeline.yaml
collector:
  max_per_role: 1000
  search_engines: [bing, google, baidu]
  download_timeout: 30

cleaner:
  min_width: 256
  min_height: 256
  min_quality_score: 0.7
  min_bbox_ratio: 0.1
  max_person_count: 1

deduplication:
  phash_threshold: 5
  clip_similarity_threshold: 0.98

annotation:
  yolo_model: yolov8n.pt
  clip_model: ViT-B/32
  auto_tagging: true

active_learning:
  confidence_threshold: 0.7
  review_batch_size: 100
```

---

### 阶段二：数据采集优化（Week 2）

#### 2.1 搜索引擎采集器
**文件**: `src/data_pipeline/collector/search_engine.py`

```python
class SearchEngineCollector:
    def __init__(self, engine: str):
        self.engine = engine

    def search(self, query: str, max_results: int = 100) -> List[str]:
        """搜索图片URL"""
        pass

    def download_images(self, urls: List[str], output_dir: str) -> List[str]:
        """下载图片"""
        pass
```

#### 2.2 角色别名映射
**文件**: `data/character_aliases.json`

```json
{
  "saber": {
    "id": 1001,
    "name": "Saber",
    "series": "Fate",
    "aliases": ["阿尔托莉雅", "Artoria", "Altria", "セイバー"]
  },
  "rem": {
    "id": 1002,
    "name": "Rem",
    "series": "Re:Zero",
    "aliases": ["雷姆", "蕾姆", "レム"]
  }
}
```

#### 2.3 采集数量控制
- 每个角色最多采集 1000 张
- 避免热门角色无限增长
- 冷门角色补充策略

---

### 阶段三：数据清洗优化（Week 3-4）

#### 3.1 CLIP去重系统
**文件**: `src/data_pipeline/collector/deduplication.py`

```python
class CLIPDeduplicator:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model = clip.load(model_name)[0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def compute_embeddings(self, images: List[str]) -> np.ndarray:
        """计算图片向量"""
        pass

    def deduplicate(self, embeddings: np.ndarray, threshold: float = 0.98) -> List[int]:
        """去重，返回保留的索引"""
        pass
```

#### 3.2 动漫/非动漫分类器
**文件**: `src/data_pipeline/cleaner/anime_classifier.py`

```python
class AnimeClassifier:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def classify(self, image_path: str) -> Tuple[bool, float]:
        """判断是否为动漫图片"""
        pass
```

#### 3.3 AI图片检测器
**文件**: `src/data_pipeline/cleaner/ai_detector.py`

```python
class AIDetector:
    def __init__(self):
        # 使用预训练的AI检测模型
        self.model = self.load_model()

    def detect(self, image_path: str) -> Tuple[bool, float]:
        """检测是否为AI生成图片"""
        pass
```

#### 3.4 图片质量评估
**文件**: `src/data_pipeline/cleaner/quality_filter.py`

```python
class QualityFilter:
    def assess_quality(self, image_path: str) -> Dict:
        """评估图片质量"""
        return {
            "width": width,
            "height": height,
            "sharpness": sharpness,
            "noise_level": noise_level,
            "overall_score": score
        }
```

#### 3.5 角色检测与裁剪
**文件**: `src/data_pipeline/cleaner/crop_detector.py`

```python
class CharacterCropper:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_and_crop(self, image_path: str) -> List[Image]:
        """检测角色并裁剪"""
        pass
```

---

### 阶段四：自动标注系统（Week 5-6）

#### 4.1 YOLO角色检测
**文件**: `src/data_pipeline/annotator/yolo_detector.py`

```python
class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path: str) -> List[Dict]:
        """检测角色"""
        return [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id
            }
        ]
```

#### 4.2 CLIP标签生成
**文件**: `src/data_pipeline/annotator/clip_tagger.py`

```python
class CLIPTagger:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model, self.preprocess = clip.load(model_name)

    def generate_tags(self, image_path: str, candidate_tags: List[str]) -> Dict:
        """生成标签"""
        pass
```

#### 4.3 多维标签系统
**文件**: `src/data_pipeline/annotator/multi_tagger.py`

```python
class MultiTagger:
    def generate_attributes(self, image_path: str) -> Dict:
        """生成多维属性标签"""
        return {
            "hair_color": "blue",
            "eye_color": "blue",
            "outfit": "maid",
            "pose": "standing",
            "expression": "smile",
            "background": "simple",
            "occlusion": false,
            "chibi": false,
            "back_view": false
        }
```

---

### 阶段五：主动学习系统（Week 7-8）

#### 5.1 困难样本识别
**文件**: `src/data_pipeline/active_learning/confidence_filter.py`

```python
class ConfidenceFilter:
    def filter_low_confidence(self, predictions: List[Dict], threshold: float = 0.7) -> List[int]:
        """筛选低置信度样本"""
        pass
```

#### 5.2 样本审核界面
**文件**: `src/data_pipeline/active_learning/review_ui.py`

```python
class ReviewUI:
    def launch(self, samples: List[Dict]) -> None:
        """启动审核界面"""
        pass
```

#### 5.3 增量训练流程
**文件**: `src/data_pipeline/active_learning/incremental_training.py`

```python
class IncrementalTrainer:
    def train_with_new_samples(self, new_samples: List[Dict]) -> str:
        """使用新样本增量训练"""
        pass
```

---

### 阶段六：流水线集成（Week 9-10）

#### 6.1 主流水线编排
**文件**: `src/data_pipeline/pipeline.py`

```python
class DataPipeline:
    def run(self, character_name: str) -> None:
        """运行完整流水线"""
        # 1. 采集
        images = self.collect(character_name)

        # 2. 去重
        unique_images = self.deduplicate(images)

        # 3. 清洗
        clean_images = self.clean(unique_images)

        # 4. 自动标注
        annotated = self.annotate(clean_images)

        # 5. 人工审核
        reviewed = self.review(annotated)

        # 6. 入库
        self.save_to_db(reviewed)
```

#### 6.2 CLI工具
**文件**: `src/data_pipeline/cli.py`

```bash
# 采集数据
python -m data_pipeline.cli collect --character "Rem" --max 1000

# 清洗数据
python -m data_pipeline.cli clean --input raw/ --output clean/

# 自动标注
python -m data_pipeline.cli annotate --input clean/ --output annotated/

# 启动审核
python -m data_pipeline.cli review --threshold 0.7

# 运行完整流水线
python -m data_pipeline.cli pipeline --character "Rem"
```

#### 6.3 Web界面
**文件**: `src/data_pipeline/web/app.py`

```python
# Flask/FastAPI Web界面
# 功能：
# - 查看采集进度
# - 审核样本
# - 查看数据统计
# - 启动训练
```

---

## 📊 依赖库清单

```txt
# 新增依赖
torch>=2.0.0
torchvision>=0.15.0
clip-by-openai>=1.0
imagehash>=4.3.0
opencv-python>=4.8.0
ultralytics>=8.0.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
sqlalchemy>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
streamlit>=1.25.0  # Web界面
```

---

## 🧪 测试计划

### 单元测试
```bash
tests/data_pipeline/
├── test_collector.py
├── test_cleaner.py
├── test_annotator.py
└── test_active_learning.py
```

### 集成测试
- 完整流水线测试
- 性能测试（处理速度）
- 准确率测试（去重、分类）

---

## 📈 成功指标

| 指标 | 目标值 | 当前值 |
|------|--------|--------|
| 去重准确率 | >95% | - |
| 动漫分类准确率 | >90% | - |
| AI检测准确率 | >85% | - |
| 标注效率提升 | 5-10倍 | - |
| 数据质量评分 | >0.8 | - |
| 模型准确率提升 | +5% | 93.92% |

---

## 📝 待办事项清单

### P0 - 必须完成
- [ ] 创建项目结构
- [ ] 设计数据库Schema
- [ ] 实现CLIP去重系统
- [ ] 实现YOLO自动标注
- [ ] 实现主动学习流程
- [ ] 集成完整流水线

### P1 - 重要功能
- [ ] 实现搜索引擎采集器
- [ ] 实现动漫/非动漫分类器
- [ ] 实现AI图片检测器
- [ ] 实现多维标签系统
- [ ] 开发CLI工具
- [ ] 开发Web界面

### P2 - 优化项
- [ ] 性能优化（并行处理）
- [ ] 缓存机制
- [ ] 错误处理与重试
- [ ] 日志系统
- [ ] 监控指标

---

## 🚀 下一步行动

1. **立即开始**: 创建项目结构
2. **本周完成**: 数据库设计 + CLIP去重原型
3. **下周目标**: YOLO自动标注 + 主动学习框架

---

**负责人**: ARD Team
**预计完成时间**: 2026-08-04（10周）
**当前状态**: 规划阶段 ✅