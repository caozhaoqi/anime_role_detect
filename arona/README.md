# Arona/Plana 专用模块

本模块包含针对蔚蓝档案阿罗娜和普拉娜角色的专用数据采集、训练、评估和预测功能。

## 目录结构

```
arona/
├── collection/          # 数据采集脚本
│   ├── collect_arona_plana.py      # 阿罗娜和普拉娜专用采集脚本
│   └── simple_collector.py         # 简单的角色数据采集脚本
├── training/            # 模型训练脚本
│   ├── train_arona_plana.py        # 阿罗娜和普拉娜专用训练脚本
│   ├── train_with_attributes.py    # 带有属性标签的角色分类模型训练
│   ├── train_generator.py          # 基于检测模型的图片生成模型训练
│   └── conditional_generator.py    # 条件生成模型训练
├── evaluation/          # 模型评估脚本
│   ├── evaluate_classification.py   # 评估分类模型性能
│   ├── test_classification_model.py # 测试分类模型效果
│   └── test_with_attributes.py     # 带有属性标签的角色分类模型测试
├── prediction/          # 预测和生成脚本
│   ├── predict_with_attributes.py   # 带有属性标签的角色分类推理
│   ├── generate_characters.py       # 角色图片生成脚本
│   ├── generate_conditional.py     # 使用条件生成模型创建角色图片
│   └── generate_with_model.py      # 使用训练好的生成模型创建角色图片
├── tagging/             # 标签处理脚本
│   ├── wd_vit_v3_tagger.py         # 使用WD Vit V3 Tagger为采集的数据打标签
│   ├── create_attribute_annotations.py  # 为训练数据创建属性标注
│   └── integrate_wd_tags.py        # 集成WD Vit V3 Tagger生成的标签到现有属性标注系统
├── models/              # 模型定义
│   └── models.py                  # 模型定义文件
├── utils/               # 工具脚本
│   └── deploy_wd_vit_v3.py        # 下载并部署WD Vit V3 Tagger模型
├── config/              # 配置文件
│   ├── attribute_annotations.json  # 属性标注文件
│   └── character_attributes.json   # 角色属性配置文件
├── docs/                # 文档
│   └── 数据整合与模型更新报告.md  # 数据整合与模型更新报告
└── README.md            # 本文件
```

## 功能说明

### 数据采集 (collection/)

#### collect_arona_plana.py
专门为阿罗娜和普拉娜两个角色优化的数据采集脚本，支持从Danbooru和Safebooru采集图像。

**使用方法:**
```bash
cd arona/collection
python collect_arona_plana.py
```

**特性:**
- 支持多个搜索标签
- 图像验证和尺寸检查
- 自动去重
- 随机延迟避免被封禁

#### simple_collector.py
为低准确率角色补充采集图像数据的简单脚本。

**使用方法:**
```bash
cd arona/collection
python simple_collector.py --target-count 60
```

### 模型训练 (training/)

#### train_arona_plana.py
阿罗娜和普拉娜专用训练脚本，针对这两个角色进行模型优化训练。

**使用方法:**
```bash
cd arona/training
python train_arona_plana.py --data-dir ../data/downloaded_images --epochs 50 --batch-size 32
```

**参数:**
- `--data-dir`: 数据目录
- `--model-type`: 模型类型 (mobilenet_v2, efficientnet_b0, resnet18)
- `--batch-size`: 批量大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--output-dir`: 输出目录

#### train_with_attributes.py
带有属性标签的角色分类模型训练脚本，可以同时预测角色和属性。

**使用方法:**
```bash
cd arona/training
python train_with_attributes.py --data-dir ../../data/downloaded_images --annotations-file ../config/attribute_annotations.json
```

#### train_generator.py
基于检测模型的图片生成模型训练脚本，使用扩散模型架构。

**使用方法:**
```bash
cd arona/training
python train_generator.py --data-dir data/train --epochs 50
```

#### conditional_generator.py
条件生成模型训练脚本，能够根据类别标签生成对应角色的图像。

**使用方法:**
```bash
cd arona/training
python conditional_generator.py --data-dir data/train --epochs 100
```

### 模型评估 (evaluation/)

#### evaluate_classification.py
评估分类模型性能的脚本。

**使用方法:**
```bash
cd arona/evaluation
python evaluate_classification.py --model-path ../models/arona_plana/model_best.pth --data-dir ../data/train
```

#### test_classification_model.py
测试分类模型效果的脚本，支持单张图像测试和批量测试。

**使用方法:**
```bash
# 批量测试
cd arona/evaluation
python test_classification_model.py --model-path ../models/arona_plana/model_best.pth --data-dir ../data/train

# 单张图像测试
python test_classification_model.py --model-path ../models/arona_plana/model_best.pth --test-image path/to/image.jpg
```

#### test_with_attributes.py
带有属性标签的角色分类模型测试脚本。

**使用方法:**
```bash
cd arona/evaluation
python test_with_attributes.py --model-path ../models/arona_plana_with_attributes/model_best.pth --annotations-file ../config/attribute_annotations.json
```

### 预测和生成 (prediction/)

#### predict_with_attributes.py
带有属性标签的角色分类推理脚本。

**使用方法:**
```bash
cd arona/prediction
python predict_with_attributes.py --model-path ../models/arona_plana_with_attributes/model_best.pth --image path/to/image.jpg
```

#### generate_characters.py
角色图片生成脚本，使用Trae API生成图片。

**使用方法:**
```bash
cd arona/prediction
python generate_characters.py --character arona --num-images 5
```

#### generate_conditional.py
使用条件生成模型创建角色图片。

**使用方法:**
```bash
cd arona/prediction
python generate_conditional.py --model-path ../models/conditional_gan/model_final.pth --num-images-per-class 5
```

#### generate_with_model.py
使用训练好的生成模型创建角色图片。

**使用方法:**
```bash
cd arona/prediction
python generate_with_model.py --model-path ../models/generator/generator_best.pth --num-images 5
```

### 标签处理 (tagging/)

#### wd_vit_v3_tagger.py
使用WD Vit V3 Tagger为采集的数据打标签。

**使用方法:**
```bash
cd arona/tagging
python wd_vit_v3_tagger.py --input-dir ../data/downloaded_images --output-dir ../data/image_tags
```

#### create_attribute_annotations.py
为训练数据创建属性标注。

**使用方法:**
```bash
cd arona/tagging
python create_attribute_annotations.py --data-dir ../../data/downloaded_images --output-file ../config/attribute_annotations.json
```

#### integrate_wd_tags.py
集成WD Vit V3 Tagger生成的标签到现有属性标注系统。

**使用方法:**
```bash
cd arona/tagging
python integrate_wd_tags.py --annotations ../config/attribute_annotations.json --tags-dir ../../data/image_tags
```

### 模型定义 (models/)

#### models.py
包含所有模型的定义，包括:
- `CharacterAttributeModel`: 带有属性预测分支的角色分类模型
- `get_model_with_attributes`: 获取带有属性预测分支的模型的辅助函数

### 工具脚本 (utils/)

#### deploy_wd_vit_v3.py
下载并部署WD Vit V3 Tagger模型。

**使用方法:**
```bash
cd arona/utils
python deploy_wd_vit_v3.py --model-id SmilingWolf/wd-vit-v3 --model-dir ../models/wd-vit-v3
```

## 数据流程

1. **数据采集**: 使用 `collection/` 中的脚本从各种数据源采集图像
2. **标签生成**: 使用 `tagging/wd_vit_v3_tagger.py` 为图像生成标签
3. **属性标注**: 使用 `tagging/create_attribute_annotations.py` 创建属性标注
4. **模型训练**: 使用 `training/` 中的脚本训练模型
5. **模型评估**: 使用 `evaluation/` 中的脚本评估模型性能
6. **预测生成**: 使用 `prediction/` 中的脚本进行预测和图像生成

## 依赖项

- torch
- torchvision
- transformers
- PIL
- requests
- scikit-learn
- tqdm
- numpy

## 配置文件

需要准备以下配置文件:

- `config/character_attributes.json`: 角色属性定义文件
- `config/attribute_annotations.json`: 属性标注文件（通过 `tagging/create_attribute_annotations.py` 生成）

## 注意事项

1. 确保数据目录结构正确
2. 训练前检查数据集大小和类别分布
3. 使用适当的模型参数和超参数
4. 定期评估模型性能
5. 保存训练好的模型和评估结果
