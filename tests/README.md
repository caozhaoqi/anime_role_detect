# 测试数据准备指南

本指南介绍如何准备测试数据，用于评估系统的性能和准确率。

## 测试数据结构

测试数据目录结构如下：

```
tests/
├── test_images/
│   ├── single_character/  # 单角色测试图片
│   │   ├── 角色1/         # 角色1的测试图片
│   │   ├── 角色2/         # 角色2的测试图片
│   │   └── ...
│   └── multiple_characters/  # 多角色测试图片
├── evaluate_system.py    # 系统评估脚本
├── prepare_test_data.py  # 测试数据准备脚本
└── README.md             # 测试数据准备指南
```

## 准备单角色测试数据

### 方法1：从现有数据集复制

如果您已经有了训练数据集，可以使用 `prepare_test_data.py` 脚本从数据集中复制测试数据：

```bash
# 从 dataset 目录准备测试数据
python tests/prepare_test_data.py --dataset_dir dataset --test_dir tests/test_images/single_character
```

### 方法2：手动收集

1. 在 `tests/test_images/single_character/` 目录下为每个角色创建一个子目录
2. 在每个角色子目录中放入该角色的测试图片（建议每个角色至少10张图片）

## 准备多角色测试数据

多角色测试数据需要包含多个角色的图片。您可以：

1. 从网络搜索包含多个动漫角色的图片
2. 使用图像处理软件将多个单角色图片合成一张多角色图片
3. 确保图片清晰，角色之间有明显的区分

将准备好的多角色图片放入 `tests/test_images/multiple_characters/` 目录。

## 运行测试

使用 `evaluate_system.py` 脚本运行测试：

```bash
# 运行完整测试
python tests/evaluate_system.py

# 自定义参数运行测试
python tests/evaluate_system.py \
    --single_character_dir tests/test_images/single_character \
    --multiple_character_dir tests/test_images/multiple_characters \
    --index_path role_index \
    --threshold 0.7
```

## 查看测试结果

测试完成后，评估报告将生成在 `tests/test_results/evaluation_report.txt` 文件中，包含以下内容：

- 单角色识别的准确率、无法识别率和平均处理时间
- 多角色识别的平均每张图片识别角色数和平均处理时间
- 系统配置信息

## 测试建议

1. **测试数据多样性**：确保测试数据包含不同角度、不同场景、不同风格的角色图片
2. **测试数据数量**：建议每个角色至少10张测试图片，多角色测试至少10张图片
3. **阈值调整**：根据测试结果调整 `threshold` 参数，平衡准确率和召回率
4. **性能测试**：在不同硬件配置下测试系统性能，评估处理速度

## 示例测试数据

如果您没有现成的测试数据，可以参考以下方法获取：

1. **使用公开数据集**：如 Danbooru、Konachan 等动漫图片数据集
2. **网络搜索**：使用搜索引擎搜索动漫角色图片
3. **动漫截图**：从动漫视频中截取角色图片

## 注意事项

1. 测试数据应与训练数据分开，避免数据泄露
2. 确保测试数据的版权合规，仅用于个人研究和测试
3. 多角色测试数据需要确保角色之间有足够的区分度，便于系统检测和识别
