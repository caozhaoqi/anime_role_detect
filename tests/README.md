# 测试目录

本目录包含项目的所有测试脚本，按功能分类存储。

## 目录结构

```
tests/
├── api/                    # API测试
│   └── test_api_accuracy.py
├── benchmark/              # 性能基准测试
│   ├── benchmark_model.py
│   ├── benchmark_models.py
│   └── compare_models.py
├── data/                   # 数据采集和准备测试
│   ├── batch_spider_roles.py
│   ├── collect_all_characters.py
│   ├── collect_test_data.py
│   ├── download_images.py
│   ├── download_images_from_urls.py
│   └── prepare_test_data.py
├── docs/                   # 测试文档和报告
│   ├── benchmark_report.md
│   ├── BILIBILI_VIDEO_COLLECTION_GUIDE.md
│   ├── VIDEO_DETECTION_TEST_REPORT.md
│   ├── test_report.md
│   ├── test_report.txt
│   ├── training_report.pdf
│   ├── dataset_distribution.png
│   ├── detailed_results.json
│   ├── insufficient_characters.png
│   ├── log_analysis_report.txt
│   ├── model_accuracy_results_short.json
│   ├── role_distribution.png
│   ├── role_index.faiss
│   ├── role_index_mapping.json
│   ├── similarity_distribution.png
│   └── test_midterm_optimization_results.json
├── evaluation/              # 模型评估测试
│   ├── evaluate_system.py
│   ├── simple_evaluate.py
│   ├── test_all_inference_modes.py
│   ├── test_all_models.py
│   ├── test_model_accuracy.py
│   └── test_model_performance.py
├── img/                    # 测试图片
│   └── 微信图片_20260204115846_481_347.jpg
├── integration/             # 集成测试
│   ├── test_api_integration.py
│   └── test_data_integration.py
├── legacy/                 # 旧版测试脚本（已废弃）
│   ├── basic_verify.py
│   ├── final_verify.py
│   ├── test_classification_fix.py
│   ├── test_config.py
│   ├── test_data_collection.py
│   ├── test_frontend.html
│   ├── test_frontend.js
│   ├── test_generator.py
│   └── test_performance.py
├── model/                  # 模型测试
│   ├── check_classifier_weights.py
│   ├── check_model.py
│   ├── check_model_classes.py
│   ├── check_models.py
│   ├── comprehensive_model_test.py
│   ├── quick_test_models.py
│   ├── test_class_mapping.py
│   ├── test_coreml_model.py
│   ├── test_coreml_performance.py
│   ├── test_ensemble_method.py
│   ├── test_individual_models.py
│   ├── test_infinity_fix.py
│   ├── test_midterm_optimization.py
│   ├── test_model_fix.py
│   ├── test_model_on_collected_data.py
│   ├── test_optimization.py
│   ├── test_processing.py
│   ├── test_single_model.py
│   ├── test_weight_loading.py
│   └── test_wuthering_waves.py
├── unit/                   # 单元测试
│   ├── test_classification.py
│   ├── test_data_collection.py
│   ├── test_data_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_model_management.py
├── workflow/                # 工作流测试
│   ├── test_workflow.py
│   ├── test_workflow_results.json
│   └── verify_pipeline.py
├── README.md              # 本文档
└── __init__.py
```

## 使用方法

### 1. 单元测试

```bash
# 运行所有单元测试
python -m pytest tests/unit/

# 运行特定单元测试
python -m pytest tests/unit/test_classification.py
```

### 2. 模型测试

```bash
# 测试单个模型
python tests/model/test_single_model.py --model models/checkpoints/model.pth

# 测试所有模型
python tests/model/test_all_models.py

# 测试模型权重
python tests/model/check_classifier_weights.py
```

### 3. 评估测试

```bash
# 评估系统性能
python tests/evaluation/evaluate_system.py

# 评估模型准确率
python tests/evaluation/test_model_accuracy.py --model models/checkpoints/model.pth

# 评估所有推理模式
python tests/evaluation/test_all_inference_modes.py
```

### 4. 基准测试

```bash
# 运行性能基准测试
python tests/benchmark/benchmark_models.py

# 比较模型性能
python tests/benchmark/compare_models.py
```

### 5. API测试

```bash
# 测试API准确率
python tests/api/test_api_accuracy.py

# 测试API集成
python tests/integration/test_api_integration.py
```

### 6. 数据采集测试

```bash
# 采集测试数据
python tests/data/collect_test_data.py --character "arona" --limit 10

# 批量采集角色
python tests/data/collect_all_characters.py

# 准备测试数据
python tests/data/prepare_test_data.py
```

### 7. 工作流测试

```bash
# 测试完整工作流
python tests/workflow/test_workflow.py

# 验证处理管道
python tests/workflow/verify_pipeline.py
```

## 测试分类说明

### 单元测试 (unit/)
- **test_classification.py**: 分类功能单元测试
- **test_data_collection.py**: 数据采集单元测试
- **test_data_preprocessing.py**: 数据预处理单元测试
- **test_feature_extraction.py**: 特征提取单元测试
- **test_model_management.py**: 模型管理单元测试

### 模型测试 (model/)
- **check_classifier_weights.py**: 检查分类器权重
- **check_model.py**: 检查模型文件
- **check_model_classes.py**: 检查模型类别
- **check_models.py**: 检查所有模型
- **test_single_model.py**: 测试单个模型
- **test_individual_models.py**: 测试独立模型
- **test_all_models.py**: 测试所有模型
- **test_model_accuracy.py**: 测试模型准确率
- **test_model_performance.py**: 测试模型性能
- **test_model_on_collected_data.py**: 在采集数据上测试模型
- **test_model_fix.py**: 测试模型修复
- **test_optimization.py**: 测试优化
- **test_processing.py**: 测试处理
- **test_weight_loading.py**: 测试权重加载
- **test_class_mapping.py**: 测试类别映射
- **test_coreml_model.py**: 测试CoreML模型
- **test_coreml_performance.py**: 测试CoreML性能
- **test_ensemble_method.py**: 测试集成方法
- **test_infinity_fix.py**: 测试无穷大修复
- **test_midterm_optimization.py**: 测试中期优化
- **test_wuthering_waves.py**: 测试呜呼波
- **comprehensive_model_test.py**: 综合模型测试
- **quick_test_models.py**: 快速模型测试

### 评估测试 (evaluation/)
- **evaluate_system.py**: 评估系统
- **simple_evaluate.py**: 简单评估
- **test_all_inference_modes.py**: 测试所有推理模式
- **test_all_models.py**: 测试所有模型
- **test_model_accuracy.py**: 测试模型准确率
- **test_model_performance.py**: 测试模型性能

### 基准测试 (benchmark/)
- **benchmark_model.py**: 基准测试模型
- **benchmark_models.py**: 基准测试所有模型
- **compare_models.py**: 比较模型

### API测试 (api/)
- **test_api_accuracy.py**: 测试API准确率

### 数据测试 (data/)
- **batch_spider_roles.py**: 批量爬取角色
- **collect_all_characters.py**: 采集所有角色
- **collect_test_data.py**: 采集测试数据
- **download_images.py**: 下载图片
- **download_images_from_urls.py**: 从URL下载图片
- **prepare_test_data.py**: 准备测试数据

### 集成测试 (integration/)
- **test_api_integration.py**: API集成测试
- **test_data_integration.py**: 数据集成测试

### 工作流测试 (workflow/)
- **test_workflow.py**: 测试工作流
- **verify_pipeline.py**: 验证管道
- **test_workflow_results.json**: 工作流测试结果

### 旧版测试 (legacy/)
- 包含已废弃的测试脚本，仅供参考

### 文档 (docs/)
- 包含测试报告、基准报告等文档

## 最佳实践

1. **单元测试优先**: 在修改代码前先运行单元测试
2. **模型验证**: 训练后使用模型测试验证模型
3. **性能评估**: 定期运行基准测试监控性能
4. **集成测试**: 确保各模块正常集成
5. **工作流验证**: 测试完整的端到端工作流

## 注意事项

1. **测试隔离**: 每个测试应该独立运行
2. **清理资源**: 测试后清理临时文件
3. **断言清晰**: 使用清晰的断言信息
4. **覆盖率**: 尽量提高测试覆盖率
5. **文档更新**: 修改测试后更新相关文档

## 扩展指南

如需添加新的测试：

1. 在相应目录下创建新的测试文件
2. 遵循现有的测试模式
3. 更新本README文档
4. 确保测试可以独立运行
