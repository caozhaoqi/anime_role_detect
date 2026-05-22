# 动漫角色分类模型训练计划

## 目录

1. [项目概述](#项目概述)
2. [当前训练进度](#当前训练进度)
3. [Phase 1：基础优化训练](#phase-1基础优化训练)
4. [Phase 2：添加注意力机制](#phase-2添加注意力机制)
5. [Phase 3：模型融合](#phase-3模型融合)
6. [训练资源配置](#训练资源配置)
7. [模型评估与验证](#模型评估与验证)
8. [风险评估与应对措施](#风险评估与应对措施)

---

## 1. 项目概述

### 1.1 项目背景
本项目旨在训练一个动漫角色分类模型，用于识别 76 个不同的动漫角色类别。

### 1.2 数据集概况
- **样本总数**：9,809 张图像
- **类别数量**：76 个角色类别
- **数据分布**：极度不平衡（最小样本数 3，最大样本数 211）

### 1.3 核心挑战
| 挑战 | 描述 | 影响 |
|------|------|------|
| 类别不平衡 | 部分类别样本极少 | 模型偏向于多数类 |
| 过拟合 | 训练准确率高，验证准确率低 | 泛化能力差 |
| 类别相似性 | 不同角色外观相似 | 容易混淆 |

---

## 2. 当前训练进度

### 2.1 实时训练状态

| 指标 | 当前值 | 目标值 |
|------|-------|--------|
| 当前 Epoch | 16/80 | 80 |
| 训练准确率 | 42.08% | 持续提升 |
| 验证准确率 | **52.70%** | 60-65% |
| 训练损失 | 2.9104 | 持续下降 |
| 验证损失 | 2.7395 | 持续下降 |

### 2.2 训练趋势分析

#### 2.2.1 性能提升曲线

| Epoch | 验证准确率 | 提升幅度 |
|-------|-----------|---------|
| 1 | 12.69% | - |
| 5 | 37.51% | +24.82% |
| 10 | 43.37% | +5.86% |
| 15 | **52.70%** | +9.33% |

#### 2.2.2 关键发现

✅ **余弦退火学习率调度器生效**
- Epoch 11 学习率重启后，验证准确率从 43.37% 跃升至 46.08%
- 说明模型成功跳出局部最优

✅ **过拟合控制优秀**
- 训练准确率（42.08%）< 验证准确率（52.70%）
- 数据增强和 Dropout 正则化效果显著

✅ **加权损失函数有效**
- 类别权重范围：0.61 ~ 43.02
- 少数类别识别能力提升

---

## 3. Phase 1：基础优化训练

### 3.1 当前配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型架构 | EfficientNet-B3 | 预训练权重 |
| Dropout 率 | 0.4 | 正则化防止过拟合 |
| 数据增强 | Heavy | 旋转/翻转/颜色抖动/高斯模糊 |
| 损失函数 | 加权交叉熵 + 标签平滑(0.1) | 解决类别不平衡 |
| 学习率 | 1e-4 (分类层) / 1e-5 (特征层) | 分层学习率 |
| 学习率调度 | 余弦退火 | T_0=10, T_mult=2 |
| 批次大小 | 32 | 平衡内存与梯度稳定性 |
| 早停耐心值 | 15 | 防止过拟合 |
| 优化器 | AdamW | 权重衰减 1e-4 |

### 3.2 预计完成时间

| 项目 | 估算 |
|------|------|
| 单 Epoch 耗时 | 10-12 分钟 |
| 剩余 Epoch | 64 |
| 总剩余时间 | 10-12 小时 |

### 3.3 预期结果

| 指标 | 预期值 |
|------|-------|
| 验证准确率 | 60-65% |
| 训练准确率 | 50-55% |
| 模型文件大小 | ~150MB |
| 模型保存位置 | `models/efficientnet_b3_loli_optimized_v2_*` |

---

## 4. Phase 2：添加注意力机制

### 4.1 实施时间
Phase 1 完成后立即开始

### 4.2 CBAM 模块设计

#### 4.2.1 通道注意力

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)
```

#### 4.2.2 空间注意力

```python
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_out, avg_out], dim=1)
        return x * self.sigmoid(self.conv(concat))
```

#### 4.2.3 CBAM 组合模块

```python
class CBAMModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
```

### 4.3 集成策略

```python
# 在 EfficientNet 的每个 MBConv 块后添加 CBAM
from torchvision.models.efficientnet import MBConv

model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

for i, block in enumerate(model.features):
    if isinstance(block, MBConv):
        model.features[i] = nn.Sequential(block, CBAMModule(block.out_channels))
```

### 4.4 预期收益

| 指标 | 预期提升 |
|------|---------|
| 验证准确率 | +3-5% |
| 模型大小 | +~5% |
| 推理速度 | -~10% |

---

## 5. Phase 3：模型融合

### 5.1 实施时间
Phase 2 完成后开始

### 5.2 EfficientNet + ViT 融合架构

#### 5.2.1 架构设计

```python
class EfficientViT(nn.Module):
    def __init__(self, num_classes=76):
        super().__init__()
        
        # EfficientNet 分支 - 提取局部特征
        self.effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.eff_features = nn.Sequential(*list(self.effnet.children())[:-2])
        
        # ViT 分支 - 提取全局特征
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()
        
        # 融合分类头
        self.fusion = nn.Sequential(
            nn.Linear(1536 + 768, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        # EfficientNet 特征
        eff_out = self.eff_features(x)
        eff_out = eff_out.flatten(2).mean(dim=2)  # [B, 1536]
        
        # ViT 特征
        vit_out = self.vit(x)  # [B, 768]
        
        # 融合
        combined = torch.cat([eff_out, vit_out], dim=1)
        return self.fusion(combined)
```

#### 5.2.2 融合策略对比

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| 特征级融合 | 拼接中间特征 | 信息保留完整 | 计算量大 |
| 决策级融合 | 加权融合预测结果 | 简单高效 | 信息丢失 |
| 注意力融合 | 动态加权特征 | 自适应学习 | 复杂度高 |

### 5.3 训练配置

| 参数 | 值 |
|------|-----|
| 初始学习率 | 5e-5 |
| 权重衰减 | 1e-4 |
| 批次大小 | 16 |
| Epoch 数 | 50 |
| 早停耐心值 | 10 |

### 5.4 预期收益

| 指标 | 预期值 |
|------|-------|
| 验证准确率 | 65%+ |
| 模型大小 | ~500MB |
| 推理速度 | ~10-15 FPS |

---

## 6. 训练资源配置

### 6.1 硬件要求

| 资源 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 3060 (12GB) | NVIDIA RTX 4090 (24GB) |
| CPU | Intel i7-8700 | Intel i9-12900K |
| 内存 | 32GB | 64GB |
| 存储 | 100GB 可用空间 | 200GB 可用空间 |

### 6.2 软件环境

| 软件 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| TorchVision | 0.15+ |
| NumPy | 1.24+ |
| Pillow | 9.0+ |

### 6.3 环境变量设置

```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPS_HIGH_WATERMARK_RATIO=0.0
```

---

## 7. 模型评估与验证

### 7.1 评估指标

| 指标 | 计算公式 | 说明 |
|------|---------|------|
| Top-1 准确率 | 正确预测数 / 总样本数 | 主要指标 |
| Top-5 准确率 | 正确预测在前 5 概率中的比例 | 辅助指标 |
| 混淆矩阵 | 各类别预测 vs 真实的矩阵 | 分析类别混淆 |
| F1 分数 | 2 * 精确率 * 召回率 / (精确率 + 召回率) | 不平衡数据评估 |
| 推理时间 | 单张图像推理耗时 | 性能指标 |

### 7.2 评估流程

```
1. 加载最佳模型
2. 运行验证集推理
3. 计算各项指标
4. 生成混淆矩阵
5. 分析错误案例
6. 生成评估报告
```

### 7.3 评估脚本

```bash
# 运行完整评估
python scripts/model_evaluation/test_all_models.py

# 分析准确率
python scripts/model_evaluation/accuracy_analysis.py
```

---

## 8. 风险评估与应对措施

### 8.1 风险清单

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| GPU 内存不足 | 中 | 训练中断 | 降低批次大小，启用混合精度 |
| 过拟合 | 低 | 泛化能力差 | 增加 Dropout，增强数据增强 |
| 训练发散 | 低 | 模型崩溃 | 降低学习率，检查数据预处理 |
| 验证准确率停滞 | 中 | 无法达到目标 | 调整学习率调度，尝试新架构 |
| 硬件故障 | 低 | 训练中断 | 定期保存检查点，使用云服务 |

### 8.2 检查点策略

```python
# 每 5 个 Epoch 保存检查点
if (epoch + 1) % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'train_history': train_history
    }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')
```

---

## 附录

### A. 训练命令汇总

```bash
# Phase 1: 基础优化训练
python scripts/model_training/train_loli_optimized_v2.py

# Phase 2: 添加 CBAM 注意力
python scripts/model_training/train_with_attention.py

# Phase 3: 模型融合
python scripts/model_training/train_ensemble.py

# 评估模型
python scripts/model_evaluation/test_all_models.py
```

### B. 文件结构

```
anime_role_detect/
├── models/                 # 模型保存目录
├── checkpoints/           # 检查点保存目录
├── logs/                  # 训练日志
├── data/
│   └── expanded_dataset/  # 数据集
├── scripts/
│   ├── model_training/    # 训练脚本
│   └── model_evaluation/  # 评估脚本
└── TRAINING_PLAN.md       # 训练计划文档
```

### C. 参考资料

1. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
2. CBAM: Convolutional Block Attention Module
3. Vision Transformer (ViT)
4. Cosine Annealing Learning Rate Scheduler

---

**文档版本**: v1.0  
**创建时间**: 2026-05-22  
**最后更新**: 2026-05-22