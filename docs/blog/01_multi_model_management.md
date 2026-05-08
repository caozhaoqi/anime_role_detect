# 【技术难点】多模型集成与性能优化

> 在动漫角色检测系统中，如何高效管理多个深度学习模型是一个核心挑战。本文详细介绍了我们的解决方案。

---

## 🔍 问题背景

系统支持4种深度学习模型：

| 模型 | 准确率 | 推理速度 | 内存占用 |
|------|-------|---------|---------|
| MobileNetV2 | 94.00% | 379.34 FPS | 低 |
| EfficientNet-B0 | 95.20% | 298.45 FPS | 中 |
| EfficientNet-B3 | 96.80% | 187.60 FPS | 高 |
| ResNet50 | 94.80% | 256.78 FPS | 中 |

**核心挑战**：同时加载多个模型会导致内存溢出，尤其是在资源有限的服务器上。

---

## 💡 解决方案：动态模型加载

### Singleton 模式

确保全局只有一个模型管理器实例：

```python
class ModelManager:
    _instance = None
    _loaded_models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 按需加载策略

```python
import gc
import torch

def load_model(self, model_name):
    """按需加载模型"""
    # 如果模型已加载，直接返回
    if model_name in self._loaded_models:
        return self._loaded_models[model_name]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 根据模型名称加载对应的模型
    if model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=65)
        model.load_state_dict(torch.load('models/mobilenetv2.pth', map_location=device))
    
    elif model_name == 'efficientnet-b3':
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=65)
        model.load_state_dict(torch.load('models/efficientnet-b3.pth', map_location=device))
    
    model.eval().to(device)
    self._loaded_models[model_name] = model
    
    # 清理其他模型节省内存（关键优化）
    for name in list(self._loaded_models.keys()):
        if name != model_name:
            # 1. 删除模型引用
            del self._loaded_models[name]
    
    # 2. 显式调用垃圾回收
    gc.collect()
    
    # 3. 如果使用 CUDA，清理显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model
```

---

## 🚀 使用示例

```python
# 获取模型管理器实例
manager = ModelManager()

# 加载 EfficientNet-B3 模型
model = manager.load_model('efficientnet-b3')

# 进行推理
image = Image.open('test.jpg').convert('RGB')
tensor = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(tensor)
    prediction = torch.argmax(output, dim=1).item()

print(f"预测结果: {prediction}")
```

---

## ⚡ 性能对比

| 策略 | 内存占用 | 首次加载时间 | 切换模型时间 |
|------|---------|-------------|-------------|
| 全部加载 | 高（>4GB） | 长（30s+） | 瞬时 |
| 按需加载 | 低（<1GB） | 中（5-10s） | 中（5-10s） |

---

## 📝 关键要点

1. **Singleton 模式**：确保全局只有一个模型管理器
2. **懒加载**：只在需要时加载模型
3. **单模型驻留**：同一时间只保持一个模型在内存中
4. **设备感知**：自动检测 GPU/CPU 并选择最优设备
5. **显存回收**：显式调用 `gc.collect()` 和 `torch.cuda.empty_cache()` 确保显存立即释放

---

## ⚠️ 常见问题与解决方案

### OOM（内存溢出）问题

**问题描述**：多次切换模型后依然触发 OOM。

**解决方案**：
```python
def clear_all_models(self):
    """彻底清理所有模型"""
    # 删除所有模型引用
    for name in list(self._loaded_models.keys()):
        del self._loaded_models[name]
    
    # 强制垃圾回收
    gc.collect()
    
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # 额外清理进程间共享内存
```

**原因分析**：
- Python 的垃圾回收不是即时的
- PyTorch 的 CUDA 缓存不会自动释放
- 模型可能被其他地方引用导致无法回收

---

*下篇预告：API Gateway 设计与实现*
