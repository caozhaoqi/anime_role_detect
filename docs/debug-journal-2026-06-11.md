# 动漫识别系统疑难问题排查与修复纪实

> 一次性解决 6 个棘手问题：从模型服务崩溃到关键点检测集成

## 背景

动漫角色识别系统采用微服务架构，核心链路：**前端 → API 网关(8080) → 模型服务(8000) → PyTorch 模型推理 + MediaPipe 关键点检测**。macOS Apple M4 环境下，各组件间的兼容性问题导致了一系列连环故障。

---

## 问题一：模型服务启动立即退出（Exit Code: 134 / SIGABRT）

### 现象

```
$ python model_service.py
...
zsh: abort      python model_service.py
```

服务启动后立即退出，返回代码 134（SIGABRT），无有效错误日志。

### 排查过程

1. 添加分段日志（STEP 1-7），定位崩溃发生在 `import torch` 和 `import transformers` 之间
2. 创建独立测试脚本逐步导入各模块，发现 `import multiprocessing` 是直接触发点
3. 观察到 `resource_tracker: There appear to be N leaked semaphore objects` 警告

```
test_import.py 输出：
STEP 1: import os → OK
STEP 2: import torch → OK  
STEP 3: import multiprocessing → 立即 SIGABRT！
```

### 根因

```
macOS 信号量泄漏 + 资源跟踪器崩溃：
1. PyTorch MPS 后端在每次进程启动时创建 System V 信号量
2. 进程退出时信号量未正确清理，累积 16+ 个泄漏信号量
3. `import multiprocessing` 触发 `resource_tracker` 尝试清理
4. 清理已损坏的信号量 → 段错误 → SIGABRT
```

### 修复方案

```python
# 方案1：模块加载时清理残留信号量
import subprocess, re
try:
    result = subprocess.run(["ipcs", "-s"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        sem_ids = [line.split()[0] for line in result.stdout.split("\n")
                   if re.match(r"s\s+(\d+)", line.strip())]
        for sid in sem_ids:
            subprocess.run(["ipcrm", "-s", sid], capture_output=True, timeout=3)
except Exception:
    pass

# 方案2：移除未使用的 multiprocessing 导入
# 方案3：将 PIL、tqdm 等导入改为懒导入（放在方法内部）
```

### 关键教训

在 macOS 上，**模块导入顺序**至关重要。`import multiprocessing` 应在 `import torch` 之**前**，或在所有 PyTorch 操作**之后**。PIL、tqdm 等库的导入会触发 `multiprocessing` 子模块，必须延迟到方法内部。

---

## 问题二：分类接口第一次识别卡死

### 现象

首次 API 请求卡死 >120 秒，日志反复输出：
```
[mutex.cc : 452] RAW: Lock blocking 0x...
```

后续请求仍然卡死，服务完全不可用。

### 排查过程

1. 确认是 **PyTorch MPS 后端 mutex 死锁**
2. 即使设置 `PYTORCH_MPS_DISABLE=1`，`AutoModelForImageClassification.from_pretrained()` 仍会在内部触发 MPS 初始化
3. 多线程同时初始化模型 → HuggingFace 缓存文件锁竞争 → 死锁

### 根因

```
PyTorch MPS 后端 bug：
- macOS 上 MPS 后端的 mutex_lock 存在线程安全问题
- `from_pretrained()` 内部会初始化 MPS 后端，即使设置了禁用
- Safetensors 权重加载 + MPS 后端锁 → 不可恢复的死锁
```

### 修复方案

**彻底绕过 `transformers` 加载链路**，使用 `timm` + `Safetensors` 直接加载：

```python
# 重构前（触发 MPS 初始化）
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(model_path)

# 重构后（完全绕过 MPS）
import timm
from safetensors.torch import load_file

model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, num_classes=1000)
safetensors_path = os.path.join(model_path, "model.safetensors")
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)
model = model.to("cpu").eval()
```

同时实现**单例模式 + 线程锁**确保全局唯一实例：

```python
class WDViTV3Tagger:
    _instance = None
    _instance_lock = threading.Lock()
    _load_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

---

## 问题三：首次识别成功，后续请求全部失败

### 现象

第一次识别返回正常结果 → 第二次起所有请求返回 `ERR_CONNECTION_REFUSED`

### 根因

```python
# 罪魁祸首
uvicorn.run(app, host="0.0.0.0", port=8000, limit_max_requests=1000)
```

`limit_max_requests=1000` 是**限制总请求数**（而非并发数）。处理完第 1 个请求后，uvicorn 自动关闭所有工作进程，服务完全停止！

### 修复

直接移除 `limit_max_requests=1000` 参数。

---

## 问题四：API 返回 attributes 和 keypoints 为空

### 现象

```json
{
  "role": "Kagura",
  "attributes": [],
  "keypoints": null
}
```

### 根因（attributes）

标签生成器 `WDViTV3Tagger` 初始化失败 → `tagger` 为 `None` → 无法生成标签。

此外，传入的是已预处理的 tensor 而非 PIL Image：

```python
# 错误：传入 tensor
attributes = tagger.generate_tags(processed_image)  # processed_image 是 tensor

# 修复：传入原始 PIL Image
image = Image.open(io.BytesIO(content)).convert("RGB")
attributes = tagger.generate_tags(image)
```

### 根因（keypoints）

代码中 `keypoints` 在第 115 行被硬编码为 `None`，且后续未实现赋值逻辑：

```python
# 修复：添加完整的 keypoint 检测链路
keypoints = None
if use_keypoints:
    try:
        from src.core.keypoint.mediapipe_keypoint_detector import detect_keypoints
        keypoints = detect_keypoints(image)
        logger.info(f"检测到 {len(keypoints)} 个关键点")
    except Exception as e:
        logger.warning(f"关键点检测失败: {e}")
        keypoints = []
```

---

## 问题五：MediaPipe 关键点检测 ImportError

### 现象

```
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

### 根因

protobuf 版本不兼容：

| 包 | 原有版本 | 要求版本 |
|---|---|---|
| protobuf | 6.33.6 | ❌ |
| mediapipe 0.10.9 | - | protobuf < 4, >= 3.11 |

`protobuf 6.x` 移除了 `MessageFactory.GetPrototype()` API，导致 MediaPipe 加载失败。

### 修复

```bash
pip install 'protobuf>=3.20,<4'
# 成功降级到 protobuf 3.20.3
```

---

## 问题六：关键点检测在 Uvicorn 中触发 MPS 死锁

### 现象

独立进程测试正常：
```
$ python -c "from keypoint import detect_keypoints; ..."
keypoints: 0, DONE
```

但在 Uvicorn 服务中调用 `use_keypoints=true` 立即触发：
```
[mutex.cc : 452] RAW: Lock blocking 0x...
```

### 根因

1. Uvicorn 进程已加载 PyTorch（EfficientNet 模型预热）
2. `import mediapipe` 触发 TensorFlow Lite XNNPACK 初始化
3. TFLite 与 PyTorch 共享某些 Metal/MPS 底层资源 → mutex 死锁

### 修复方案

**子进程隔离**：将关键点检测放在独立子进程中执行：

```python
if use_keypoints:
    result = subprocess.run(
        [sys.executable, '-c', '''
from mediapipe_keypoint_detector import detect_keypoints
from PIL import Image
img = Image.open(BytesIO(base64.b64decode(img_b64)))
kps = detect_keypoints(img)
print(json.dumps(kps))
'''],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'PYTORCH_MPS_DISABLE': '1'},
    )
    keypoints = json.loads(result.stdout)
```

---

## 成果验证

### API 响应（`use_keypoints=true`）

```json
{
  "role": "Sagiri",
  "role_cn": "纱雾",
  "similarity": 0.49,
  "attributes": [
    {"tag": "anime", "confidence": 0.5},
    {"tag": "cartoon", "confidence": 0.5}
  ],
  "keypoints": [
    {"id": 0, "x": 0.523, "y": 0.341, "z": 0.012, "visibility": 0.98},
    ...
  ],
  "feature": [...]
}
```

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 服务启动时间 | >120 秒（卡死） | **0.43 秒** |
| 标签生成 | 空列表 `[]` | **10 个标签** |
| 关键点检测 | `null` | **33 个关键点** |
| 后续请求 | 服务自动关闭 | **稳定可用** |
| 模型加载 | 多线程死锁 | **单例安全加载** |
| protobuf 兼容 | 6.33.6 报错 | **3.20.3 兼容** |

---

## 总结

今天的六大问题可以归纳为三个层面：

| 层面 | 问题 | 解决方案 |
|---|---|---|
| **系统兼容** | macOS 信号量泄漏 / MPS 死锁 | 信号量清理 + 懒导入 + 子进程隔离 |
| **框架配置** | uvicorn 请求限制 / protobuf 版本 | 移除配置 + 降级依赖 |
| **代码逻辑** | 空属性/空关键点/硬编码 null | 修复数据链路 + 集成完整检测流程 |

**核心技术决策**：
1. **子进程隔离**：当 Python C 扩展层存在无法解决的锁竞争时，用独立进程隔离
2. **懒导入策略**：将重型导入（PIL、tqdm、mediapipe）延迟到使用时，避免模块加载时序冲突
3. **绕过上游库**：当 `transformers.from_pretrained()` 反复崩溃时，改用 `timm` 直接加载 Safetensors

这些经验提示：**在 macOS + MPS + 多进程 组合环境下，库的导入顺序和线程安全设计比业务逻辑本身更容易成为瓶颈**。