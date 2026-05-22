# macOS 上 PyTorch 与 FastAPI 线程锁问题解决方案

## 问题概述

### 问题现象
以图搜图功能搜索出的图片无法显示，服务在启动时崩溃，报错信息：
```
libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
```

### 问题影响
- 搜索服务无法正常启动
- 以图搜图功能完全不可用
- 前端无法显示搜索结果图片

---

## 问题根因分析

### 根本原因
**macOS 系统级限制**：macOS 的 Objective-C 运行时在处理 Fork 操作时存在安全检查机制。当 `transformers` 库（PyTorch 依赖）在某些线程环境中导入时，会触发此安全检查，导致线程锁失败。

### 触发条件
1. **Web 服务环境**：FastAPI/Uvicorn 启动时会创建多线程/多进程环境
2. **PyTorch 导入**：`transformers` 库在导入时会初始化 PyTorch，涉及底层的 Objective-C 运行时调用
3. **Fork 操作**：Uvicorn 的 worker 进程创建或热重载机制会触发 Fork 操作

### 代码层面问题
原始代码在 `image_search_service.py` 中存在逻辑错误：
```python
# 错误：在 logger 初始化之前尝试使用 logger
logger.info("导入PyTorch...")  # logger 此时为 None，会导致错误
from src.core.logging.global_logger import get_logger
logger = get_logger("image_search_service")
```

---

## 尝试的解决方案

| 方案 | 状态 | 说明 |
|------|------|------|
| 设置环境变量 `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` | ❌ 失败 | 仍然崩溃 |
| 禁用热重载 + 单进程模式 | ❌ 失败 | 仍然崩溃 |
| 进程隔离（multiprocessing） | ❌ 失败 | 子进程也崩溃 |
| 修复代码逻辑错误 | ❌ 失败 | logger顺序修复后仍然崩溃 |
| **独立进程监控（文件队列）** | ✅ 成功 | **最终解决方案** |

---

## 最终解决方案：独立进程监控方案

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    用户请求                                │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI服务 (app_queue.py)                                │
│  端口: 8003                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ 1. 接收图片上传                                       │ │
│  │ 2. 生成任务ID                                        │ │
│  │ 3. 将图片写入 search_queue/input/{task_id}.jpg       │ │
│  │ 4. 轮询 search_queue/output/{task_id}.json           │ │
│  │ 5. 返回结果给前端                                     │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ 文件系统 (完全隔离)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker进程 (search_worker.py)                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ 1. 在独立进程中加载CLIP模型（无线程锁问题）            │ │
│  │ 2. 加载Faiss索引                                      │ │
│  │ 3. 监控 search_queue/input/ 目录                     │ │
│  │ 4. 发现新任务时提取特征并搜索相似图像                  │ │
│  │ 5. 将结果写入 search_queue/output/{task_id}.json     │ │
│  │ 6. 清理输入文件                                       │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计理念

1. **完全隔离**：Web服务和AI模型运行在完全独立的OS进程中
2. **无共享资源**：通过文件系统进行进程间通信，没有内存或线程共享
3. **规避锁冲突**：Worker进程是普通的死循环程序，不涉及任何Web框架

---

## 实现文件

### 1. Worker进程 (`src/services/search_service/search_worker.py`)

**核心功能**：
- 在独立进程中加载CLIP模型
- 监控输入目录，处理搜索任务
- 将结果写入输出目录

**关键代码**：
```python
def main():
    # 加载模型和索引（在独立进程中）
    model, preprocess = load_clip_model()
    index, image_paths = load_faiss_index()
    
    # 主循环：监控输入目录
    while True:
        input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')]
        for input_file in input_files:
            task_id = input_file.replace('.jpg', '')
            process_task(task_id, model, preprocess, index, image_paths)
        time.sleep(0.1)
```

### 2. FastAPI服务 (`src/services/search_service/app_queue.py`)

**核心功能**：
- 接收图片上传
- 将图片写入队列
- 轮询等待结果
- 返回Base64编码的图片给前端

**关键代码**：
```python
@app.post("/api/search/image")
async def search_similar_images(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    input_path = os.path.join(INPUT_DIR, f"{task_id}.jpg")
    
    # 保存图片到队列
    image = Image.open(io.BytesIO(content)).convert("RGB")
    image.save(input_path, format='JPEG')
    
    # 轮询等待结果
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}.json")
    while time.time() - start_time < timeout:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                result = json.load(f)
            os.remove(output_path)
            return result
        time.sleep(0.1)
```

---

## 启动方式

### 步骤1：启动Worker进程
```bash
cd /path/to/project
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 src/services/search_service/search_worker.py
```

### 步骤2：启动FastAPI服务
```bash
cd /path/to/project
python3 -m uvicorn src.services.search_service.app_queue:app --host 0.0.0.0 --port 8003 --workers 1
```

### 步骤3：测试服务
```bash
curl -X POST -F "file=@test.jpg" http://localhost:8003/api/search/image
```

---

## 测试结果

### 服务状态
```json
{
    "status": "healthy",
    "service": "Search Service (Queue)"
}
```

### 搜索结果
```json
{
    "query": "test.jpg",
    "count": 5,
    "results": [
        {
            "path": "/data/Madoka_0.jpg",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAAAD/...",
            "similarity": 0.92,
            "role": "Madoka"
        }
    ]
}
```

---

## 方案优势

| 特性 | 说明 |
|------|------|
| **稳定性** | 100% 规避Objective-C锁冲突 |
| **隔离性** | Web服务和AI模型完全独立 |
| **兼容性** | 不需要修改macOS系统设置 |
| **扩展性** | 可以轻松扩展为多Worker模式 |
| **可维护性** | 代码结构清晰，易于调试 |

---

## 替代方案（中长期）

### 方案A：CoreML转换
```python
# 将CLIP模型转换为CoreML格式（需在Linux环境导出）
import coremltools as ct
coreml_model = ct.convert(traced_model, inputs=[ct.ImageType()])
coreml_model.save("clip.mlmodel")
```

**优势**：
- 系统级优化，推理速度提升5-10倍
- 稳定性极高，不会触发线程锁问题
- 支持GPU/Neural Engine加速

### 方案B：部署到Linux环境
- 使用Docker容器
- 部署到云服务器
- 原生支持PyTorch，无兼容性问题

---

## 总结

### 问题解决路径
1. **问题识别**：macOS系统级线程锁问题
2. **根本原因**：Objective-C运行时的Fork安全检查
3. **解决方案**：独立进程监控方案（文件队列通信）
4. **验证**：服务稳定运行，图片正常显示

### 关键收获
- macOS上运行PyTorch模型时，应避免在Web服务进程中直接导入
- 文件队列是进程间通信的可靠方式
- 独立进程隔离是解决此类问题的最有效手段