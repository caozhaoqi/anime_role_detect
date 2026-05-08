# 【技术难点】前端实时性与 WebSocket 实现

> 当模型推理时间较长时，如何为用户提供实时的进度反馈？本文介绍 WebSocket 在长耗时任务中的应用。

---

## 🔍 问题背景

模型推理的特点：

| 模型 | 推理时间 | 用户体验问题 |
|------|---------|-------------|
| MobileNetV2 | ~50ms | 响应快，用户无感知 |
| EfficientNet-B0 | ~100ms | 轻微延迟 |
| EfficientNet-B3 | ~500ms | 明显等待 |
| 多角色检测 | 1-3s | 需要进度提示 |

**核心挑战**：如何在长耗时任务中提供实时反馈，避免用户以为页面卡死？

---

## 💡 解决方案：WebSocket 实时通信

### 后端 WebSocket 服务

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, List
import asyncio
import json

app = FastAPI(title="Real-time Recognition Service")

# 存储活动连接
active_connections: Dict[str, WebSocket] = {}

class RecognitionTask(BaseModel):
    task_id: str
    image_hash: str
    status: str = "pending"  # pending, processing, completed, failed
    progress: int = 0
    result: dict = None
    error: str = None

tasks: Dict[str, RecognitionTask] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_connections[client_id] = websocket
    print(f"✅ 客户端 {client_id} 已连接")
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_recognition":
                task_id = message["task_id"]
                image_hash = message["image_hash"]
                
                # 创建任务
                tasks[task_id] = RecognitionTask(
                    task_id=task_id,
                    image_hash=image_hash,
                    status="processing"
                )
                
                # 模拟长时间推理过程
                await simulate_recognition(task_id, client_id)
            
            elif message["type"] == "query_status":
                task_id = message["task_id"]
                if task_id in tasks:
                    await websocket.send_text(json.dumps({
                        "type": "status_update",
                        "task_id": task_id,
                        "status": tasks[task_id].status,
                        "progress": tasks[task_id].progress,
                        "result": tasks[task_id].result,
                        "error": tasks[task_id].error
                    }))
    
    except WebSocketDisconnect:
        del active_connections[client_id]
        print(f"❌ 客户端 {client_id} 已断开连接")

async def simulate_recognition(task_id: str, client_id: str):
    """模拟长时间的识别过程"""
    websocket = active_connections.get(client_id)
    if not websocket:
        return
    
    # 阶段1: 图片预处理 (0-30%)
    for i in range(0, 31, 10):
        tasks[task_id].progress = i
        await websocket.send_text(json.dumps({
            "type": "progress",
            "task_id": task_id,
            "progress": i,
            "stage": "图片预处理"
        }))
        await asyncio.sleep(0.2)
    
    # 阶段2: 特征提取 (30-60%)
    for i in range(30, 61, 10):
        tasks[task_id].progress = i
        await websocket.send_text(json.dumps({
            "type": "progress",
            "task_id": task_id,
            "progress": i,
            "stage": "特征提取"
        }))
        await asyncio.sleep(0.3)
    
    # 阶段3: 模型推理 (60-90%)
    for i in range(60, 91, 10):
        tasks[task_id].progress = i
        await websocket.send_text(json.dumps({
            "type": "progress",
            "task_id": task_id,
            "progress": i,
            "stage": "模型推理"
        }))
        await asyncio.sleep(0.4)
    
    # 阶段4: 结果整理 (90-100%)
    for i in range(90, 101, 10):
        tasks[task_id].progress = i
        await websocket.send_text(json.dumps({
            "type": "progress",
            "task_id": task_id,
            "progress": i,
            "stage": "结果整理"
        }))
        await asyncio.sleep(0.1)
    
    # 完成
    tasks[task_id].status = "completed"
    tasks[task_id].result = {
        "prediction": 42,
        "confidence": 0.95,
        "character_name": "纳西妲",
        "attributes": {"hair_color": "green", "eye_color": "green"}
    }
    
    await websocket.send_text(json.dumps({
        "type": "completed",
        "task_id": task_id,
        "result": tasks[task_id].result
    }))
```

### 前端 WebSocket 客户端

```javascript
// WebSocket 客户端封装
class RecognitionClient {
  constructor() {
    this.ws = null;
    this.clientId = this.generateClientId();
    this.callbacks = {};
  }

  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`ws://localhost:8000/ws/${this.clientId}`);

      this.ws.onopen = () => {
        console.log('✅ WebSocket 连接成功');
        resolve();
      };

      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket 错误:', error);
        reject(error);
      };

      this.ws.onclose = () => {
        console.log('❌ WebSocket 连接关闭');
      };
    });
  }

  on(eventType, callback) {
    if (!this.callbacks[eventType]) {
      this.callbacks[eventType] = [];
    }
    this.callbacks[eventType].push(callback);
  }

  handleMessage(message) {
    const callbacks = this.callbacks[message.type];
    if (callbacks) {
      callbacks.forEach(callback => callback(message));
    }
  }

  startRecognition(imageHash) {
    const taskId = `task_${Date.now()}`;
    this.ws.send(JSON.stringify({
      type: 'start_recognition',
      taskId,
      imageHash
    }));
    return taskId;
  }

  queryStatus(taskId) {
    this.ws.send(JSON.stringify({
      type: 'query_status',
      taskId
    }));
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用示例
const client = new RecognitionClient();

// 监听进度更新
client.on('progress', (message) => {
  console.log(`进度: ${message.progress}% - ${message.stage}`);
  updateProgressBar(message.progress, message.stage);
});

// 监听完成
client.on('completed', (message) => {
  console.log('识别完成:', message.result);
  displayResult(message.result);
});

// 启动识别
async function recognize(imageHash) {
  try {
    await client.connect();
    const taskId = client.startRecognition(imageHash);
    console.log(`任务ID: ${taskId}`);
  } catch (error) {
    console.error('连接失败:', error);
  }
}
```

---

## 🚀 使用示例

### 前端进度条组件

```jsx
import { useState, useEffect, useCallback } from 'react';

function RecognitionProgress({ imageHash }) {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('准备中');
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleProgress = useCallback((message) => {
    setProgress(message.progress);
    setStage(message.stage);
  }, []);

  const handleCompleted = useCallback((message) => {
    setResult(message.result);
    setIsProcessing(false);
  }, []);

  useEffect(() => {
    const client = new RecognitionClient();
    
    client.on('progress', handleProgress);
    client.on('completed', handleCompleted);

    const startRecognition = async () => {
      setIsProcessing(true);
      setProgress(0);
      try {
        await client.connect();
        client.startRecognition(imageHash);
      } catch (error) {
        console.error('识别失败:', error);
        setIsProcessing(false);
      }
    };

    if (imageHash && isProcessing) {
      startRecognition();
    }

    return () => {
      client.disconnect();
    };
  }, [imageHash, isProcessing, handleProgress, handleCompleted]);

  return (
    <div className="recognition-container">
      {isProcessing && (
        <div className="progress-wrapper">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="progress-text">
            {stage} - {progress}%
          </div>
        </div>
      )}
      
      {result && (
        <div className="result-card">
          <h3>识别结果</h3>
          <p>角色: {result.character_name}</p>
          <p>置信度: {result.confidence * 100}%</p>
        </div>
      )}
    </div>
  );
}
```

### 长轮询替代方案（不支持 WebSocket 时）

```javascript
// 长轮询实现
async function longPollStatus(taskId) {
  while (true) {
    try {
      const response = await fetch(`/api/recognition/status/${taskId}`);
      const data = await response.json();

      if (data.status === 'completed' || data.status === 'failed') {
        console.log('任务完成:', data);
        displayResult(data.result);
        break;
      }

      console.log(`进度: ${data.progress}%`);
      updateProgressBar(data.progress);

      // 等待1秒后再次查询
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (error) {
      console.error('轮询失败:', error);
      break;
    }
  }
}
```

---

## ⚡ 实时通信流程图

```
用户上传图片
    ↓
WebSocket 连接建立
    ↓
发送 start_recognition 消息
    ↓
┌─────────────────────────────────────────────────────┐
│              后端处理过程                            │
├─────────────────────────────────────────────────────┤
│  阶段1: 图片预处理 (0-30%)                          │
│    ↓                                               │
│  阶段2: 特征提取 (30-60%)                          │
│    ↓                                               │
│  阶段3: 模型推理 (60-90%)                          │
│    ↓                                               │
│  阶段4: 结果整理 (90-100%)                         │
└─────────────────────────────────────────────────────┘
    ↓
每阶段发送 progress 消息
    ↓
发送 completed 消息
    ↓
前端显示结果
```

---

## 📝 关键要点

1. **WebSocket 优势**：实时双向通信，避免频繁轮询
2. **进度分阶段**：将长任务拆分为多个阶段，提供更细粒度的反馈
3. **错误处理**：处理连接断开、超时等异常情况
4. **任务追踪**：使用 task_id 关联请求和结果
5. **降级方案**：提供长轮询作为 WebSocket 的替代方案
6. **客户端管理**：维护活跃连接列表，支持多客户端同时连接

---

## 📚 系列文章汇总

| 文章 | 主题 | 文件 |
|------|------|------|
| 第1篇 | 多模型集成与性能优化 | `01_multi_model_management.md` |
| 第2篇 | API Gateway 设计与实现 | `02_api_gateway.md` |
| 第3篇 | 分布式服务协调 | `03_distributed_coordination.md` |
| 第4篇 | 图像预处理与特征提取 | `04_image_preprocessing.md` |
| 第5篇 | NSFW 内容过滤 | `05_nsfw_detection.md` |
| 第6篇 | 爬虫反爬机制突破 | `06_anti_crawler.md` |
| 第7篇 | 数据持久化与缓存层 | `07_storage_and_cache.md` |
| 第8篇 | Docker Compose 部署 | `08_docker_deployment.md` |
| 第9篇 | 前端实时性与 WebSocket | `09_websocket_realtime.md` |

---

*感谢阅读！如有问题欢迎留言讨论。*
