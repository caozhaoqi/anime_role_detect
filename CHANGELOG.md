# CHANGELOG



## v2.3.1
### 2026-07-25

- 模型检测bug修复


## v2.3.0
### 2026-07-20

- JWT SECRET_KEY 不再硬编码，每次启动生成随机临时密钥，生产环境须通过环境变量注入
- Supervisord 弱密码替换为 `CHANGE_ME_supervisor_admin/CHANGE_ME_supervisor_pwd`
- 模型服务临时文件路径注入修复：文件名清洗 + 限定系统临时目录 + 使用 uuid
- 模型服务添加内部认证中间件（X-Internal-Service-Token / 内部 IP 白名单）
- ONNX API 移除免认证豁免（/api/v1/onnx 不再在 exempt_paths 中）
- conftest.py 语法错误修复（if 语句缺少函数体）


## v2.0.0
### 2026-07-01

- 微服务架构重构
- 新增 API Gateway 统一入口
- YOLOv8 多角色检测集成
- Docker Compose 一键部署
- Prometheus 监控体系
- RabbitMQ 异步任务队列

## v1.2.0
### 2026-06-20

- 新增搜索服务（FAISS 向量索引）
- CLIP 特征提取集成
- 多媒体服务（视频帧提取、OCR）
- DeepDanbooru 标签生成

## v1.1.0
### 2026-06-10

- 数据管道重构（采集/清洗/去重）
- 训练脚本优化（MPS 加速支持）
- 前端 Next.js 迁移
- 用户认证系统

## v1.0.1
### 2026-06-01

- 链路追踪功能添加
- 修复视频识别错误

## v1.0.0
### 2026-05-01

- 初始版本发布