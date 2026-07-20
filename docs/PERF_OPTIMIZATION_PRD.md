# 性能优化 PRD — Anime Role Detect 系统

## 项目信息

| 字段 | 值 |
|------|-----|
| Language | 中文 |
| Project Name | `anime_role_detect_perf_opt` |
| 原始需求 | 对动漫角色识别系统进行全系统性能审查与优化，覆盖模型推理速度、API/后端性能、前端加载与体验三大方向 |
| 技术栈 | Python 3.9+, FastAPI, SQLAlchemy 2.0, PyTorch 2.1+CPU, Next.js 15, React 18, Redis 7, MySQL 8/SQLite, RabbitMQ |
| 部署方式 | Docker Compose + Supervisor 管理 13 个进程 |

---

## 1. 产品目标

### 目标 1：模型推理延迟降低 60%+
- 单张图片端到端推理（上传→返回结果）从当前 **~3-5s** 降至 **<1.5s**
- 启用 MPS/GPU 加速或优化 CPU 推理路径，消除每次请求 fork 子进程等不合理开销
- 批量预测吞吐量提升 **3x+**

### 目标 2：API 并发能力提升至 50+ QPS
- 消除全局单连接、session 泄漏、响应体全量消费等并发瓶颈
- uvicorn 并发限制从 10 提升至 100+
- 中间件零拷贝化，不阻断流式响应

### 目标 3：前端首屏加载时间降低 50%+
- 从 dev 模式切换到 production build，启用代码压缩与 Tree Shaking
- 首屏 JS bundle 体积降低 40%+（代码分割 + 按需加载）
- 图片上传压缩并行化，批量操作响应时间降低 50%+

---

## 2. 用户故事

### US-1：单图识别用户
> 作为一个动漫爱好者，我希望上传一张角色图片后能在 **1.5 秒内** 得到识别结果，这样我可以在浏览时快速验证角色身份，而不需要长时间等待。

### US-2：批量识别用户
> 作为一个内容创作者，我希望能够批量上传多张图片并同时处理，而不是逐张串行等待，这样我可以高效地整理角色素材库。

### US-3：多角色检测用户
> 作为一个同人画师，我希望上传包含多个角色的图片时，系统能快速检测并识别所有角色，关键点检测不会额外增加 2 秒延迟。

### US-4：系统运维者
> 作为一个系统管理员，我希望在高并发场景下系统不会因为数据库连接耗尽或中间件阻塞而崩溃，监控中间件不会拖慢正常请求的响应速度。

### US-5：前端用户
> 作为一个网页端用户，我希望页面加载快速流畅，首次访问不需要下载未使用的组件代码，图片上传时压缩过程不会阻塞 UI。

---

## 3. 需求池

### P0 — 必须修复（阻断性瓶颈）

#### P0-1：启用 MPS/GPU 加速推理
- **需求描述**：当前 `model_service/app.py` 强制禁用 MPS（`PYTORCH_MPS_DISABLE=1` + monkey-patch `_mps_is_available`），且安装 `torch==2.1.0+cpu`，所有推理在 CPU 上运行，速度慢 5-20x。需评估 MPS 死锁问题的根因，尝试通过进程隔离、超时保护或升级 PyTorch 版本解决，恢复 GPU 加速。
- **预期收益**：单张推理延迟降低 5-20x（从 ~2s 降至 ~100-400ms）
- **涉及文件**：`model_service/app.py:14-38`、`requirements.txt`、`supervisord.conf:14`

#### P0-2：消除关键点检测的子进程 fork
- **需求描述**：`routes.py:163` 每次关键点检测请求都通过 `subprocess.run` 启动新 Python 进程执行 mediapipe 检测，每次增加 ~2s 延迟（进程启动 + 模型加载）。需将 mediapipe 检测器改为进程内常驻单例，通过线程池异步调用。
- **预期收益**：关键点检测延迟从 ~2s 降至 ~100-200ms
- **涉及文件**：`model_service/routes.py:150-187`、`src/core/keypoint/mediapipe_keypoint_detector.py`

#### P0-3：前端切换到 Production Build
- **需求描述**：`supervisord.conf:147-149` 中前端以 `npm run dev` + `NODE_ENV=development` 运行，无代码压缩、无 Tree Shaking、无静态优化。需切换到 `npm run build && npm start`，设置 `NODE_ENV=production`。
- **预期收益**：首屏加载时间降低 50%+，JS bundle 体积降低 60%+
- **涉及文件**：`supervisord.conf:146-161`、`src/frontend/package.json`

---

### P1 — 重要优化（显著性能提升）

#### P1-1：批量预测改为真正的批处理推理
- **需求描述**：`routes.py:574` 批量端点使用 `for file in files` 逐张处理，无真正批处理。需将图片张量 stack 后一次性送入模型推理，利用 PyTorch 的 batch dimension 加速。
- **预期收益**：8 张图片批量推理时间从 ~24s 降至 ~4-6s（4-6x 提升）
- **涉及文件**：`model_service/routes.py:560-636`、`src/core/feature_extraction/feature_extraction.py`

#### P1-2：监控中间件改为流式透传
- **需求描述**：`middleware/monitoring.py:40-51` 每个响应都完整读入内存（`body += chunk`）再重建 Response 对象，阻断流式响应，且大响应体导致内存峰值。需改为仅记录响应大小（通过 `Content-Length` 或 chunk 计数），不消费 body_iterator。
- **预期收益**：消除中间件内存峰值，恢复流式响应能力，减少每个请求的内存分配开销
- **涉及文件**：`src/middleware/monitoring.py:30-65`

#### P1-3：SQLite 连接池优化
- **需求描述**：`config/database.py:109` 使用 `StaticPool`（单连接），所有并发请求共享一个数据库连接，形成瓶颈。需改用 `QueuePool` 或提高 `StaticPool` 连接数，或迁移到 MySQL 作为默认后端。
- **预期收益**：数据库并发能力从 1 提升至 10+ 并发连接
- **涉及文件**：`src/core/config/database.py:106-111`

#### P1-4：aiohttp ClientSession 全局复用
- **需求描述**：`model_processor.py:83` 每次调用模型服务都 `async with aiohttp.ClientSession()` 创建新 session，TCP 连接无法复用，每次都有 DNS 解析 + TCP 握手开销。需创建全局单例 ClientSession，在应用生命周期内复用。
- **预期收益**：API→Model 服务调用延迟降低 ~50-100ms/次（省去连接建立开销）
- **涉及文件**：`src/services/processor/model_processor.py:82-91`

#### P1-5：启用 Next.js 图片优化
- **需求描述**：`next.config.js:6` 设置 `unoptimized: true`，禁用了 Next.js 内置的图片优化（WebP 转换、尺寸适配、懒加载）。需移除该配置或设为 `false`，启用 `next/image` 的自动优化。
- **预期收益**：图片加载体积降低 30-50%，改善图片密集页面的加载体验
- **涉及文件**：`src/frontend/next.config.js:4-7`

#### P1-6：修复 EfficientNet 特征投影层随机初始化问题
- **需求描述**：`classifiers.py:122-125` 中 1536→512 的特征投影层使用 `torch.nn.init.xavier_normal_` 随机初始化，而非训练好的权重。这导致 FAISS 检索使用的 512 维特征向量是随机投影结果，检索质量极低。需加载预训练的投影权重或使用模型原始 1536 维特征。
- **预期收益**：FAISS 检索准确率显著提升，减少降级到 FAISS 搜索时的错误识别
- **涉及文件**：`src/services/model_service/classifiers.py:101-136`

---

### P2 — 进一步优化（效率与资源利用）

#### P2-1：ThreadPoolExecutor 全局复用
- **需求描述**：`routes.py` 多处（行 159、193、216、268、282）每次调用都创建 `ThreadPoolExecutor(max_workers=1)` 临时实例，用完即销毁。需创建模块级全局线程池复用。
- **预期收益**：消除线程创建/销毁开销，每请求节省 ~1-5ms
- **涉及文件**：`model_service/routes.py`（5 处）、`model_service/app.py:159`

#### P2-2：修复全局 DB Session 未归还问题
- **需求描述**：`database_service.py:28` 中 `get_db_service()` 使用全局单例 `_db_session`，通过 `next(get_db())` 获取后永不关闭，导致连接泄漏。需改为每次请求获取独立 session 并在请求结束时归还。
- **预期收益**：消除连接泄漏，提升数据库连接池利用率
- **涉及文件**：`src/services/support/database_service.py:19-29`

#### P2-3：N+1 删除改为批量 DELETE
- **需求描述**：`database_service.py:134-144`（及 342-352）的 `delete_by_user` 方法先 `query().all()` 加载所有记录到内存，再逐条 `db.delete(record)`。需改为 `db.query(Model).filter(...).delete()` 单条 SQL 批量删除。
- **预期收益**：大批量删除从 O(n) 次查询降为 1 次 SQL，删除 100 条记录从 ~100ms 降至 ~5ms
- **涉及文件**：`src/services/support/database_service.py:134-144, 342-352`

#### P2-4：前端批量压缩改为并行
- **需求描述**：`imageCompression.ts:159` 使用 `for` 循环串行 `await compressImage(file)`，批量压缩时总耗时 = N × 单张时间。需改为 `Promise.all` 并行压缩。
- **预期收益**：8 张图片批量压缩时间从 ~8 × 单张 降至 ~1.5 × 单张（受 CPU 核心数限制）
- **涉及文件**：`src/frontend/app/utils/imageCompression.ts:153-177`

#### P2-5：前端代码分割与按需加载
- **需求描述**：`page.tsx`（1191 行）静态导入所有组件（16 个 import），无 `next/dynamic` 使用。首屏加载包含所有页面的组件代码。需将非首屏组件改为 `dynamic(() => import(...), { ssr: false })` 按需加载。
- **预期收益**：首屏 JS bundle 体积降低 30-50%，改善首次加载体验
- **涉及文件**：`src/frontend/app/page.tsx`、相关组件文件

#### P2-6：提升 uvicorn 并发限制
- **需求描述**：`app.py:190` 设置 `limit_concurrency=10`，严重限制并发处理能力。需提升至 100+ 或移除限制（由系统资源自然约束）。
- **预期收益**：并发请求处理能力从 10 提升至 100+
- **涉及文件**：`model_service/app.py:190`

#### P2-7：优化 BLAS 线程配置
- **需求描述**：`app.py:20-24` 将 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 等全部设为 1，完全单线程。需根据 CPU 核心数设置为合理值（如 4 或物理核心数）。
- **预期收益**：CPU 推理时矩阵运算加速 2-4x
- **涉及文件**：`model_service/app.py:20-24`、`supervisord.conf:14,31,115`

#### P2-8：提升 model-service Docker CPU 限制
- **需求描述**：`docker-compose.yml:194` 中 model-service 限制为 `cpus: "1.5"`，对于深度学习推理过低。需提升至 2-4 CPU。
- **预期收益**：模型推理并行度提升，减少 CPU 争抢导致的延迟
- **涉及文件**：`docker-compose.yml:190-194`

---

### P3 — 低优先级（边际优化）

#### P3-1：追踪中间件精简请求头记录
- **需求描述**：`tracing.py:71` 将 `dict(request.headers)` 完整记录为 span 属性，包含 Cookie、Authorization 等敏感信息且增加内存开销。需仅记录必要的请求头（如 User-Agent、Content-Type）。
- **预期收益**：减少每请求 ~2-5KB 内存分配，降低追踪数据存储开销
- **涉及文件**：`src/middleware/tracing.py:64-72`

#### P3-2：精简常驻监控进程
- **需求描述**：`supervisord.conf` 中有 3 个监控守护进程（health-check、log-monitor、resource-monitor），持续消耗 CPU 和内存。需评估是否可合并为 1 个轻量级进程，或降低采样频率。
- **预期收益**：释放 ~100-200MB 内存和 ~0.3 CPU 用于核心服务
- **涉及文件**：`supervisord.conf:180-229`、`scripts/monitoring/health_check.py`、`scripts/monitoring/log_monitor.py`、`scripts/monitoring/resource_monitor.py`

---

## 4. 待确认问题

### 技术决策点

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| Q1 | **MPS 死锁问题是否已修复？** 当前注释说明因 "macOS MPS 后端存在 mutex 死锁问题" 而禁用。是否可以升级 PyTorch 到 2.2+ 并重新测试 MPS 稳定性？如果死锁仍存在，是否考虑 ONNX Runtime / CoreML 作为替代加速方案？ | P0-1 的实现路径 | 先升级 PyTorch 2.2+ 测试 MPS；若仍不稳定则评估 ONNX Runtime EP |
| Q2 | **生产部署环境是 macOS 还是 Linux？** 当前代码大量 macOS 特定逻辑（MPS、OBJC 环境变量）。如果生产环境是 Linux + NVIDIA GPU，则直接启用 CUDA 即可，无需处理 MPS 问题。 | P0-1、P2-7 的实现方案 | 需用户明确生产部署目标平台 |
| Q3 | **数据库后端选择：SQLite 还是 MySQL？** 当前默认 SQLite + StaticPool，但 Docker Compose 已配置 MySQL 8。是否将生产默认切换到 MySQL？还是保持 SQLite 但优化连接池？ | P1-3 的实现方案 | 生产环境建议切换 MySQL，开发环境保留 SQLite |
| Q4 | **EfficientNet 特征投影层是否有训练好的权重？** 当前 `classifiers.py:122` 使用随机初始化。是否存在预训练的投影层权重文件？或者是否应直接使用 1536 维特征进行 FAISS 检索（需重建索引）？ | P1-6 的实现方案 | 需确认是否有训练好的投影权重；若无则评估重建 FAISS 索引 |
| Q5 | **关键点检测的 mediapipe 与 PyTorch MPS 冲突的根因是什么？** 当前通过子进程隔离解决。如果启用 MPS，mediapipe 是否仍会冲突？是否可以通过 mediapipe 的 GPU 后端配置解决？ | P0-2 的实现方案 | 需测试 mediapipe 在 MPS 启用环境下的行为 |
| Q6 | **前端是否需要支持 SSR？** 当前 `next.config.js` 配置 `output: 'standalone'`。代码分割时哪些组件可以安全地 `ssr: false`？ | P2-5 的实现方案 | 需确认各组件的 SSR 需求 |
| Q7 | **并发量预期是多少？** uvicorn `limit_concurrency` 和数据库连接池大小需要根据预期并发量设定。系统预计同时服务多少用户？ | P2-6、P1-3 的参数设定 | 需用户提供目标并发量 |

---

## 5. 优先级与里程碑建议

| 阶段 | 包含需求 | 预期工期 | 验收标准 |
|------|----------|----------|----------|
| **Phase 1：P0 瓶颈修复** | P0-1, P0-2, P0-3 | 3-5 天 | 单图推理 <1.5s；前端首屏 <2s；关键点检测 <300ms |
| **Phase 2：P1 核心优化** | P1-1 ~ P1-6 | 5-7 天 | 批量 8 图 <6s；并发 50 QPS 稳定；FAISS 检索准确率提升 |
| **Phase 3：P2 效率提升** | P2-1 ~ P2-8 | 5-7 天 | 连接泄漏消除；首屏 JS 降低 40%+；uvicorn 并发 100+ |
| **Phase 4：P3 边际优化** | P3-1, P3-2 | 2-3 天 | 监控开销降低；追踪数据精简 |

---

*PRD 版本：v1.0 | 创建者：许清楚（产品经理） | 日期：2025-07-01*
