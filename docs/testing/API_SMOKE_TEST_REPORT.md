# API 端点枚举与冒烟测试报告

> 生成时间：2026-07-30 15:30 (GMT+8)
> 测试环境：本机 supervisord 拉起的 5 个受管 HTTP 服务（model/api/multimedia/search/gateway 全 RUNNING）
> 测试方法：拉取各服务 OpenAPI 描述（权威枚举）+ 源码路由扫描 + 实启 HTTP 冒烟测试（curl / requests）
> 分支：`fix/ci-diagnostic`

---

## 1. 服务与端口总览

| 服务 | 端口 | 进程状态 | Health |
|---|---|---|---|
| model-service | 8000 | RUNNING | ✅ 200 |
| api-service | 8001 | RUNNING | ✅ 200 |
| api-gateway | 8080 | RUNNING | ✅ 200 |
| multimedia-service | 8002 | RUNNING | ✅ 200 |
| search-service | 8003 | RUNNING | ✅ 200 |
| monitor-dashboard | 8888 | 不在 supervisord 管理（独立 K8s Deployment/脚本） | ⚠️ 未在本机启动 |

> 注：api-gateway 通过 `/api/{path}` 通配代理把请求转发到后端各服务，因此绝大多数业务端点既可直接访问后端服务，也可经网关访问。

---

## 2. 端点枚举（按服务）

### 2.1 model-service (8000) — 10 个端点
| Method | Path | 说明 |
|---|---|---|
| POST | `/api/classify` | 单图分类（核心推理） |
| POST | `/api/model/predict` | 预测 |
| POST | `/api/model/detect-yolo` | YOLO 多角色检测 |
| POST | `/api/model/detect-multiple` | 多角色检测 |
| POST | `/api/model/extract` | 特征提取 |
| POST | `/api/model/batch-predict` | 批量预测 |
| GET | `/api/health` | 健康检查 |
| GET | `/live` `/ready` | K8s 探针 |
| GET | `/model_service` | 根 |

### 2.2 api-service (8001) — 83 个路由处理函数（按模块分组）
| 模块 | 关键端点 |
|---|---|
| 认证 `auth.py` | `POST /api/auth/register`、`POST /api/auth/login`、`GET /api/auth/me`、`POST /api/auth/refresh`、`GET /api/admin/test` |
| 分类 `classification.py` | `POST /api/classify`、`POST /api/classify/async`(需鉴权)、`POST /api/batch_classify`、`POST /api/classify/multi-role`、`POST /api/classify/multi-model`、`GET /api/task/{task_id}`、`GET /api/task/{task_id}/result` |
| 模型 `models.py` | `GET /api/models`、`GET/POST /api/model-versions/*`、`GET/POST /api/multi-model/*` |
| 历史 `history.py` | `GET /api/history`、`DELETE /api/history/{record_id}` |
| 健康/监控 `health.py` | `GET /api/health`、`GET /api/health/detailed`、`GET /api/monitoring` |
| 杂项 `misc.py` | `GET /api/config`、`GET /api/docs/info`、`POST /api/feedback`、`GET /api/version` |
| 检索 `search_routes.py` | `POST /build-index`、`POST /image`、`GET /health` |
| 清洗 `cleaning_routes.py` | `GET /browse`、`GET /config/default` |
| 追踪 `tracing.py` | `POST /cleanup`、`GET /health` |
| 采集 `collector.py` | `POST /download/stop` |
| 异步推理 `async_inference.py` | `POST /inference` |

> 注：api-service 不暴露 `/openapi.json`（访问 404，且 `/docs` 会挂起），上述端点来自源码路由扫描。各子路由含 `APIRouter(prefix=...)` 前缀，实际对外路径以表中为准。

### 2.3 multimedia-service (8002) — 15 个端点
| Method | Path | 说明 |
|---|---|---|
| POST | `/search/image` | 以图搜图（search/inference 双模式） |
| GET | `/search/stats` | 检索统计 |
| POST | `/search/build-index` | 构建索引 |
| POST | `/video/extract` | 视频抽帧 |
| POST | `/video/recognize` | 视频识别 |
| POST | `/video/recognize-with-overlay` | 带 overlay 视频识别 |
| POST | `/video/result/cleanup` | 清理结果视频 |
| GET | `/video/result/{filename}` | 下载结果视频 |
| GET | `/video/stats` | 视频统计 |
| GET | `/video/task/{task_id}` | 视频任务状态 |
| GET | `/api/health` `/health` `/info` `/live` `/ready` | 健康/信息/探针 |

### 2.4 search-service (8003) — 8 个端点
| Method | Path | 说明 |
|---|---|---|
| POST | `/api/search/image` | 以图搜图（经文件队列 + 独立 Worker） |
| POST | `/api/search/build-index` | 构建检索索引 |
| GET | `/api/search/stats` | 检索统计（index_count / worker 状态） |
| GET | `/api/search/queue-status` | 队列状态 |
| POST | `/api/video/recognize` | 视频识别 |
| GET | `/live` `/ready` `/api/health` | 探针/健康 |

### 2.5 api-gateway (8080) — 22 个端点（含代理）
| 类别 | 端点 |
|---|---|
| 代理 | `GET/POST/PUT/DELETE /api/{path}`（通配转发到后端） |
| 网关自身 | `GET /`、`GET /api/health`、`GET /api/services`、`GET /health`、`GET /live`、`GET /ready` |
| 日志 `logging` | `GET /logs/`、`GET /logs/services`、`GET /logs/stats`、`GET /logs/tail` |
| 监控 `monitoring` | `GET /monitor/`、`GET /monitor/services`、`GET /monitor/cleaning`(+progress/reset)、`GET /monitor/tracing`(+recent/search/stats/{trace_id})、`GET /api/video/result/{filename}`(代理下载) |

---

## 3. 冒烟测试结果

### 3.1 认证与用户 API（api-service）
| 测试 | 请求 | 结果 | 备注 |
|---|---|---|---|
| 用户注册 | `POST /api/auth/register` | ✅ 200 `success:true` + JWT | 表单字段 username/password/email |
| 正确登录 | `POST /api/auth/login` | ✅ 200 `success:true` + JWT | |
| **错误密码登录** | `POST /api/auth/login` (错密码) | ✅ 200 `success:false`「用户名或密码错误」 | 设计如此（非 401，但正确拒绝） |
| 携带 Token 获取信息 | `GET /api/auth/me` (Bearer) | ✅ 200 返回 username/role | |
| **无 Token 获取信息** | `GET /api/auth/me` | ✅ 403 `Not authenticated` | 鉴权生效 |

### 3.2 模型识别 API
| 测试 | 请求 | 结果 | 备注 |
|---|---|---|---|
| model-service 直连 | `POST /api/classify` (随机噪声图) | ✅ 200 `role=Kirara, similarity=0.175` | 流程通，低相似度为预期 |
| api-service | `POST /api/classify` | ✅ 200 同结果 | 无需鉴权 |
| 网关代理 | `POST /api/classify` (via :8080) | ✅ 200 同结果 | 代理转发正常 |

### 3.3 网关 / 多媒体 / 搜索 / 监控 API
| 测试 | 请求 | 结果 |
|---|---|---|
| 网关健康 | `GET /api/health` | ✅ 200 |
| 网关服务清单 | `GET /api/services` | ✅ 200 列出各服务 |
| 网关监控服务状态 | `GET /monitor/services` | ✅ 200 |
| 网关日志服务 | `GET /logs/services` | ✅ 200 |
| 网关追踪统计页 | `GET /monitor/tracing/stats` | ✅ 200 (HTML) |
| 网关清洗进度 | `GET /monitor/cleaning/progress` | ✅ 200 |
| 多媒体视频统计 | `GET /video/stats` | ✅ 200 |
| 多媒体检索统计 | `GET /search/stats` | ✅ 200（**修复后**，见 §4） |
| 多媒体以图搜图 | `POST /search/image` | ✅ 200（**修复后**，见 §4） |
| 搜索检索统计 | `GET /api/search/stats` | ✅ 200 `index_count=1000` (FAISS 索引已建) |
| 搜索以图搜图 | `POST /api/search/image` | ✅ 200 返回 5 条相似图 |

**结论：所有冒烟测试通过，无 5xx 服务端错误。**

---

## 4. 测试中发现并修复的 Bug

### Bug A（P1）：`multimedia /search/stats` 崩溃
- **现象**：`GET /search/stats` 返回 `{"success":false,"error":"'SimpleImageSearchService' object has no attribute 'get_index_stats'"}`
- **根因**：`multimedia-service/routes.py` 调用了 `SimpleImageSearchService.get_index_stats()`，但该方法从未定义。
- **修复**：在 `src/services/search_service/simple_search_service.py` 新增 `get_index_stats()`，只读 `feature_store` 统计、**不触发 CLIP 模型加载**（避免 OOM）。
- **提交**：`b99b738`

### Bug B（P0，进程级）：`multimedia /search/image` 让服务进程被杀
- **现象**：`POST /search/image`（search 模式）导致服务端连接中断（`RemoteDisconnected`），multimedia 进程被杀死。
- **根因**：本地 `data/feature_store/` 为空（无 FAISS 索引），但 `SimpleImageSearchService` 仍会在首次 `identify()` 时加载 CLIP(ViT-B/32 ~600MB)，在已加载模型的内存受限环境下 OOM 杀进程。
- **修复**：`_ensure_initialized()` 增加守卫——当 FAISS 索引文件不存在时**跳过检索器初始化**，优雅降级返回空结果（`count:0`）；生产环境索引存在时行为不变。
- **提交**：`b99b738`
- **验证**：修复后 `/search/stats` 与 `/search/image` 均稳定返回 200，不再崩溃。

### 备注（非阻塞）
- 网关 `/monitor/services` 把自身 `API Gateway` 标记为 `unhealthy`——疑似自检测逻辑对网关自身 URL 的探活判定有误，不影响业务，建议后续单独排查。
- `monitor-dashboard`(8888) 不在 supervisord 管理内，本机未起；其端点未纳入本次冒烟。

---

## 5. 端点总数统计
| 服务 | 端点数 |
|---|---|
| model-service | 10 |
| api-service | 83（路由函数，含子路由前缀） |
| multimedia-service | 15 |
| search-service | 8 |
| api-gateway | 22（含通配代理 + 日志/监控） |
| **合计** | **138** |

---

## 6. 后续建议
1. 将 `fix/ci-diagnostic` 合并进 `main` 并触发 deploy-k8s CI，验证端到端（含本修复）。
2. 排查网关自检测 `API Gateway: unhealthy` 误判。
3. 评估 multimedia 是否应把图像检索**代理给 search-service**（与 classify 代理给 model-service 一致），从架构上消除 multimedia 进程内加载 CLIP 的隐患（本次仅做了优雅降级）。
4. 补 `monitor-dashboard` 的本地启动/健康检查纳入 supervisord 或文档说明。
