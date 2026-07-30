# 项目代码结构 (Project Structure)

> 最后更新：2026-07-30 · 适用版本：v2.3.0
> 本文档描述当前以 `src/` 为核心的实际代码结构（旧的 `scripts/main.py` 体系已废弃）。

## 1. 总览

本项目是一个 AI 驱动的动漫/游戏角色识别系统，采用微服务架构：

- **后端**：FastAPI（Python 3.9+），可编辑安装 `pip install -e .`，包根 `src/`
- **前端**：Next.js 15 + React 18 + TypeScript（位于 `src/frontend/`）
- **基础设施**：Redis 7、MySQL 8.0（或 SQLite 三层降级）、RabbitMQ 3
- **进程管理**：`supervisord.conf`（11 个 program）
- **部署**：Docker Compose + Kubernetes（Kustomize，base/ci 覆盖）

## 2. 顶层目录

```
anime_role_detect/
├── src/                    # 全部源码（可编辑安装）
├── models/                 # 模型权重（git-ignored）
├── tests/                  # 测试套件
├── docs/                   # 文档（架构/部署/训练/博客）
├── scripts/                # 运维/工具脚本（k8s、monitoring、data_*、评估…）
│   └── skillhub/           # ⚠️ 遗留实验子项目（88MB，无外部引用，不纳入构建）
├── archived/               # 历史/损坏模块（spider_image_system、arona、broken_modules）
├── deployment/             # K8s 与 Docker 部署文件
├── k8s/                    # Kustomize overlays（base / ci）+ 本地 registry 辅助脚本
├── config/                 # 配置模板
├── supervisord.conf        # 进程管理器配置（11 个 program）
├── docker-compose.yml      # Docker Compose 编排
├── Dockerfile / Dockerfile.model
├── requirements-*.txt       # 分层依赖（base / ml / model-service / scripts / dev）
├── pyproject.toml          # 项目元数据（v2.3.0）+ 工具配置
└── .env.example            # 环境变量模板
```

## 3. `src/` 模块说明

### 3.1 服务入口（supervisord 实际运行的 11 个 program）

| 模块 | 入口文件 | 端口 | 说明 |
|------|----------|------|------|
| model-service | `src/services/model_service/app.py` | 8000 | 模型推理服务 |
| api-service | `src/api/app.py` | 8001 | 主 API 服务 |
| api-gateway | `src/services/api_gateway/app.py` | 8080 | API 网关（聚合文档） |
| multimedia-service | `src/services/multimedia_service/multimedia_service_app.py` | 8002 | 多媒体服务 |
| search-service | `src/services/search_service/app_queue.py` | 8003 | 搜索服务 |
| search-worker | `src/run/sh/start_search_worker.sh` | — | 搜索 worker |
| inference-worker | `src/services/inference_worker/worker.py` | — | CLIP 推理 worker |
| frontend | `src/frontend` (`npm run build && npm start`) | 3000 | 前端 |
| health-check | `scripts/monitoring/health_check.py` | — | 健康检查 |
| log-monitor | `scripts/monitoring/log_monitor.py` | — | 日志监控 |
| resource-monitor | `scripts/monitoring/resource_monitor.py` | — | 资源监控 |

### 3.2 核心业务模块

| 模块 | 职责 |
|------|------|
| `src/api/` | FastAPI 应用与路由（`routes/`：classify、auth、collector、cleaning、tracing…） |
| `src/core/` | 核心能力：分类(classification)、检测(detection)、打标(tagging)、识别(recognition)、OCR、特征提取、预处理、日志、配置(config)、缓存(cache)、错误/异常、版本、服务(service)、Celery 配置 |
| `src/services/` | 微服务实现：api_gateway、model_service、multimedia_service、search_service、inference_worker、cache_service、video_service、messaging(aio_pika)、processor(模型加载/图像处理)、model、support、training |
| `src/models/` | 训练(training)、评估(evaluation)、预测(prediction)、部署(deployment) |
| `src/tasks/` | Celery 任务：classify_tasks、image_tasks、video_tasks、model_tasks、cleanup |
| `src/data/` | 数据采集(`collection/`)与预处理(`preprocessing/`) |
| `src/data_pipeline/` | 数据清洗(cleaners/cleaner)、构建(build)、标注(annotator)、WebUI、数据库 |
| `src/data_collection/` | （遗留）关键词采集入口，被 `distributed_manager` / `series_based_collector` 引用 |
| `src/utils/` | 公共工具：image_utils、http_utils、concurrency_manager、monitoring(tracing/opentelemetry) |
| `src/middleware/` | HTTP 中间件 |
| `src/run/` | 服务管理（services_config、start_all*、start_core）、监控面板(monitor/) |
| `src/frontend/` | Next.js 前端应用（`app/`、`out/`、`node_modules/`） |

## 4. 遗留 / 未使用代码（2026-07-30 清理记录）

为降低维护负担，本次对已确认无引用或已损坏的代码做了处理：

1. **死亡模块归档**：`src/models/training/convert_model_format.py` 存在缩进语法错误且全项目无引用 → 移至 `archived/broken_modules/`。
2. **`scripts/skillhub/`**：自带 venv / `.pytest_cache` 的内嵌实验子项目（88MB），无任何外部引用 → 保留作历史，从构建路径排除（`Dockerfile` 分层镜像不安装它）。
3. **遗留启动器**：`src/run/start_all.py`、`start_all_stable.py`、`src/application.py`、`src/run/start_core.py` 不被 supervisord/k8s/docker 使用，仅自引用 → 保留为开发工具，README 中标注为非生产入口。
4. **Lint 清理**：移除 `src/` 内约 30 处未使用导入/未使用变量（如 `api_gateway/app.py` 的 `RedirectResponse`/`get_swagger_ui_html`、`message_queue_service.py` 的 `pika`、多个 `typing.Set` 等），均为低风险修改，已通过 `py_compile` 校验。
5. **重复目录树（技术债，未删除）**：`src/data/`、`src/data_collection/`、`src/data_pipeline/` 三棵数据采集树存在功能重叠（`data_collection/keyword_based_collector.py` 与 `data/collection/general/keyword_based_collector.py` 同名但实现不同），因仍被引用，本次仅记录，待后续统一。

## 5. 运行时产物（均 git-ignored，不应提交）

`logs/`、`data/`、`models/`、`*.db`、`dump.rdb`、`recognition.db`、`.coverage`、
`src/cache/`、`src/clip_cache/`、`src/huggingface_cache/`、`src/temp/`、`src/anime_role_detect_logs/`、
`outputs/`、`deliverables/`、`src/frontend/.next/`、`src/frontend/out/`、`egg-info` 等。

## 6. 编码规范

- Python 3.9+，PEP 8，UTF-8
- 命名：模块 `snake_case`，类 `PascalCase`，函数/变量 `snake_case`
- 统一通过 `src.` 绝对导入（已移除全部 `sys.path.insert`）
- 配置集中在 `src/config/`，支持 MySQL → SQLite → Memory 三层降级
