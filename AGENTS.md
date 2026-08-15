# AGENTS.md — anime_role_detect 项目协作规范

> 本文件供 AI 编码助手（WorkBuddy / CodeBuddy / Cursor / Claude 等）阅读，
> 目的是把已验证的工程铁律固化成护栏，避免重蹈覆辙。

## 1. 项目性质
- AI 驱动的动漫/游戏角色识别系统，微服务架构（约 13 个服务）。
- 技术栈：FastAPI + Next.js 15/React 18/TS + Redis + MySQL + RabbitMQ。
- 生产模型：EfficientNet-B3（分类）+ YOLOv8（多角色检测）+ FAISS/CLIP/WD ViT/EasyOCR。

## 2. 资源约束（最高优先级）
- 本机为 macOS（Apple Silicon, MPS）。**严禁在本机拉起端到端重负载推理**
  （尤其 t2i_service 的 SD1.5+IP-Adapter，会占满 CPU/MPS 致整机卡死）。
- 验证手段仅限：py_compile / tsc / grep / 读码 / 轻量探针（如 `resource_monitor --once`）。
  真出图交给用户本机手动执行。
- t2i_service 运行于独立的 `t2i-mac` venv（torch 2.3.1 + diffusers 0.30.1），与项目主 `.venv` 隔离。

## 3. 模型基线铁律（v7 诚实基线，2026-08-10）
- **对外报数一律用 167 类契约口径**（从 171 原始类剔除 TRAIN_ONLY 不可评类）。
- **禁用 `macro_f1_all171`**：该指标含整库仅 1 张图的不可评类，是固定 2.34% 的会计折扣，无信息量。
- **评估预处理必须用 canonical**：`Resize(288) -> CenterCrop(256)`。
  legacy 的 `Resize((256,256))` 改变长宽比、与训练分布不匹配（val 上差 1.77~2.02 点，属正确性问题非性能）。
- **必须保留 v7 复算锚点**（split_hash `71b7101b47eb266579dea81bd837dded9c55e2e09c6e31124de1133abd733eeb`）：
  TEST Top-1 0.5989 / MacroF1 0.5533 / Top-5 0.7767；VAL Top-1 0.6183 / MacroF1 0.5738。
- 标签契约的权威来源是 `configs/label_contract.yaml`——改标签空间/口径时须同步更新该文件，
  禁止代码与报告各说各话。

## 4. 数据泄漏铁律
- 切分必须按 `post_id` 分组（见 `src/core/data/split_utils.py`），保证同一 post 的所有图落在同一 split。
- 严禁使用 `random_split` 做训练/测试切分（会引入同源泄漏，v6 因此虚高约 13 点）。
- 约 30 个非核心训练脚本仍用 random_split，如需重训须先统一到 split_utils。

## 5. 运维约定
- supervisord 用 `.venv` 启动：`.venv/bin/supervisord -c supervisord.conf`；
  鉴权 `-u CHANGE_ME_supervisor_admin -p CHANGE_ME_supervisor_pwd`。
- 外部依赖：Redis/RabbitMQ 已运行，MySQL 未运行 → 认证/核心库走 SQLite 三层降级（MySQL→SQLite→Memory）。
- 前端在沙箱无法 `npm run build`（safe-delete 清 .next）；需本机 `npm run dev`。
- 健康检查 `/<svc>/api/health`；api-gateway 根 `/` 返回 200；认证走表单
  `POST /api/auth/register`、`/api/auth/login`。

## 6. 日志约定（2026-08-15 整改）
- 双日志体系：老 `logging`（`unified_logger`/`log_manager` → `anime_role_detect_logs/*.log`）
  + 新 `loguru`（`logging_setup` → `logs/anime_role_detect_structured_*.jsonl`）。
- 心跳/资源采样已降级为 DEBUG，仅超阈值 WARNING 进 jsonl；
  第三方库（uvicorn/transformers/diffusers/modelscope）级别已抬到 WARNING。
- t2i_service 已加 uvicorn `log_config` 带时间戳 + `[t2i] logger` 带时间戳。
- 改动任何日志相关代码后，须 `py_compile` 校验。

## 7. 不要做
- 不要在本机跑端到端推理 / 调 `/generate` 做冒烟。
- 不要改评估口径却不更新 `configs/label_contract.yaml`。
- 不要引入 9 种数据库方言 / 元数据驱动 UI / 多 SKU 构建这类企业平台复杂度（不适用本项目）。
- 不要把 `.env` / 凭据 / 模型权重提交进 git（见 `.gitignore`）。
