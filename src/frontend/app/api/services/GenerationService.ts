import { apiClient } from '../client';

export interface T2IRole {
  role: string;
  image_count: number;
  lora_ready: boolean;
}

export interface GenerateRequest {
  role: string;
  prompt?: string;
  negative?: string;
  method?: 'ip_adapter' | 'lora';
  scale?: number;
  steps?: number;
  cfg?: number;
  num?: number;
  num_ref?: number;
  seed?: number;
  device?: string | null;
}

/** 生成结果（作业成功时由后端 result 字段返回） */
export interface GenerateResult {
  role: string;
  method: string;
  requested_method: string;
  fell_back: boolean;
  prompt: string;
  images: string[]; // base64 data URI 列表
  saved_paths: string[];
  device: string;
}

export interface TrainRequest {
  role: string;
  rank?: number;
  epochs?: number;
  resolution?: number;
  lr?: number;
  batch_size?: number;
}

export interface TrainResponse {
  success: boolean;
  job_id: string;
  role: string;
  status: string;
}

export type TrainStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export interface TrainJobStatus {
  job_id: string;
  role: string;
  status: TrainStatus;
  progress: string;
  progress_pct?: number;
  log_tail: string[];
  log_lines: number;
  created_at: number;
  finished_at: number | null;
  output_dir: string;
}

/** 生成/训练作业状态（生成作业额外带 result / error / type 字段） */
export interface GenerateJobStatus extends TrainJobStatus {
  type?: 'train' | 'generate';
  result?: GenerateResult;
  error?: string;
  progress_pct?: number;
}

/** 提交生成后即刻返回（不再同步等待出图） */
export interface SubmitGenResponse {
  success: boolean;
  job_id: string;
  role: string;
  status: string;
  message?: string;
}

/** 提交对话生成后即刻返回 */
export interface SubmitChatResponse {
  success: boolean;
  job_id?: string;
  matched_role: string | null;
  reply: string;
  available_roles?: string[];
}

/**
 * 角色图像生成服务层：封装 t2i_service 的 HTTP 接口。
 * 所有路径相对 apiClient 的 baseURL `/api`，由网关转发到 t2i 微服务：
 *   /api/t2i/roles        GET   角色列表
 *   /api/t2i/generate     POST 提交生成作业（立即返回 job_id）
 *   /api/t2i/train        POST 训练 LoRA
 *   /api/t2i/jobs/{id}    GET   作业状态（含生成结果）
 *   /api/t2i/jobs         GET   作业列表
 *   /api/t2i/chat         POST 对话生成（立即返回 job_id）
 *
 * 生成是长任务（首次加载 SD1.5+IP-Adapter 需数十秒~数分钟），统一采用
 * "提交即返回 job_id + 轮询" 模式，避免长连接超时与 UI 卡死。
 */
/**
 * 同一 jobId 只允许一个轮询循环在跑（去重）。
 * 当多个入口（生成面板 / 对话 / React StrictMode 双调用）同时 poll 同一作业时，
 * 复用同一个 Promise，避免重复请求把网关日志刷成"一直转发"。
 */
const _activePolls = new Map<string, Promise<GenerateJobStatus>>();

/** 连续网络不可达达到该次数即判定 t2i-service 已崩溃/重启中，快速失败而非死轮到 30 分钟 */
const POLL_FAIL_FAST_THRESHOLD = 4;

export class GenerationService {
  static async listRoles(): Promise<T2IRole[]> {
    try {
      const res = await apiClient.get<{ success: boolean; roles: T2IRole[] }>('/t2i/roles');
      return res.data.roles ?? [];
    } catch {
      return [];
    }
  }

  /** 提交生成作业，立即返回 job_id（不等待出图） */
  static async generate(req: GenerateRequest): Promise<SubmitGenResponse> {
    const res = await apiClient.post<SubmitGenResponse>('/t2i/generate', req);
    return res.data;
  }

  static async train(req: TrainRequest): Promise<TrainResponse> {
    const res = await apiClient.post<TrainResponse>('/t2i/train', req);
    return res.data;
  }

  static async getJob(jobId: string): Promise<GenerateJobStatus> {
    const res = await apiClient.get<{ success: boolean } & GenerateJobStatus>(`/t2i/jobs/${jobId}`);
    return res.data;
  }

  static async listJobs(): Promise<GenerateJobStatus[]> {
    const res = await apiClient.get<{ success: boolean; jobs: GenerateJobStatus[] }>('/t2i/jobs');
    return res.data.jobs ?? [];
  }

  /** 对话生成：提交任务并立即返回 job_id（命中的话），未命中角色则无 job_id */
  static async chat(
    message: string,
    method: 'ip_adapter' | 'lora' = 'ip_adapter',
    num = 1
  ): Promise<SubmitChatResponse> {
    const res = await apiClient.post<SubmitChatResponse>('/t2i/chat', { message, method, num });
    return res.data;
  }

  /**
   * 轮询作业直到 succeeded/failed。每 intervalMs 拉一次状态，超时 timeoutMs 抛错。
   * onUpdate 可在每次拉取时拿到最新进度（用于 UI 展示"排队中/生成中…"）。
   *
   * 健壮性：
   * - 去重：同一 jobId 并发只跑一个循环（见 _activePolls）。
   * - 快速失败：t2i-service 崩溃/重启中时 getJob 会持续连接失败；连续
   *   POLL_FAIL_FAST_THRESHOLD 次不可达即判定服务已挂，立即 reject，避免前端
   *   静默死轮到 30 分钟、把网关日志刷成"一直转发"。
   */
  static pollJob(
    jobId: string,
    opts: { intervalMs?: number; timeoutMs?: number; onUpdate?: (job: GenerateJobStatus) => void } = {}
  ): Promise<GenerateJobStatus> {
    const existing = _activePolls.get(jobId);
    if (existing) return existing;

    const intervalMs = opts.intervalMs ?? 2500;
    // 兜底超时设为 30 分钟：LoRA 训练在 Mac CPU 上单步可达数百秒、整轮常 >10 分钟，
    // 原 10 分钟兜底会在作业完成前放弃 → 误报"超时"。运行中的作业不应被墙钟超时打断。
    const timeoutMs = opts.timeoutMs ?? 1800000; // 30 分钟兜底
    const promise = new Promise<GenerateJobStatus>((resolve, reject) => {
      const start = Date.now();
      let timer: ReturnType<typeof setInterval> | null = null;
      let failStreak = 0; // 连续网络不可达计数
      const cleanup = () => {
        _activePolls.delete(jobId);
        if (timer) {
          clearInterval(timer);
          timer = null;
        }
      };
      const tick = async () => {
        try {
          const job = await GenerationService.getJob(jobId);
          failStreak = 0; // 成功拿到状态 → 重置失败计数
          opts.onUpdate?.(job);
          if (job.status === 'succeeded') {
            cleanup();
            resolve(job);
            return;
          }
          if (job.status === 'failed') {
            cleanup();
            reject(new Error(job.error || '生成任务失败'));
            return;
          }
        } catch (err: any) {
          // 作业不存在（服务重启清掉了内存中的作业线程）→ 立即失败，避免空转
          const status = err?.response?.status;
          const msg = String(err?.message || "");
          if (status === 404 || msg.includes("不存在") || msg.includes("not found")) {
            cleanup();
            reject(new Error("生成作业已失效（服务可能重启过），请重新提交"));
            return;
          }
          // 网关返回 503 = 上游 t2i-service 明确不可用（已崩溃/重启中）。这是确定性故障，
          // 不必再等 4 次，立即快速失败，避免继续把网关日志刷成"一直转发"。
          if (status === 503) {
            cleanup();
            reject(new Error("生成服务不可用（t2i-service 已停止或正在重启），请检查服务状态后重试"));
            return;
          }
          // 网络不可达（服务崩溃 / 重启中 / 连接拒绝）：累计失败次数，超阈值快速失败
          failStreak += 1;
          if (failStreak >= POLL_FAIL_FAST_THRESHOLD) {
            cleanup();
            reject(new Error("生成服务无响应（t2i-service 可能已崩溃或正在重启），请检查服务状态后重试"));
            return;
          }
        }
        if (Date.now() - start > timeoutMs) {
          cleanup();
          reject(new Error('生成轮询超时，请稍后在「图像生成」页重试'));
        }
      };
      timer = setInterval(tick, intervalMs);
      tick();
    });
    _activePolls.set(jobId, promise);
    return promise;
  }
}
