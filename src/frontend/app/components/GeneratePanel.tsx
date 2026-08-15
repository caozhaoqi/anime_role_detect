"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Wand2,
  Sparkles,
  Cpu,
  RefreshCw,
  Loader2,
  CheckCircle2,
  XCircle,
  Images,
  AlertTriangle,
} from "lucide-react";
import { GenerationService, T2IRole, TrainJobStatus, GenerateJobStatus } from "../api/services/GenerationService";

interface GeneratePanelProps {
  darkMode: boolean;
}

const METHOD_LABELS: Record<string, string> = {
  ip_adapter: "IP-Adapter（免训练，秒级）",
  lora: "LoRA（需先训练，一致性更高）",
};

export default function GeneratePanel({ darkMode }: GeneratePanelProps) {
  const [roles, setRoles] = useState<T2IRole[]>([]);
  const [rolesLoading, setRolesLoading] = useState(true);
  const [selectedRole, setSelectedRole] = useState<string>("");

  // 生成参数
  const [prompt, setPrompt] = useState("");
  const [method, setMethod] = useState<"ip_adapter" | "lora">("ip_adapter");
  const [scale, setScale] = useState(0.6);
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(7.5);
  const [num, setNum] = useState(1);
  const [numRef, setNumRef] = useState(12);
  const [seed, setSeed] = useState(42);

  // 生成结果
  const [generating, setGenerating] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  const [viewIdx, setViewIdx] = useState(0);
  const [genInfo, setGenInfo] = useState<{ method: string; fellBack: boolean; device: string } | null>(null);
  const [genError, setGenError] = useState<string | null>(null);
  const [genProgress, setGenProgress] = useState<string>("");
  const [genJob, setGenJob] = useState<GenerateJobStatus | null>(null);

  // 训练状态
  const [training, setTraining] = useState(false);
  const [activeJob, setActiveJob] = useState<TrainJobStatus | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadRoles = useCallback(async () => {
    setRolesLoading(true);
    const list = await GenerationService.listRoles();
    setRoles(list);
    setRolesLoading(false);
    if (!selectedRole && list.length > 0) {
      setSelectedRole(list[0].role);
    }
  }, [selectedRole]);

  useEffect(() => {
    loadRoles();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const currentRole = roles.find((r) => r.role === selectedRole) || null;

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const pollJob = useCallback((jobId: string) => {
    stopPolling();
    let failStreak = 0;
    const FAIL_THRESHOLD = 4; // 连续 4 次网络不可达(~10s) → 判定 t2i-service 已挂，停止死轮
    pollRef.current = setInterval(async () => {
      try {
        const job = await GenerationService.getJob(jobId);
        failStreak = 0; // 成功拿到状态 → 重置失败计数
        setActiveJob(job);
        if (job.status === "succeeded" || job.status === "failed") {
          stopPolling();
          setTraining(false);
          if (job.status === "succeeded") {
            // 训练成功 → 刷新角色列表（lora_ready 会变）
            loadRoles();
          }
        }
      } catch (err: any) {
        const status = err?.response?.status;
        const msg = String(err?.message || "");
        // 作业被清除（服务重启）→ 立即终止，避免静默死轮
        if (status === 404 || msg.includes("不存在") || msg.includes("not found")) {
          stopPolling();
          setTraining(false);
          setTrainError("训练作业已失效（t2i-service 可能重启过），请重新提交");
          return;
        }
        // 网络不可达（服务崩溃 / 重启中）：累计失败次数，超阈值停止轮询并报错
        failStreak += 1;
        if (failStreak >= FAIL_THRESHOLD) {
          stopPolling();
          setTraining(false);
          setTrainError("训练服务无响应（t2i-service 可能已崩溃或正在重启），请检查服务状态后重试");
        }
      }
    }, 2500);
  }, [loadRoles]);

  const handleGenerate = useCallback(async () => {
    if (!selectedRole) return;
    setGenerating(true);
    setGenError(null);
    setImages([]);
    setViewIdx(0);
    setGenInfo(null);
    setGenProgress("已提交，排队中…");
    setGenJob(null);
    try {
      // 1) 提交作业，立即拿到 job_id（请求秒回，不阻塞）
      const submit = await GenerationService.generate({
        role: selectedRole,
        prompt: prompt.trim() || undefined,
        method,
        scale,
        steps,
        cfg,
        num,
        num_ref: numRef,
        seed,
        device: "auto",
      });
      // 2) 轮询作业直到完成（后台线程推理，UI 始终可响应）
      const job = await GenerationService.pollJob(submit.job_id, {
        onUpdate: (j) => {
          setGenJob(j);
          setGenProgress(j.progress || j.status);
        },
      });
      const result = job.result!;
      setImages(result.images ?? []);
      setGenInfo({
        method: result.method,
        fellBack: result.fell_back,
        device: result.device,
      });
      setGenProgress("");
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail ||
        e?.response?.data?.message ||
        e?.message ||
        "生成失败";
      setGenError(String(msg));
      setGenProgress("");
    } finally {
      setGenerating(false);
    }
  }, [selectedRole, prompt, method, scale, steps, cfg, num, numRef, seed]);

  const handleTrain = useCallback(async () => {
    if (!selectedRole) return;
    setTraining(true);
    setTrainError(null);
    try {
      const res = await GenerationService.train({ role: selectedRole });
      setActiveJob({
        job_id: res.job_id,
        role: res.role,
        status: "running",
        progress: "已提交训练任务…",
        log_tail: [],
        log_lines: 0,
        created_at: Date.now() / 1000,
        finished_at: null,
        output_dir: "",
      });
      pollJob(res.job_id);
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail ||
        e?.response?.data?.message ||
        e?.message ||
        "提交训练失败";
      setTrainError(String(msg));
      setTraining(false);
    }
  }, [selectedRole, pollJob]);

  const cardBase = darkMode
    ? "bg-gray-800 border-gray-700"
    : "bg-white border-gray-200";
  const labelCls = darkMode ? "text-gray-300" : "text-gray-600";
  const inputCls = `w-full px-3 py-2 rounded-lg border text-sm transition-all focus:outline-none focus:ring-2 focus:ring-generate/50 ${
    darkMode ? "bg-gray-700 border-gray-600 text-white" : "bg-gray-50 border-gray-200 text-gray-900"
  }`;
  const accentBtn = `inline-flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg font-medium text-white bg-generate hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed`;

  return (
    <div className="animate-fade-in">
      {/* 进度条条纹动画（仅定义一次；genprogress-stripe 类供 GenProgressBlock 使用） */}
      <style>{`
        @keyframes genprogress-stripe { 0% { background-position: 0 0; } 100% { background-position: 32px 0; } }
        .genprogress-stripe {
          background-image: linear-gradient(45deg, rgba(255,255,255,0.28) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.28) 50%, rgba(255,255,255,0.28) 75%, transparent 75%, transparent);
          background-size: 32px 32px;
          animation: genprogress-stripe 0.9s linear infinite;
        }
      `}</style>
      <div className="flex items-center space-x-3 mb-5">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${darkMode ? "bg-generate/20 text-generate" : "bg-generate text-white"}`}>
          <Wand2 className="h-5 w-5" />
        </div>
        <div>
          <h1 className="text-xl md:text-2xl font-bold">角色图像生成</h1>
          <p className={`text-xs ${labelCls}`}>
            选择数据集中角色 → 基于参考图生成同人图；可训练 LoRA 提升一致性
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* ---- 左侧：角色选择与生成 ---- */}
        <div className={`rounded-2xl border shadow-sm p-4 md:p-5 ${cardBase}`}>
          <h2 className="font-semibold mb-3 flex items-center space-x-2">
            <Sparkles className="h-4 w-4 text-generate" />
            <span>生成设置</span>
          </h2>

          {/* 角色选择 */}
          <div className="mb-3">
            <label className={`block text-xs font-medium mb-1 ${labelCls}`}>角色（数据集）</label>
            {rolesLoading ? (
              <div className={`${inputCls} flex items-center space-x-2 opacity-70`}>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>加载角色列表中…</span>
              </div>
            ) : (
              <select
                value={selectedRole}
                onChange={(e) => setSelectedRole(e.target.value)}
                className={inputCls}
              >
                {roles.length === 0 && <option value="">（无可用角色）</option>}
                {roles.map((r) => (
                  <option key={r.role} value={r.role}>
                    {r.role} · {r.image_count} 张
                    {r.lora_ready ? " · LoRA✓" : ""}
                  </option>
                ))}
              </select>
            )}
            {currentRole && (
              <p className={`mt-1 text-xs ${labelCls}`}>
                {currentRole.lora_ready
                  ? "该角色已训练 LoRA，可选 LoRA 方法获得更高一致性"
                  : "尚未训练 LoRA，使用 LoRA 方法将自动回退 IP-Adapter"}
              </p>
            )}
          </div>

          {/* 方法 */}
          <div className="mb-3">
            <label className={`block text-xs font-medium mb-1 ${labelCls}`}>生成方法</label>
            <div className="grid grid-cols-2 gap-2">
              {(["ip_adapter", "lora"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMethod(m)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium border transition-all ${
                    method === m
                      ? "border-generate text-generate bg-generate/10"
                      : darkMode
                      ? "border-gray-600 text-gray-300 hover:border-gray-500"
                      : "border-gray-200 text-gray-600 hover:border-gray-300"
                  }`}
                >
                  {METHOD_LABELS[m]}
                </button>
              ))}
            </div>
          </div>

          {/* Prompt */}
          <div className="mb-3">
            <label className={`block text-xs font-medium mb-1 ${labelCls}`}>
              提示词（可选，留空用默认）
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={2}
              placeholder="例如：smiling, cherry blossoms background, summer uniform"
              className={inputCls}
            />
          </div>

          {/* 参数 */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div>
              <label className={`block text-xs font-medium mb-1 ${labelCls}`}>参考图数量 {numRef}</label>
              <input type="range" min={1} max={20} value={numRef} onChange={(e) => setNumRef(Number(e.target.value))} className="w-full accent-generate" />
            </div>
            <div>
              <label className={`block text-xs font-medium mb-1 ${labelCls}`}>生成张数 {num}</label>
              <input type="range" min={1} max={4} value={num} onChange={(e) => setNum(Number(e.target.value))} className="w-full accent-generate" />
            </div>
            <div>
              <label className={`block text-xs font-medium mb-1 ${labelCls}`}>IP 强度 {scale}</label>
              <input type="range" min={0} max={1} step={0.05} value={scale} onChange={(e) => setScale(Number(e.target.value))} className="w-full accent-generate" />
            </div>
            <div>
              <label className={`block text-xs font-medium mb-1 ${labelCls}`}>步数 {steps}</label>
              <input type="range" min={10} max={50} step={5} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className="w-full accent-generate" />
            </div>
            <div>
              <label className={`block text-xs font-medium mb-1 ${labelCls}`}>CFG {cfg}</label>
              <input type="range" min={1} max={15} step={0.5} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} className="w-full accent-generate" />
            </div>
            <div>
              <label className={`block text-xs font-medium mb-1 ${labelCls}`}>随机种子 {seed}</label>
              <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className={inputCls} />
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={!selectedRole || generating}
            className={accentBtn + " w-full"}
          >
            {generating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>生成中…</span>
              </>
            ) : (
              <>
                <Wand2 className="h-4 w-4" />
                <span>生成图像</span>
              </>
            )}
          </button>

          {generating && (
            <div className="mt-3">
              <GenProgressBlock job={genJob} darkMode={darkMode} />
            </div>
          )}

          {genError && (
            <div className="mt-3 flex items-start space-x-2 text-xs text-danger bg-danger/10 dark:bg-danger/20 rounded-lg p-2">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{genError}</span>
            </div>
          )}

          {genInfo && (
            <div className={`mt-3 text-xs ${labelCls} flex flex-wrap items-center gap-2`}>
              <span className={`px-2 py-0.5 rounded-full ${darkMode ? "bg-generate/20 text-generate" : "bg-generate/10 text-generate"}`}>
                方法：{genInfo.method}
              </span>
              {genInfo.fellBack && (
                <span className="px-2 py-0.5 rounded-full bg-generate/10 text-generate">
                  已回退 IP-Adapter
                </span>
              )}
              <span>设备：{genInfo.device}</span>
            </div>
          )}
        </div>

        {/* ---- 右侧：结果 + 训练 ---- */}
        <div className="space-y-4">
          {/* 生成结果 */}
          <div className={`rounded-2xl border shadow-sm p-4 md:p-5 ${cardBase}`}>
            <h2 className="font-semibold mb-3 flex items-center space-x-2">
              <Images className="h-4 w-4 text-generate" />
              <span>生成结果</span>
            </h2>
            {generating && (
              <div className="py-8">
                <GenProgressBlock job={genJob} darkMode={darkMode} />
              </div>
            )}
            {!generating && images.length === 0 && !genError && (
              <div className="py-10 text-center text-sm text-gray-400">
                选择角色并点击「生成图像」
              </div>
            )}
            {!generating && images.length > 0 && (
              <div className="space-y-3">
                {/* 一次生成多张时只展示一张（首图），下方缩略图可切换查看其余 */}
                <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={images[viewIdx]} alt={`generated-${viewIdx + 1}`} className="w-full h-auto object-cover" />
                </div>
                {images.length > 1 && (
                  <>
                    <div className={`flex items-center justify-between text-xs ${labelCls}`}>
                      <span>共 {images.length} 张 · 当前第 {viewIdx + 1} 张</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {images.map((src, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => setViewIdx(i)}
                          aria-label={`查看第 ${i + 1} 张`}
                          className={`h-14 w-14 rounded-lg overflow-hidden border-2 transition ${
                            i === viewIdx
                              ? "border-generate"
                              : "border-transparent opacity-70 hover:opacity-100"
                          }`}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={src} alt={`thumb-${i + 1}`} className="h-full w-full object-cover" />
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* 训练区块 */}
          <div className={`rounded-2xl border shadow-sm p-4 md:p-5 ${cardBase}`}>
            <h2 className="font-semibold mb-3 flex items-center space-x-2">
              <Cpu className="h-4 w-4 text-generate" />
              <span>训练 LoRA</span>
            </h2>
            <p className={`text-xs ${labelCls} mb-3`}>
              为「{selectedRole || "—"}」基于其参考图训练 LoRA（输出至 <code className="px-1 rounded bg-gray-100 dark:bg-gray-700">outputs/t2i_lora/{selectedRole || "角色"}_v1/</code>）。Mac CPU 训练较慢，请耐心等待。
            </p>
            <button
              onClick={handleTrain}
              disabled={!selectedRole || training}
              className={accentBtn + " w-full"}
            >
              {training ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>训练中…</span>
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4" />
                  <span>开始训练</span>
                </>
              )}
            </button>

            {trainError && (
              <div className="mt-3 flex items-start space-x-2 text-xs text-danger bg-danger/10 dark:bg-danger/20 rounded-lg p-2">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                <span>{trainError}</span>
              </div>
            )}

            {activeJob && (
              <div className="mt-3">
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-xs font-medium ${labelCls}`}>
                    作业 {activeJob.job_id}
                  </span>
                  <StatusBadge status={activeJob.status} darkMode={darkMode} />
                </div>
                {activeJob && activeJob.status !== "succeeded" && (
                  <GenProgressBlock job={activeJob} darkMode={darkMode} kind="train" />
                )}
                {activeJob.progress && (
                  <p className={`text-xs mb-2 ${labelCls}`}>{activeJob.progress}</p>
                )}
                {activeJob.log_tail.length > 0 && (
                  <pre className={`text-[10px] leading-tight max-h-40 overflow-y-auto rounded-lg p-2 ${
                    darkMode ? "bg-gray-900 text-gray-300" : "bg-gray-100 text-gray-700"
                  }`}>
                    {activeJob.log_tail.slice(-20).join("\n")}
                  </pre>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status, darkMode }: { status: TrainJobStatus["status"]; darkMode: boolean }) {
  const map: Record<TrainJobStatus["status"], { icon: any; cls: string; text: string }> = {
    queued: { icon: Loader2, cls: "text-gray-500", text: "排队中" },
    running: { icon: Loader2, cls: "text-generate", text: "训练中" },
    succeeded: { icon: CheckCircle2, cls: "text-success", text: "完成" },
    failed: { icon: XCircle, cls: "text-danger", text: "失败" },
  };
  const m = map[status];
  const Icon = m.icon;
  return (
    <span className={`inline-flex items-center space-x-1 text-xs ${m.cls}`}>
      <Icon className={`h-3.5 w-3.5 ${status === "running" || status === "queued" ? "animate-spin" : ""}`} />
      <span>{m.text}</span>
    </span>
  );
}

/**
 * 由作业状态 + 后端 progress_pct（生成按阶段、训练按 epoch/step 实时赋值）推断
 * "明确的"进度（百分比 + 阶段标签）。后端已给真实百分比时优先采用；仅在后端暂无
 * 精确值时，按 progress 文本把生成细分为阶段（加载/生成/收尾）兜底。
 * kind 区分生成/训练，仅影响运行中的默认标签文案。
 */
function computeProgress(
  job: { status: string; progress?: string; progress_pct?: number } | null,
  kind: "generate" | "train" = "generate"
): { pct: number; label: string; active: boolean; failed: boolean } {
  const backendPct = typeof job?.progress_pct === "number" ? job.progress_pct : 0;
  const txt = String(job?.progress || "").toLowerCase();
  if (job?.status === "succeeded") return { pct: 100, label: "完成", active: false, failed: false };
  if (job?.status === "failed") return { pct: Math.max(backendPct, 90), label: "失败", active: false, failed: true };
  if (job?.status === "queued") return { pct: backendPct || 8, label: "排队中", active: true, failed: false };

  // running：优先采用后端真实百分比（训练=epoch/step、生成=逐步骤度均已实时赋值）
  if (backendPct > 0) {
    let label = kind === "train" ? "训练中" : "生成中";
    if (kind === "generate") {
      if (backendPct < 50) label = "加载/准备模型";
      else if (backendPct >= 95) label = "收尾中";
      else label = "推理中";
    }
    return { pct: backendPct, label, active: true, failed: false };
  }
  // 后端暂无精确百分比：生成按阶段文本细分，训练兜底为"训练中"
  let stagePct = 72;
  let label = kind === "train" ? "训练中" : "生成中";
  if (kind === "generate") {
    if (txt.includes("加载")) { stagePct = 40; label = "加载模型"; }
    else if (txt.includes("收尾") || backendPct >= 88) { stagePct = 90; label = "收尾中"; }
    else if (txt.includes("生成") || txt.includes("推理")) { stagePct = 72; label = "生成中"; }
  }
  return { pct: Math.max(stagePct, backendPct), label, active: true, failed: false };
}

/** 进度条：百分比 + 阶段标签 + 进行中条纹动画；失败显红并停止动画。
 *  生成与训练共用此组件，保证两处进度条视觉完全一致。
 *  creep：后端在生成长任务中只给阶段性百分比（如生成固定 45%），为避免"卡在 45%"
 *  的观感，运行中让显示值平滑爬升（封顶 94），真实完成/失败时吸附到终值。 */
function GenProgressBlock({
  job,
  darkMode,
  kind = "generate",
}: {
  job: TrainJobStatus | GenerateJobStatus | null;
  darkMode: boolean;
  kind?: "generate" | "train";
}) {
  const { pct, label, active, failed } = computeProgress(job, kind);
  const clamped = Math.max(0, Math.min(100, Math.round(pct)));
  const labelCls = darkMode ? "text-gray-300" : "text-gray-600";
  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1.5">
        <span className={`text-xs font-medium flex items-center space-x-1.5 ${labelCls}`}>
          {active && <Loader2 className="h-3.5 w-3.5 animate-spin text-generate" />}
          <span>{label}</span>
        </span>
        <span className={`text-xs font-semibold tabular-nums ${failed ? "text-danger" : "text-generate"}`}>{clamped}%</span>
      </div>
      <div className={`w-full h-2.5 rounded-full overflow-hidden ${darkMode ? "bg-gray-700" : "bg-gray-200"}`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${failed ? "bg-danger" : "bg-generate"} ${active && !failed ? "genprogress-stripe" : ""}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
      {job?.progress && job.status !== "succeeded" && (
        <p className={`mt-1 text-xs ${labelCls} truncate`}>{job.progress}</p>
      )}
    </div>
  );
}
