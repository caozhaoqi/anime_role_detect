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
import { GenerationService, T2IRole, TrainJobStatus } from "../api/services/GenerationService";

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
  const [genInfo, setGenInfo] = useState<{ method: string; fellBack: boolean; device: string } | null>(null);
  const [genError, setGenError] = useState<string | null>(null);
  const [genProgress, setGenProgress] = useState<string>("");

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
    pollRef.current = setInterval(async () => {
      try {
        const job = await GenerationService.getJob(jobId);
        setActiveJob(job);
        if (job.status === "succeeded" || job.status === "failed") {
          stopPolling();
          setTraining(false);
          if (job.status === "succeeded") {
            // 训练成功 → 刷新角色列表（lora_ready 会变）
            loadRoles();
          }
        }
      } catch {
        // 轮询失败忽略，下一周期重试
      }
    }, 2500);
  }, [loadRoles]);

  const handleGenerate = useCallback(async () => {
    if (!selectedRole) return;
    setGenerating(true);
    setGenError(null);
    setImages([]);
    setGenInfo(null);
    setGenProgress("已提交，排队中…");
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
        onUpdate: (j) => setGenProgress(j.progress || j.status),
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
            <p className={`mt-2 text-xs ${labelCls} flex items-center space-x-1`}>
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>{genProgress || "正在生成，请稍候…"}</span>
            </p>
          )}

          {genError && (
            <div className="mt-3 flex items-start space-x-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 rounded-lg p-2">
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
                <span className="px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300">
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
              <div className="flex items-center justify-center py-10 text-sm text-gray-400">
                <Loader2 className="h-5 w-5 animate-spin mr-2" />
                {genProgress || "正在生成，请稍候…"}
              </div>
            )}
            {!generating && images.length === 0 && !genError && (
              <div className="py-10 text-center text-sm text-gray-400">
                选择角色并点击「生成图像」
              </div>
            )}
            {!generating && images.length > 0 && (
              <div className="grid grid-cols-2 gap-3">
                {images.map((src, i) => (
                  <div key={i} className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm">
                    {/* base64 data URI 直接渲染 */}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={src} alt={`generated-${i + 1}`} className="w-full h-auto object-cover" />
                  </div>
                ))}
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
              <div className="mt-3 flex items-start space-x-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 rounded-lg p-2">
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
    succeeded: { icon: CheckCircle2, cls: "text-green-500", text: "完成" },
    failed: { icon: XCircle, cls: "text-red-500", text: "失败" },
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
