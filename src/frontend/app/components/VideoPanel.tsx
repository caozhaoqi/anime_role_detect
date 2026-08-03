"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Video, Play, Pause, X, Clock, AlertTriangle, CheckCircle, Download, Loader2, UploadCloud, Film, SlidersHorizontal } from "lucide-react";
import axios from "axios";
import EmptyState from "./EmptyState";

interface VideoResult {
  timestamp: number;
  frame_number: number;
  roles: {
    role: string;
    similarity: number;
    box?: { x: number; y: number; w: number; h: number };
  }[];
}

interface VideoPanelProps {
  darkMode: boolean;
  accessToken?: string;
}

export default function VideoPanel({ darkMode, accessToken }: VideoPanelProps) {
  const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<VideoResult[]>([]);
  const [frameInterval, setFrameInterval] = useState(1.0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [recognitionMode, setRecognitionMode] = useState("search");
  const [modelName, setModelName] = useState("efficientnet_b3_loli_optimized_v2_20260529_133654");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [outputVideo, setOutputVideo] = useState(false);
  const [resultVideoUrl, setResultVideoUrl] = useState<string | null>(null);

  // 进度相关状态
  const [taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [showSuccess, setShowSuccess] = useState(false);
  const [showError, setShowError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  // 拖拽上传视频
  const applyVideoFile = useCallback((file: File) => {
    setSelectedVideo(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setVideoPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    setResults([]);
    setResultVideoUrl(null);
    setShowSuccess(false);
    setShowError(null);
    setProgress(0);
  }, []);

  // 清理轮询
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (selectedVideo && videoPreview) {
      const video = document.createElement("video");
      video.src = videoPreview;
      video.onloadedmetadata = () => {
        setDuration(video.duration);
      };
    }
  }, [selectedVideo, videoPreview]);

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      applyVideoFile(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("video/")) {
      applyVideoFile(file);
    }
  };

  const removeVideo = useCallback(() => {
    setSelectedVideo(null);
    setVideoPreview(null);
    setResults([]);
    setCurrentTime(0);
    setDuration(0);
    setResultVideoUrl(null);
    setProgress(0);
    setProgressMessage("");
    setShowSuccess(false);
    setShowError(null);
    setTaskId(null);
  }, []);

  // 轮询任务状态
  const startPolling = useCallback((tid: string) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    pollTimerRef.current = setInterval(async () => {
      try {
        const headers: any = {};
        if (accessToken) headers["Authorization"] = `Bearer ${accessToken}`;

        const resp = await axios.get(`/api/video/task/${tid}`, { headers });
        const data = resp.data;

        if (data.success) {
          setProgress(data.progress);
          setProgressMessage(data.message || "");

          if (data.status === "completed") {
            clearInterval(pollTimerRef.current!);
            pollTimerRef.current = null;
            setIsProcessing(false);
            setShowSuccess(true);

            // 取结果
            if (data.result) {
              setResults(data.result.results || []);
              if (data.result.result_video_url) {
                setResultVideoUrl(data.result.result_video_url);
              }
            }

            // 3秒后自动隐藏成功提示
            setTimeout(() => setShowSuccess(false), 5000);
          } else if (data.status === "failed") {
            clearInterval(pollTimerRef.current!);
            pollTimerRef.current = null;
            setIsProcessing(false);
            setShowError(data.error || "视频处理失败");
          }
        }
      } catch (e) {
        console.error("轮询任务状态失败:", e);
      }
    }, 1000); // 每秒轮询
  }, [accessToken]);

  const handleRecognize = useCallback(async () => {
    if (!selectedVideo || isProcessing) return;

    setIsProcessing(true);
    setResults([]);
    setResultVideoUrl(null);
    setShowSuccess(false);
    setShowError(null);
    setProgress(0);
    setProgressMessage("准备中...");

    try {
      const formData = new FormData();
      formData.append("file", selectedVideo);

      const params = new URLSearchParams({
        frame_interval: frameInterval.toString(),
        confidence_threshold: confidenceThreshold.toString(),
        recognition_mode: recognitionMode,
        model_name: modelName,
      });

      const headers: any = {};
      if (accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
      }

      // 根据是否勾选"生成标注视频"选择不同的API端点
      if (outputVideo) {
        // 异步任务模式：提交任务 → 轮询进度
        const response = await axios.post(
          `/api/video/recognize-with-overlay?${params}`,
          formData,
          { headers }
        );

        if (response.data.success && response.data.task_id) {
          const tid = response.data.task_id;
          setTaskId(tid);
          startPolling(tid);
        } else {
          setIsProcessing(false);
          setShowError(response.data.error || "提交任务失败");
        }
      } else {
        // 非标注模式保持同步
        const response = await axios.post(`/api/video/recognize?${params}`, formData, { headers });

        if (response.data.success) {
          const results = response.data.data?.results || response.data.results || [];
          setResults(results);
          setShowSuccess(true);
          setTimeout(() => setShowSuccess(false), 5000);
        } else {
          setShowError(response.data.error || "识别失败");
        }
        setIsProcessing(false);
      }
    } catch (error: any) {
      console.error("视频识别失败:", error);
      setIsProcessing(false);
      setShowError(error?.response?.data?.error || error?.message || "视频识别失败");
    }
  }, [selectedVideo, frameInterval, confidenceThreshold, recognitionMode, modelName, isProcessing, accessToken, outputVideo, startPolling]);

  const handleTimeUpdate = (e: React.ChangeEvent<HTMLVideoElement>) => {
    setCurrentTime(e.target.currentTime);
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // 进度栏样式
  const progressBarBg = darkMode ? "bg-gray-700" : "bg-gray-200";
  const progressBarFill = outputVideo
    ? "bg-gradient-to-r from-green-500 to-emerald-500"
    : "bg-gradient-to-r from-blue-500 to-indigo-500";

  return (
    <div className={`${darkMode ? "bg-gray-800" : "bg-white"} rounded-xl shadow-lg border ${darkMode ? "border-gray-700" : "border-gray-200"} overflow-hidden animate-fade-in`}>
      {/* 标题栏 */}
      <div className={`p-4 border-b ${darkMode ? "border-gray-700" : "border-gray-200"}`}>
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-green-100 dark:bg-green-900/50 flex items-center justify-center">
            <Video className="h-4.5 w-4.5 text-green-500" />
          </div>
          <div>
            <h2 className="text-lg font-semibold leading-tight">视频实时识别</h2>
            <p className="text-xs text-gray-500 mt-0.5">上传视频，抽帧识别画面角色</p>
          </div>
        </div>
      </div>

      {/* 成功提示横幅 */}
      {showSuccess && (
        <div className="mx-4 mt-4 px-4 py-3 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-700 rounded-lg flex items-center space-x-3 animate-pulse">
          <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
          <div className="text-sm text-green-700 dark:text-green-300 font-medium">
            识别完成{outputVideo ? "，标注视频已生成" : ""}！
          </div>
        </div>
      )}

      {/* 错误提示横幅 */}
      {showError && (
        <div className="mx-4 mt-4 px-4 py-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-lg flex items-center space-x-3">
          <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0" />
          <div className="text-sm text-red-700 dark:text-red-300 font-medium flex-1">{showError}</div>
          <button onClick={() => setShowError(null)} className="text-red-400 hover:text-red-600">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* 内容区 */}
      <div className="p-4 md:p-5">
        {/* 视频上传区（拖拽） */}
        {!videoPreview && (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-xl p-8 md:p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 mb-4 ${
              isDragging
                ? "border-green-500 bg-green-50 dark:bg-green-900/20 scale-[1.01]"
                : darkMode
                ? "border-gray-600 hover:border-green-500 hover:bg-gray-700/40"
                : "border-gray-300 hover:border-green-400 hover:bg-green-50/40"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleVideoSelect}
              className="hidden"
            />
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-3 transition-transform ${isDragging ? "scale-110" : ""} ${darkMode ? "bg-gray-700 text-green-400" : "bg-green-50 text-green-500"}`}>
              <UploadCloud className="h-8 w-8" />
            </div>
            <p className="font-medium text-sm md:text-base">点击选择或拖拽视频到此处</p>
            <p className={`text-xs mt-1.5 ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
              支持 MP4 / WebM / MOV，≤ 200MB
            </p>
          </div>
        )}

        {/* 参数设置 */}
        <div className={`p-3 rounded-lg border mb-4 ${darkMode ? "bg-gray-700/50 border-gray-600" : "bg-gray-50 border-gray-200"}`}>
          <div className="flex items-center space-x-2 mb-3">
            <SlidersHorizontal className={`h-4 w-4 ${darkMode ? "text-green-400" : "text-green-500"}`} />
            <span className="text-sm font-medium">识别参数</span>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                抽帧间隔: <span className={darkMode ? "text-green-400" : "text-green-600"}>{frameInterval}s</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="10"
                step="0.1"
                value={frameInterval}
                onChange={(e) => setFrameInterval(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                置信度阈值: <span className={darkMode ? "text-green-400" : "text-green-600"}>{(confidenceThreshold * 100).toFixed(0)}%</span>
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
              />
            </div>
          </div>
        </div>

        {/* 识别模式选择 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">识别模式</label>
          <div className="flex space-x-3">
            <button
              onClick={() => setRecognitionMode("search")}
              className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                recognitionMode === "search"
                  ? "bg-green-500 text-white border-green-500"
                  : darkMode
                  ? "bg-gray-700 border-gray-600 text-gray-300 hover:border-green-500"
                  : "bg-white border-gray-300 text-gray-700 hover:border-green-500"
              }`}
            >
              <div className="font-medium">搜图模式</div>
              <div className="text-xs mt-1 opacity-80">快速，适合实时处理</div>
            </button>
            <button
              onClick={() => setRecognitionMode("inference")}
              className={`flex-1 px-4 py-2 rounded-lg border transition-all ${
                recognitionMode === "inference"
                  ? "bg-blue-500 text-white border-blue-500"
                  : darkMode
                  ? "bg-gray-700 border-gray-600 text-gray-300 hover:border-blue-500"
                  : "bg-white border-gray-300 text-gray-700 hover:border-blue-500"
              }`}
            >
              <div className="font-medium">模型推理</div>
              <div className="text-xs mt-1 opacity-80">更准确，速度较慢</div>
            </button>
          </div>
        </div>

        {/* 生成标注视频选项 */}
        <div className="mb-4">
          <label className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={outputVideo}
              onChange={(e) => setOutputVideo(e.target.checked)}
              className="w-4 h-4 text-green-500 border-gray-300 rounded focus:ring-green-500"
            />
            <div>
              <span className="text-sm font-medium">生成标注结果视频</span>
              <p className={`text-xs mt-0.5 ${darkMode ? "text-gray-400" : "text-gray-500"}`}>
                在视频帧上绘制识别的角色名和置信度，可下载对比识别效果
              </p>
            </div>
          </label>
        </div>

        {/* 视频预览 */}
        {videoPreview && (
          <div className={`mb-4 relative ${darkMode ? "bg-gray-900" : "bg-gray-900"} rounded-lg overflow-hidden aspect-video`}>
            <video
              ref={videoRef}
              src={videoPreview}
              className="w-full h-full"
              controls={false}
              onTimeUpdate={handleTimeUpdate}
              onClick={togglePlay}
            />

            {/* 播放控制覆盖层 */}
            <div className="absolute inset-0 flex items-center justify-center bg-black/30">
              <button
                onClick={togglePlay}
                className="p-4 bg-white/20 backdrop-blur-sm rounded-full text-white hover:bg-white/30 transition-colors"
              >
                {isPlaying ? <Pause className="h-8 w-8" /> : <Play className="h-8 w-8" />}
              </button>
            </div>

            {/* 时间显示 */}
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
              <div className="flex items-center justify-between text-white text-sm">
                <span>{formatTime(currentTime)}</span>
                <span>{formatTime(duration)}</span>
              </div>
              <input
                type="range"
                min="0"
                max={duration || 1}
                value={currentTime}
                onChange={(e) => {
                  const time = Number(e.target.value);
                  setCurrentTime(time);
                  if (videoRef.current) {
                    videoRef.current.currentTime = time;
                  }
                }}
                className="w-full h-1 bg-white/30 rounded-full appearance-none cursor-pointer mt-1"
              />
            </div>

            {/* 关闭按钮 */}
            <button
              onClick={removeVideo}
              className="absolute top-2 right-2 p-1 bg-black/50 rounded-full text-white hover:bg-black/70 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* 进度显示（异步任务模式下） */}
        {isProcessing && outputVideo && (
          <div className="mb-4 p-4 rounded-lg border border-green-200 dark:border-green-700 bg-green-50/50 dark:bg-green-900/20">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 text-green-500 animate-spin" />
                <span className="text-sm font-medium text-green-700 dark:text-green-300">
                  {progressMessage || "处理中..."}
                </span>
              </div>
              <span className="text-sm font-mono text-green-600 dark:text-green-400">{progress}%</span>
            </div>
            {/* 进度条 */}
            <div className={`w-full h-3 ${progressBarBg} rounded-full overflow-hidden`}>
              <div
                className={`h-full ${progressBarFill} rounded-full transition-all duration-500 ease-out`}
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* 识别按钮 */}
        <button
          onClick={handleRecognize}
          disabled={!selectedVideo || isProcessing}
          className={`w-full bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white py-3 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 ${!selectedVideo || isProcessing ? "opacity-50 cursor-not-allowed" : "transform hover:scale-[1.02]"}`}
        >
          {isProcessing ? (
            <>
              <svg className="h-5 w-5 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>{outputVideo ? `处理中 ${progress}%` : "识别中..."}</span>
            </>
          ) : (
            <>
              <Video className="h-5 w-5" />
              <span>开始识别</span>
            </>
          )}
        </button>

        {/* 识别结果 */}
        {(results.length > 0 || resultVideoUrl) && (
          <div className="mt-6 space-y-4">
            {/* 标注结果视频播放器 */}
            {resultVideoUrl && (
              <div className={`rounded-lg overflow-hidden border ${darkMode ? "border-gray-700" : "border-gray-200"}`}>
                <div className={`px-4 py-2 ${darkMode ? "bg-gray-700" : "bg-gray-100"} flex items-center justify-between`}>
                  <div className="flex items-center space-x-2">
                    <Video className="h-4 w-4 text-purple-500" />
                    <span className="text-sm font-medium">标注结果视频</span>
                  </div>
                  <a
                    href={resultVideoUrl}
                    download
                    className="flex items-center space-x-1 px-3 py-1 text-xs font-medium bg-purple-500 text-white rounded hover:bg-purple-600 transition-colors"
                  >
                    <Download className="h-3 w-3" />
                    <span>下载</span>
                  </a>
                </div>
                <div className="bg-black">
                  <video
                    src={resultVideoUrl}
                    controls
                    className="w-full max-h-80"
                    preload="metadata"
                  >
                    您的浏览器不支持视频播放
                  </video>
                </div>
              </div>
            )}

            {/* 识别结果列表 */}
            <div>
              <h3 className="text-sm font-medium mb-3">识别结果 ({results.length} 帧)</h3>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {results.map((result, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-lg border ${darkMode ? "bg-gray-700 border-gray-600" : "bg-gray-50 border-gray-200"} transition-all hover:border-green-400 hover:shadow-sm`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${darkMode ? "bg-green-900/60 text-green-300" : "bg-green-100 text-green-700"}`}>
                          #{index + 1}
                        </span>
                        <Clock className={`h-3.5 w-3.5 ${darkMode ? "text-gray-400" : "text-gray-500"}`} />
                        <span className="text-sm font-medium">第 {result.frame_number} 帧</span>
                        <span className={`text-xs ${darkMode ? "text-gray-400" : "text-gray-500"}`}>({result.timestamp.toFixed(2)}s)</span>
                      </div>
                      {result.roles.length > 0 ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {result.roles.length > 0 ? (
                        result.roles.map((role, rIndex) => (
                          <span
                            key={rIndex}
                            className={`px-2 py-1 rounded-full text-xs font-medium ${
                              role.similarity > 0.8
                                ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                                : role.similarity > 0.5
                                ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                                : "bg-gray-100 text-gray-800 dark:bg-gray-600 dark:text-gray-200"
                            }`}
                          >
                            {role.role} ({(role.similarity * 100).toFixed(0)}%)
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-gray-500">未检测到角色</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}